import os
import subprocess
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import esm
from typing import Callable, Union, Tuple, Optional,List,Dict
from torch import Tensor
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
)
from kan import KAN
from torch.utils.data import DataLoader, Subset
import gc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit

# ====================  序列清洗与文件 IO ====================
def clean_and_filter_seqs(sequences: List[str]) -> List[str]:
    """过滤异常氨基酸并限制长度在 5-50 之间"""
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    cleaned = []
    for seq in sequences:
        if not isinstance(seq, str):
            continue
        seq = seq.upper().strip()
        if 5 <= len(seq) <= 50 and all(char in valid_chars for char in seq):
            cleaned.append(seq)
    return cleaned
def write_to_fasta(sequences: List[str], file_path: str, prefix: str):
    """序列写入 FASTA 格式"""
    with open(file_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}_{i}\n{seq}\n")
def read_from_fasta(file_path: str) -> List[str]:
    """从 FASTA 读取纯序列"""
    sequences = []
    with open(file_path, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.strip()
        if seq:
            sequences.append(seq)
    return sequences
# ==================== PyTorch Dataset 与 Collate ====================
class FeatureDataset(Dataset):
    def __init__(self, features, masks, labels, bac_ids):
        self.features = features.cpu()
        self.masks = masks.cpu()
        self.labels = labels.cpu()
        self.bac_ids = bac_ids.cpu()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.labels[idx],self.bac_ids[idx]


def feature_collate_fn(batch):
    # 已经是 Tensor 了，直接堆叠
    features = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    bac_ids = torch.stack([item[3] for item in batch])
    return features, masks, labels, bac_ids


def extract_global_features(all_sequences, esm_model_name, device):
    """统一长度填充，解决 torch.cat 维度不匹配问题"""
    print(f"\n [Global Feature Caching] 正在编码全量数据 (共 {len(all_sequences)} 条)...")
    encoder = ESMEncoderWrapper(model_name=esm_model_name, device=device, pooling_strategy="none")
    all_embs = []
    all_masks = []
    ext_batch_size = 64
    # 统一填充到 50（限制了最大长度为 50）
    # 考虑到 ESM-2 会自动加 <cls> 和 <eos>，实际上 tokens 长度会是 50 + 2 = 52
    FIXED_LEN = 52
    with torch.no_grad():
        for i in tqdm(range(0, len(all_sequences), ext_batch_size), desc="ESM 编码进度"):
            batch_seqs = all_sequences[i: i + ext_batch_size]
            # 1. 提取原始 tokens
            _, _, batch_tokens = encoder.batch_converter([(f"s{j}", s) for j, s in enumerate(batch_seqs)])
            # [核心修改]：创建一个全为 padding_idx 的固定长度 Tensor
            # 然后把当前 batch 的内容填进去，确保每批出来的长度都是一样的
            curr_batch_size, curr_len = batch_tokens.shape
            fixed_tokens = torch.full((curr_batch_size, FIXED_LEN), encoder.alphabet.padding_idx, dtype=torch.long)
            # 将实际内容拷贝进去（截断或填充）
            fixed_tokens[:, :curr_len] = batch_tokens
            fixed_tokens = fixed_tokens.to(device)
            mask = (fixed_tokens != encoder.alphabet.padding_idx).float().cpu()
            # 2. 提取特征向量 [Batch, 52, 1280]
            results = encoder.model(
                fixed_tokens,
                repr_layers=[encoder.num_layers],
                return_contacts=False
            )
            emb = results["representations"][encoder.num_layers].cpu()
            all_embs.append(emb)
            all_masks.append(mask)
    del encoder
    torch.cuda.empty_cache()
    print(f" 全量特征提取完成！特征矩阵形状: {torch.cat(all_embs).shape}")
    return torch.cat(all_embs).cpu(), torch.cat(all_masks).cpu()

## ==================== ESM编码器定义====================
class ESMEncoderWrapper(nn.Module):
    """
    冻结参数的预训练ESM编码器封装
    """
    def __init__(self,
                 model_name: str = "esm2_t33_650M_UR50D",
                 device: str = "cuda",
                 pooling_strategy: str = "none",
                 max_seq_len: int = 1000):#esm最多处理1024个氨基酸的肽
        """
        初始化ESM编码器
        参数:
            model_name: ESM预训练模型名称
            device: 计算设备
            pooling_strategy: 池化策略 ("mean", "cls", "none")
            max_seq_len: 最大序列长度（用于位置编码）
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pooling_strategy = pooling_strategy
        self.max_seq_len = max_seq_len
        # 加载预训练ESM模型
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 设置为评估模式
        self.model.eval()
        # 创建批次转换器
        self.batch_converter = self.alphabet.get_batch_converter()
        # 获取模型维度信息
        self.hidden_size = self.model.embed_dim
        self.num_layers = self.model.num_layers
        print(f" 加载ESM模型: {model_name}")
        print(f"  隐藏层维度: {self.hidden_size}")
        print(f"  层数: {self.num_layers}")
        print(f"  池化策略: {pooling_strategy}")
        print(f"  设备: {self.device}")
    def forward(self, sequences: Union[str, List[str]]) -> Tensor:
        """
        编码氨基酸序列为固定维度的嵌入
        参数:
            sequences: 单个序列字符串或序列列表
        返回:
            形状为 [batch_size, hidden_size] 或 [batch_size, seq_len, hidden_size] 的张量
        """
        if isinstance(sequences, torch.Tensor):
            # 如果已经是Tensor，直接返回（避免重复编码）
            print("警告: ESM编码器接收到Tensor输入，可能存在重复编码")
            return sequences
        if isinstance(sequences, str):
            sequences = [sequences]
        # 准备输入数据
        batch_labels, batch_strs, batch_tokens = self.batch_converter([
            (f"seq_{i}", seq) for i, seq in enumerate(sequences)
        ])
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():  # 确保不计算梯度
            # 获取模型输出
            results = self.model(
                batch_tokens,
                repr_layers=[self.num_layers],  # 只取最后一层
                need_head_weights=False,
                return_contacts=False
            )
            # 提取最后一层的表示 [batch_size, seq_len, hidden_size]
            token_representations = results["representations"][self.num_layers]
        # 应用池化策略
        if self.pooling_strategy == "cls":
            # 使用CLS token（第一个token）
            embeddings = token_representations[:, 0, :]  # [batch_size, hidden_size]
        elif self.pooling_strategy == "mean":
            # 平均池化（排除填充token）
            attention_mask = (batch_tokens != self.alphabet.padding_idx).float()
            # 扩展掩码维度以匹配隐藏层维度
            attention_mask = attention_mask.unsqueeze(-1).expand_as(token_representations)
            # 计算有效token的平均值
            sum_embeddings = (token_representations * attention_mask).sum(dim=1)
            num_valid_tokens = attention_mask.sum(dim=1)
            embeddings = sum_embeddings / num_valid_tokens.clamp(min=1e-8)
        else:  # "none"
            # 返回所有token的表示
            embeddings = token_representations
        return embeddings
    def encode_sequence(self, sequences: str) -> Tensor:
        """编码单个序列"""
        return self.forward(sequences)
    def encode_batch(self, sequences: List[str]) -> Tensor:
        """批量编码序列"""
        return self.forward(sequences)
    def get_embedding_dim(self) -> int:
        """返回嵌入维度"""
        return self.hidden_size
    def get_max_seq_len(self) -> int:
        """返回支持的最大序列长度"""
        return self.max_seq_len
    def validate_sequence(self, sequence: str) -> bool:
        """验证序列是否包含有效氨基酸字符"""
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        return all(char in valid_chars for char in sequence.upper())
## ==================== ESM编码器定义====================


class ELK_Regressor(nn.Module):
    # 维度从 1284 升级到 1312 (1280 ESM + 32 Bac_Emb)
    def __init__(self, esm_dim: int = 1280, bac_emb_dim: int = 32,kan_hidden_dim: int = 64):
        super().__init__()
        # 1. 细菌特征的动态字典
        self.bac_embedding = nn.Embedding(num_embeddings=4, embedding_dim=bac_emb_dim)
        d_model = esm_dim + bac_emb_dim  # = 1312

        # 1. 轻量化 xLSTM
        slstm_backend = "vanilla"
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=32, num_heads=4)),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(backend=slstm_backend, num_heads=4, conv1d_kernel_size=4)),
            context_length=256, num_blocks=2, embedding_dim=d_model, slstm_at=[1]
        )
        self.xlstm = xLSTMBlockStack(xlstm_config)
        # 2. 决策层 KAN
        #  输出维度依然是 1。在回归任务中，这 1 个神经元直接输出 MIC 数值！
        self.dropout = nn.Dropout(p=0.1)
        self.kan = KAN(layers_hidden=[d_model, kan_hidden_dim, 1])
    def forward(self, esm_embeddings: torch.Tensor, padding_mask: torch.Tensor, bac_ids: torch.Tensor) -> torch.Tensor:
        # 1. 查表：拿到当前 Batch 对应细菌的 32 维专属密码 -> [Batch, 32]
        bac_emb = self.bac_embedding(bac_ids)
        # 2. 把这个密码广播到多肽的每一个氨基酸旁边 -> [Batch, Seq_Len, 32]
        seq_len = esm_embeddings.size(1)
        bac_emb_expanded = bac_emb.unsqueeze(1).expand(-1, seq_len, -1)
        # 3. 拼接：1280 + 32 = 1312 维特征
        fused_features = torch.cat([esm_embeddings, bac_emb_expanded], dim=-1)

        xlstm_out = self.xlstm(fused_features)
        expanded_mask = padding_mask.unsqueeze(-1)
        masked_embeddings = xlstm_out * expanded_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_lengths = padding_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled_features = sum_embeddings / valid_lengths
        pooled_features = self.dropout(pooled_features)
        logits = self.kan(pooled_features)
        return logits


def calculate_regression_metrics(y_true, y_pred):
    """
    计算回归任务：PCC, R2, MSE, MAE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # 1. MSE & RMSE (均方误差)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # 直接开根号就是 RMSE
    # 2. MAE (平均绝对误差)
    mae = mean_absolute_error(y_true, y_pred)
    # 3. R2 (决定系数)
    r2 = r2_score(y_true, y_pred)
    # 4. PCC (皮尔逊相关系数)
    # pearsonr 返回 (statistic, pvalue)，只需要统计量
    if len(y_true) > 1:
        pcc, _ = pearsonr(y_true, y_pred)
    else:
        pcc = 0.0
    return pcc, r2, mse,rmse,mae


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """回归任务的单轮训练"""
    model.train()
    epoch_loss = 0.0
    all_targets, all_preds = [], []  # 变量名从 probs 改为 preds (预测值)
    pbar = tqdm(dataloader, desc=" Train", leave=False)
    for embs, masks, targets,bac_ids in pbar:
        embs, masks, targets, bac_ids = embs.to(device), masks.to(device), targets.to(device), bac_ids.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(embs, masks, bac_ids)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        #  直接把 logits (数值) 存进列表
        all_preds.extend(logits.detach().float().cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    avg_loss = epoch_loss / len(dataloader)
    # 接收回归的 5 个指标
    pcc, r2, mse, rmse, mae = calculate_regression_metrics(all_targets, all_preds)
    return avg_loss, pcc, r2, mse, rmse, mae


def evaluate(model, dataloader, criterion, device, phase="Valid"):
    """回归任务的评估模块"""
    model.eval()
    epoch_loss = 0.0
    all_targets, all_preds = [], []
    with torch.no_grad():
        for embs, masks, targets, bac_ids in tqdm(dataloader, desc=f" {phase}", leave=False):
            embs, masks, targets, bac_ids = embs.to(device), masks.to(device), targets.to(device), bac_ids.to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(embs, masks, bac_ids)
                loss = criterion(logits, targets)
            epoch_loss += loss.item()
            all_preds.extend(logits.float().cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    avg_loss = epoch_loss / len(dataloader)
    pcc, r2, mse, rmse, mae = calculate_regression_metrics(all_targets, all_preds)
    # 最后把全量的真实值和预测值返回去画散点图
    return avg_loss, pcc, r2, mse, rmse, mae, all_targets, all_preds


# ==========================================
# 统揽全局的完整训练大循环
# ==========================================
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=100, fold_idx=1):
    """
    回归任务的完整多 Epoch 训练引擎，基于 Valid MSE 寻找最佳模型
    """
    # 更新画图历史记录的字段
    history = {'train_loss': [], 'valid_loss': [], 'valid_mse': [], 'valid_rmse': [], 'valid_pcc': []}
    #  [核心修改1]：寻找最小的 MSE，所以初始记录要设为无穷大 (infinity)
    best_valid_mse = float('inf')
    best_model_path = f"best_mic_model_fold_{fold_idx}.pth"
    scaler = torch.amp.GradScaler('cuda')
    print(f"\n 开始 Fold {fold_idx} 的训练，总计 {num_epochs} Epochs...")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        # 1. 训练 (接收回归的 6 个返回值)
        t_loss, t_pcc, t_r2, t_mse, t_rmse, t_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        # 2. 验证 (接收回归的 8 个返回值)
        v_loss, v_pcc, v_r2, v_mse, v_rmse, v_mae, _, _ = evaluate(
            model, valid_loader, criterion, device, phase="Valid")
        # 记录历史数据用于画图
        history['train_loss'].append(t_loss)
        history['valid_loss'].append(v_loss)
        history['valid_mse'].append(v_mse)
        history['valid_rmse'].append(v_rmse)
        history['valid_pcc'].append(v_pcc)
        # 打印面板：回归四大金刚全部就位
        print(
            f" Train | Loss: {t_loss:.4f} | MSE: {t_mse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f} | PCC: {t_pcc:.4f} | R2: {t_r2:.4f}")
        print(
            f" Valid | Loss: {v_loss:.4f} | MSE: {v_mse:.4f} | RMSE: {v_rmse:.4f} | MAE: {v_mae:.4f} | PCC: {v_pcc:.4f} | R2: {v_r2:.4f}")
        #  保存最佳模型 (看 v_mse 是不是比历史最小值还要小)
        if v_mse < best_valid_mse:
            best_valid_mse = v_mse
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, best_model_path))
            print(f"  [新记录] Valid MSE 降低至 {v_mse:.4f} (RMSE: {v_rmse:.4f}, PCC: {v_pcc:.4f}),MAE: {v_mae:.4f} ，模型已保存！")
    print(f" Fold {fold_idx} 训练结束！最佳 Valid MSE: {best_valid_mse:.4f}")
    return history, best_model_path



# 顶刊级可视化画图工具
# ==========================================
def plot_learning_curves(history, fold_idx, save_dir="C:/Users/Mordred/Desktop/MIC_Output_Results/plot"):
    """绘制 Train Loss 和 Valid Loss 曲线，检查是否过拟合"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['valid_loss'], 'r--', label='Valid Loss', linewidth=2)
    plt.title(f'Learning Curves', fontsize=16)
    #plt.title(f'Learning Curves - Fold {fold_idx}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/PMIC_learning_curve_fold_{fold_idx}.png", dpi=300)
    plt.close()


def plot_regression_scatter(y_true, y_pred, fold_idx, pcc_score, save_dir="C:/Users/Mordred/Desktop/MIC_Output_Results/Plot"):
    """绘制真实值与预测值的回归散点图"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    # 绘制散点，透明度设为0.6方便看密集区域
    plt.scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', edgecolors='k',
                label=f'Predictions (PCC = {pcc_score:.4f})')
    # 获取坐标轴的范围，用来画完美的 y=x 对角线
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    # 绘制 y=x 理想红线 (如果点都在这条线上)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')
    plt.xlabel('True Log10(MIC)', fontsize=14)
    plt.ylabel('Predicted Log10(MIC)', fontsize=14)
    plt.title(f'Regression Scatter Plot ', fontsize=16)
    #plt.title(f'Regression Scatter Plot - Fold {fold_idx}', fontsize=16)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/regression_scatter_fold_{fold_idx}.png", dpi=300)
    plt.close()


######################main###########################

# ---  模型架构配置 ---
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
KAN_HIDDEN_DIM = 64
# ---  训练核心超参数 ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100  #测试设置1，训练设置200
K_FOLDS = 5#5折或者10折
# True: 只跑第1折就结束
# False: 完整的跑完 5 折交叉验证
RUN_ONE_FOLD_ONLY = True
# True: WeightedRandomSampler (解决数据不平衡)
# False: 启用普通洗牌 shuffle=True (保持原始真实分布)
USE_WEIGHTED_SAMPLER = True

# 自动检测算力设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================
#  2. 模型训练与验证引擎 (Train Mode)
# ==========================================
def run_train():
    print("\n" + "=" * 50)
    print(f"  [TRAIN MODE] 启动：条件输入大一统 MIC 回归预测")
    print("=" * 50)
    # --- 第一步：读取大一统数据 ---
    # 假设您的 Excel 包含这三列: 'Sequence', 'Log10_MIC', 'Bacteria_ID'
    if not os.path.exists(MASTER_MIC_EXCEL_PATH):
        print(f"找不到数据文件: {MASTER_MIC_EXCEL_PATH}，请检查路径！")
        return
    df = pd.read_excel(MASTER_MIC_EXCEL_PATH)

    # 1. 简单清洗一下，确保没有空值
    df = df.dropna(subset=['Sequence', 'Log10_MIC', 'Bacteria_ID'])
    # [新增安检门]：严格过滤 5-50 长度，且只允许包含 20 种标准氨基酸！
    df['Sequence'] = df['Sequence'].astype(str).str.upper().str.strip()  # 统一转大写并去空格
    df['seq_len'] = df['Sequence'].apply(len)
    # 长度筛选：5-50
    df = df[(df['seq_len'] >= 5) & (df['seq_len'] <= 50)]
    # 字符筛选：用正则表达式确保只有这 20 个合法字母
    df = df[df['Sequence'].str.contains(r'^[ACDEFGHIKLMNPQRSTVWY]+$', regex=True)]
    print(f" 数据清洗完毕！保留了 {len(df)} 条纯净的 (5-50) 氨基酸序列。")
    all_sequences = df['Sequence'].tolist()
    all_targets = df['Log10_MIC'].astype(float).tolist()
    all_bacteria_ids = df['Bacteria_ID'].astype(int).tolist()
    # 目标值转为 Tensor
    all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32).view(-1, 1)
    # --- 第二步：全量特征预提取 (ESM-2) ---
    global_embs, global_masks = extract_global_features(all_sequences, ESM_MODEL_NAME, DEVICE)
    global_embs = global_embs.half()  # 保持 1280 维不变
    # 只需要把细菌标签转成 Tensor 备用
    all_bacteria_tensor = torch.tensor(all_bacteria_ids, dtype=torch.long)

    gc.collect()
    torch.cuda.empty_cache()
    print(f" 当前输入矩阵维度: {global_embs.shape} (1280 + 4)")
    # --- 第三步：绝对隔离独立测试集 (基于序列的防泄露划分) ---
    groups = np.array(all_sequences)  # 以序列作为分组依据
    indices = np.arange(len(all_sequences))
    # 使用 GroupShuffleSplit，保证同一个序列的不同细菌数据，绝对不会跨界！
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    cv_idx, test_idx = next(gss.split(indices, all_targets, groups))

    test_df = df.iloc[test_idx].copy()
    test_excel_path = os.path.join(OUTPUT_DIR, "Independent_Test_Set.xlsx")
    test_df.to_excel(test_excel_path, index=False)
    print(f"🛡 独立测试集已存为: {test_excel_path}")


    test_feat_dataset = FeatureDataset(global_embs[test_idx], global_masks[test_idx], all_targets_tensor[test_idx],all_bacteria_tensor[test_idx])
    test_loader = DataLoader(test_feat_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=feature_collate_fn)
    print(f"🛡 已隔离独立测试集: {len(test_idx)} 条 (基于序列分组，绝不泄露)")
    # --- 第四步：Group K 折交叉验证 ---
    # 使用 GroupKFold 替换原先的 StratifiedKFold
    gkf = GroupKFold(n_splits=K_FOLDS)
    fold_results_pcc = []
    y_cv = [all_targets[i] for i in cv_idx]
    groups_cv = [groups[i] for i in cv_idx]
    for fold_idx, (train_idx_in_cv, valid_idx_in_cv) in enumerate(gkf.split(cv_idx, y_cv, groups_cv)):
        print(f"\n" + "-" * 30 + f" 开始第 {fold_idx + 1} 折 " + "-" * 30)
        actual_train_idx = cv_idx[train_idx_in_cv]
        actual_valid_idx = cv_idx[valid_idx_in_cv]

        fold_train_df = df.iloc[actual_train_idx].copy()
        fold_valid_df = df.iloc[actual_valid_idx].copy()

        # 2. 拼装保存路径 (存入 OUTPUT_DIR)
        train_save_path = os.path.join(OUTPUT_DIR, f"Fold_{fold_idx + 1}_Train_Set.xlsx")
        valid_save_path = os.path.join(OUTPUT_DIR, f"Fold_{fold_idx + 1}_Valid_Set.xlsx")

        # 3. 写入 Excel (index=False 防止多出一列没用的序号)
        fold_train_df.to_excel(train_save_path, index=False)
        fold_valid_df.to_excel(valid_save_path, index=False)

        print(
            f"  [数据溯源] Fold {fold_idx + 1} 训练集已存 ({len(fold_train_df)}条) -> {os.path.basename(train_save_path)}")
        print(
            f"  [数据溯源] Fold {fold_idx + 1} 验证集已存 ({len(fold_valid_df)}条) -> {os.path.basename(valid_save_path)}")



        if USE_WEIGHTED_SAMPLER:
            print("  [策略启用] 当前使用 WeightedRandomSampler 平衡稀有细菌数据。")
            train_bacteria_labels = [all_bacteria_ids[i] for i in actual_train_idx]
            class_counts = np.bincount(train_bacteria_labels)
            class_weights = 1.0 / (class_counts + 1e-8)
            sample_weights = [class_weights[label] for label in train_bacteria_labels]
            sample_weights = torch.DoubleTensor(sample_weights)

            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle_flag = False  # 用了 Sampler 必须关掉 shuffle
        else:
            print("  [策略启用] 当前使用普通 Shuffle，保持数据原始自然分布。")
            sampler = None
            shuffle_flag = True  # 不用 Sampler 就必须开启 shuffle



        train_loader = DataLoader(
            FeatureDataset(global_embs[actual_train_idx], global_masks[actual_train_idx],
                           all_targets_tensor[actual_train_idx],all_bacteria_tensor[actual_train_idx]),
            batch_size=BATCH_SIZE, sampler=sampler,shuffle=shuffle_flag, collate_fn=feature_collate_fn, pin_memory=True, num_workers=0
        )
        valid_loader = DataLoader(
            FeatureDataset(global_embs[actual_valid_idx], global_masks[actual_valid_idx],
                           all_targets_tensor[actual_valid_idx],all_bacteria_tensor[actual_valid_idx]),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=feature_collate_fn, pin_memory=True, num_workers=0
        )
        #  [极其关键] 实例化回归模型，输入维度必须是 1284
        model = ELK_Regressor( kan_hidden_dim=KAN_HIDDEN_DIM).to(DEVICE)
        # 1. 换用抗噪能力更强的 HuberLoss (平滑 L1)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        # 开始训练
        history, best_model_path = train_model(
            model=model, train_loader=train_loader, valid_loader=valid_loader,
            criterion=criterion, optimizer=optimizer, device=DEVICE,
            num_epochs=NUM_EPOCHS, fold_idx=fold_idx + 1,
        )
        # 画出 Loss 曲线
        plot_learning_curves(history, fold_idx=fold_idx + 1)
        # 加载最佳模型并测试
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, best_model_path)))

        # 1. 输出验证集指标
        _, v_pcc, v_r2, v_mse, v_rmse, v_mae, _, _ = evaluate(model, valid_loader, criterion, DEVICE, "Valid_Best")

        # 2. 输出独立测试集指标 (测试集 test_loader)
        _, t_pcc, t_r2, t_mse, t_rmse, t_mae, all_t, all_p = evaluate(model, test_loader, criterion, DEVICE,
                                                                      "Independent_Test")

        print("\n" + "=" * 30 + " 本折大考成绩单 " + "=" * 30)
        print(f"【验证集(Valid)】 PCC: {v_pcc:.4f} | R2: {v_r2:.4f} | MSE: {v_mse:.4f}| RMSE: {v_rmse:.4f}| MAE: {v_mae:.4f}")
        print(f"【测试集(Test) 】 PCC: {t_pcc:.4f} | R2: {t_r2:.4f} | MSE: {t_mse:.4f}| RMSE: {t_rmse:.4f}| MAE: {t_mae:.4f}")
        print("=" * 76)


        #  呼叫全新的散点图绘制函数
        plot_regression_scatter(all_t, all_p, fold_idx + 1, v_pcc)
        fold_results_pcc.append(v_pcc)
        gc.collect()
        torch.cuda.empty_cache()
        print(f" Fold {fold_idx + 1} 资源已彻底清理！")
        if RUN_ONE_FOLD_ONLY: break
    print("\n" + "=" * 50)
    print(f"  训练全部完成！平均 Valid PCC: {np.mean(fold_results_pcc):.4f}")
    print("=" * 50)



# ==========================================
#  3. 加载模型预测新序列 (Predict Mode)
# ==========================================
def run_predict(target_bacteria_id: int = 0):
    """
    参数:
        target_bacteria_id: 0(大肠杆菌), 1(金葡菌), 2(铜绿假单胞), 3(枯草芽孢)
    """
    print("\n" + "=" * 50)
    print(f"  [PREDICT MODE] 启动：靶向条件 MIC 回归预测")
    print("=" * 50)

    BACTERIA_NAMES = {
        0: "E. coli (大肠杆菌)",
        1: "S. aureus (金黄色葡萄球菌)",
        2: "P. aeruginosa (铜绿假单胞菌)",
        3: "B. subtilis (枯草芽孢杆菌)"
    }
    if target_bacteria_id not in BACTERIA_NAMES:
        print(f" 错误：细菌编号必须是 0, 1, 2, 3 中的一个！当前输入的是 {target_bacteria_id}")
        return
    target_name = BACTERIA_NAMES[target_bacteria_id]
    print(f"  当前锁定预测靶标: [{target_bacteria_id}] {target_name}")
    if not os.path.exists(NEW_PEPTIDES_PATH):
        print(f"找不到待测文件: {NEW_PEPTIDES_PATH}，请检查路径！")
        return
    print("  正在读取待测序列...")
    try:
        #  iloc[:, 1] 表示提取 Excel 的第二列作为序列
        #raw_sequences = pd.read_excel(NEW_PEPTIDES_PATH).iloc[:, 0].dropna().astype(str).tolist()
        raw_sequences = pd.read_excel(NEW_PEPTIDES_PATH, sheet_name=NEW_PEPTIDES_SHEET).iloc[:, 0].dropna().astype(
            str).tolist()
    except Exception as e:
        print(f" 读取 Excel 失败: {e}")
        return
    # --- 带有索引追踪的清洗逻辑 ---
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    valid_seqs = []
    valid_indices = []
    for idx, seq in enumerate(raw_sequences):
        seq_clean = seq.upper().strip()
        if 5 <= len(seq_clean) <= 50 and all(char in valid_chars for char in seq_clean):
            valid_seqs.append(seq_clean)
            valid_indices.append(idx)
    print(
        f"  数据检查完成：总计 {len(raw_sequences)} 条 | 合格送检 {len(valid_seqs)} 条 | 过滤 {len(raw_sequences) - len(valid_seqs)} 条")
    if len(valid_seqs) == 0:
        print(" 没有找到有效的多肽序列，预测终止。")
        return
    # --- 极速提取 ESM-2 特征 ---
    print("\n  正在提取 ESM-2 全局特征...")
    embs, masks = extract_global_features(valid_seqs, ESM_MODEL_NAME, DEVICE)
    embs = embs.half()  # 保持 1280 维不变！


    # 给送检序列打上参数传进来的细菌标签
    num_valid = len(valid_seqs)

    bacteria_tensor = torch.full((num_valid,), target_bacteria_id, dtype=torch.long)

    gc.collect()
    torch.cuda.empty_cache()
    # --- 构建 DataLoader ---
    dummy_labels = torch.zeros((num_valid, 1), dtype=torch.float32)
    predict_dataset = FeatureDataset(embs, masks, dummy_labels, bacteria_tensor)
    predict_loader = DataLoader(predict_dataset, batch_size=512, shuffle=False, collate_fn=feature_collate_fn,
                                pin_memory=True)
    # --- 加载回归模型 ---
    model_path = R"E:/Users/Mordred/Desktop/MIC_Output_Results\best_mic_model.pth"
    if not os.path.exists(model_path):
        print(f"  找不到模型权重文件: {model_path}，请先运行 train 模式！")
        return
    print(f"\n  正在加载回归模型权重: {model_path} ...")
    model = ELK_Regressor(kan_hidden_dim=KAN_HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    # --- 开始推理 ---
    print(" 开始批量推理...")
    valid_preds = []
    with torch.no_grad():
        for batch_embs, batch_masks, _ , batch_bac_ids in tqdm(predict_loader, desc=" 预测进度"):
            batch_embs, batch_masks ,batch_bac_ids= batch_embs.to(DEVICE), batch_masks.to(DEVICE),batch_bac_ids.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(batch_embs, batch_masks, batch_bac_ids)
            valid_preds.extend(logits.float().cpu().numpy().flatten())
    # --- 还原完整结果列表并导出 ---
    print("\n  正在生成最终预测报告...")
    final_log10_mic = ["Invalid (过滤)"] * len(raw_sequences)
    final_raw_mic = ["N/A"] * len(raw_sequences)

    #  新增多个维度的统计计数器
    mic_under_5_count = 0
    mic_under_10_count = 0
    mic_under_20_count = 0

    for v_idx, pred in zip(valid_indices, valid_preds):
        final_log10_mic[v_idx] = round(float(pred), 4)
        raw_val = 10 ** float(pred)

        if pred > 6:
            final_raw_mic[v_idx] = ">1000000 (极弱/无活性)"
        else:
            final_raw_mic[v_idx] = round(raw_val, 4)
            # 梯度统计
            if raw_val < 5:
                mic_under_5_count += 1
            if raw_val < 10:
                mic_under_10_count += 1
            if raw_val < 20:
                mic_under_20_count += 1

    results_df = pd.DataFrame({
        "Original_Sequence": raw_sequences,
        "Target_Bacteria": [target_name] * len(raw_sequences),
        "Predicted_Log10_MIC": final_log10_mic,
        "Predicted_MIC_Value": final_raw_mic
    })
    output_filename = f"Predicted_MIC_Results_Target_{target_bacteria_id}.xlsx"
    results_df.to_excel(output_filename, index=False)

    #  计算三个梯度的占比
    total_count = len(raw_sequences)
    ratio_5 = (mic_under_5_count / total_count) * 100
    ratio_10 = (mic_under_10_count / total_count) * 100
    ratio_20 = (mic_under_20_count / total_count) * 100

    print("=" * 50)
    print("  预测完成")
    print(f"  结果已保存至: {output_filename}")
    print("-" * 50)
    print(f"  统计信息: 输入总序列 {total_count} 条")
    print(f"  顶级杀手 (MIC < 5)  : {mic_under_5_count} 条，占比 {ratio_5:.2f}%")
    print(f" 强效序列 (MIC < 10) : {mic_under_10_count} 条，占比 {ratio_10:.2f}%")
    print(f"  优良序列 (MIC < 20) : {mic_under_20_count} 条，占比 {ratio_20:.2f}%")
    print("=" * 50)





#  核心开关：在这里切换任务模式！
RUN_MODE = "predict"  # 可选: "train" (训练模型) 或 "predict" (预测新序列)
# ---  文件路径配置 ---
OUTPUT_DIR = "E:/Users/Mordred/Desktop/MIC_Output_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MASTER_MIC_EXCEL_PATH = "E:/Users/Mordred/Desktop/grampa_mic.xlsx"  # 样本数据
NEW_PEPTIDES_PATH = r"E:\Users\Mordred\Desktop\Finetune_Output\BS.xlsx"  # 未来给 predict 模式留的待测数据
NEW_PEPTIDES_SHEET = "5"  #表格名称


# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"  当前计算设备: {DEVICE}")
    print(f" ⚙ 当前执行模式: {RUN_MODE}")
    print("=" * 60)
    if RUN_MODE == "train":
        run_train()
    elif RUN_MODE == "predict":
        # 在这里直接传递细菌编号！
        # 0:大肠杆菌, 1:金葡菌, 2:铜绿假单胞, 3:枯草芽孢
        TARGET_BACTERIA = 3
        run_predict(target_bacteria_id=TARGET_BACTERIA)
        # 如果想一键测试四种细菌，只需要写个循环：
        # for i in range(4):
        #     run_predict(target_bacteria_ibd=i)
    else:
        print("❌ 错误：RUN_MODE 必须是 'train' 或 'predict'！")


