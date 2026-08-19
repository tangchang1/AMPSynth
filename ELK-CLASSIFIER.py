import os
import subprocess
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import esm
from typing import Callable, Union, Tuple, Optional,List,Dict
from torch import Tensor
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
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

# ====================自动化去冗余 ====================
def run_cdhit(input_fasta: str, output_fasta: str):
    """这里使用 Python 字典进行 100% 序列精确去重。
    """
    print(f"Python 执行 100% 精确去重")
    unique_seqs = set()
    # 1. 纯手工读取 FASTA
    with open(input_fasta, 'r') as f:
        seq = ""
        for line in f:
            if line.startswith(">"):
                if seq:
                    unique_seqs.add(seq)
                    seq = ""
            else:
                seq += line.strip()
        if seq:
            unique_seqs.add(seq)
    # 2. 将绝对不重复的序列写回新的 FASTA
    with open(output_fasta, 'w') as f:
        for i, sq in enumerate(list(unique_seqs)):
            f.write(f">seq_{i}\n{sq}\n")
    print(f" 提取出 {len(unique_seqs)} 条唯一序列。")

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
    def __init__(self, features, masks, labels):
        self.features = features.cpu()
        self.masks = masks.cpu()
        self.labels = labels.cpu()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.labels[idx]


def feature_collate_fn(batch):
    # 已经是 Tensor 了，直接堆叠
    features = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return features, masks, labels


def extract_global_features(all_sequences, esm_model_name, device):
    """统一长度填充"""
    print(f"\n [Global Feature Caching] 正在编码全量数据 (共 {len(all_sequences)} 条)...")
    encoder = ESMEncoderWrapper(model_name=esm_model_name, device=device, pooling_strategy="none")
    all_embs = []
    all_masks = []
    ext_batch_size = 64
    # 统一填充到 50
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


class ELK_Classifier(nn.Module):
    def __init__(self, d_model: int = 1280, kan_hidden_dim: int = 64):
        super().__init__()
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
        self.dropout = nn.Dropout(p=0.3)
        # 2. 决策层 KAN
        self.kan = KAN(layers_hidden=[d_model, kan_hidden_dim, 1])
    def forward(self, esm_embeddings: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # 现在的 forward 只需要纯粹的特征矩阵计算
        xlstm_out = self.xlstm(esm_embeddings)
        expanded_mask = padding_mask.unsqueeze(-1)
        masked_embeddings = xlstm_out * expanded_mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_lengths = padding_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
        pooled_features = sum_embeddings / valid_lengths
        pooled_features = self.dropout(pooled_features)
        logits = self.kan(pooled_features)
        return logits


def calculate_metrics(y_true, y_probs):
    """
    计算：Acc, Sn (Sensitivity), Sp (Specificity), MCC, AUC
    """
    y_pred = (np.array(y_probs) >= 0.5).astype(int)
    y_true = np.array(y_true)
    # 1. Accuracy (准确率)
    acc = accuracy_score(y_true, y_pred)
    # 2. AUC (曲线下面积)
    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc_score = 0.5
    # 3. MCC (马修斯相关系数)
    mcc = matthews_corrcoef(y_true, y_pred)
    # --- 防崩盘写法：提取混淆矩阵 ---
    # 强制指定 labels=[0, 1]，防止某一批次全是正样本或全负样本导致解包失败
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # 4. Sensitivity (Sn / 灵敏度) - 防止分母为 0
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # 5. Specificity (Sp / 特异性) - 防止分母为 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, sn, sp, mcc, auc_score


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """适配 FeatureDataset + AMP 混合精度加速的单轮训练"""
    model.train()
    epoch_loss = 0.0
    all_targets, all_probs = [], []
    pbar = tqdm(dataloader, desc=" Train", leave=False)
    for embs, masks, targets in pbar:
        embs, masks, targets = embs.to(device), masks.to(device), targets.to(device)
        optimizer.zero_grad()
        #  [核心提速]：开启自动混合精度计算
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(embs, masks)
            loss = criterion(logits, targets)
        # 放大梯度进行反向传播，防止 FP16 下梯度下溢
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = epoch_loss / len(dataloader)
    acc, sn, sp, mcc, auc_score = calculate_metrics(all_targets, all_probs)
    return avg_loss, acc, sn, sp, mcc, auc_score


def evaluate(model, dataloader, criterion, device, phase="Valid"):
    """适配 FeatureDataset 的评估 (混合精度支持)"""
    model.eval()
    epoch_loss = 0.0
    all_targets, all_probs = [], []

    with torch.no_grad():
        for embs, masks, targets in tqdm(dataloader, desc=f" {phase}", leave=False):
            embs, masks, targets = embs.to(device), masks.to(device), targets.to(device)
            #   验证阶段也必须开启混合精度，匹配 .half() 压缩后的特征
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(embs, masks)
                loss = criterion(logits, targets)
            epoch_loss += loss.item()
            # 注意这里要把 logits 转回 float32 再过 sigmoid，防止精度溢出
            all_probs.extend(torch.sigmoid(logits.float()).cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = epoch_loss / len(dataloader)
    acc, sn, sp, mcc, auc_score = calculate_metrics(all_targets, all_probs)
    return avg_loss, acc, sn, sp, mcc, auc_score, all_targets, all_probs


# ==========================================
# 统揽全局的完整训练大循环
# ==========================================
def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=100, fold_idx=1):
    """
    完整的多 Epoch 训练引擎，带有早停机制 (基于 Valid AUC) 和最佳模型保存
    """
    history = {'train_loss': [], 'valid_loss': [], 'valid_auc': [], 'valid_mcc': []}
    best_valid_auc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, f"best_Classifier_model_fold_{fold_idx}.pth")
    #  初始化 AMP 梯度缩放器
    scaler = torch.amp.GradScaler('cuda')
    patience = 10
    patience_counter = 0
    print(f"\n 开始 Fold {fold_idx} 的训练，总计 {num_epochs} Epochs...")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        #  记得把 scaler 传进函数里
        t_loss, t_acc, t_sn, t_sp, t_mcc, t_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        # 2. 验证
        v_loss, v_acc, v_sn, v_sp, v_mcc, v_auc, _, _ = evaluate(model, valid_loader, criterion, device, phase="Valid")
        # 记录历史数据用于画图
        history['train_loss'].append(t_loss)
        history['valid_loss'].append(v_loss)
        history['valid_auc'].append(v_auc)
        history['valid_mcc'].append(v_mcc)
        # 打印面板：黄金五小强全部就位
        print(
            f" Train | Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | Sn: {t_sn:.4f} | Sp: {t_sp:.4f} | MCC: {t_mcc:.4f} | AUC: {t_auc:.4f}")
        print(
            f" Valid | Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | Sn: {v_sn:.4f} | Sp: {v_sp:.4f} | MCC: {v_mcc:.4f} | AUC: {v_auc:.4f}")
        # 3. 保存最佳模型 (以 Valid AUC 为最高优先级)
        # 3. 保存最佳模型
        if v_auc > best_valid_auc:
            best_valid_auc = v_auc
            torch.save(model.state_dict(), best_model_path)
            print(f" [新记录] Valid AUC 提升至 {v_auc:.4f}，模型已保存！")
            patience_counter = 0  #  有提升就清零计数器
        else:
            patience_counter += 1  #  没提升就涨计数器
            print(f"  [早停监控] AUC 未提升 ({patience_counter}/{patience})")
        #  触发早停
        if patience_counter >= patience:
            print(f"\n  [触发早停] 连续 {patience} 个 Epoch Valid AUC 未提升，提前结束本折训练，防止 Loss 过拟合爆炸！")
            break
    return history, best_model_path
# ==========================================
def predict_new_sequence(model, new_sequence, device):
    """
    用训练好的模型预测一条未知的新序列
    """
    model.eval()  # 必须关掉训练模式
    with torch.no_grad():  # 预测时不算梯度，省显存且速度极快
        # 1. 模型吐出原始 Logits (比如 3.2)
        logits = model([new_sequence])
        # 2. 核心！手动加上 Sigmoid 把 Logits 变成 0~1 的概率！(比如 0.96)
        probability = torch.sigmoid(logits).item()
        # 3. 按照 0.5 划定及格线，给出最终结论
        is_amp = True if probability >= 0.5 else False
        print(f"序列: {new_sequence}")
        print(f"模型确信度 (概率): {probability * 100:.2f}%")
        print(f"最终判定: {' 是抗菌肽 (AMP)' if is_amp else ' 不是抗菌肽 (Non-AMP)'}")
    return probability, is_amp

# 可视化画图工具
# ==========================================
def plot_learning_curves(history, fold_idx, save_dir="plots"):
    """绘制 Train Loss 和 Valid Loss 曲线，检查是否过拟合"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['valid_loss'], 'r--', label='Valid Loss', linewidth=2)
    plt.title(f'Learning Curves - Fold {fold_idx}', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curve_fold_{fold_idx}.png", dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_probs, fold_idx, auc_score, save_dir="plots"):
    """绘制高质量 ROC 曲线"""
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 对角线，代表瞎猜
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title(f'Receiver Operating Characteristic - Fold {fold_idx}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curve_fold_{fold_idx}.png", dpi=300)
    plt.close()



# ==========================================
#  2. 兵工厂：模型训练与验证引擎 (Train Mode)
# ==========================================
def run_train():
    print("\n" + "=" * 50)
    print(f" [TRAIN MODE] 启动")
    print("=" * 50)
    # --- 第一步：原始数据读取与去重 (不直接封装 Loader) ---
    os.makedirs("./temp_data", exist_ok=True)
    amp_raw = pd.read_excel(AMP_EXCEL_PATH).iloc[:, 1].dropna().astype(str).tolist()
    non_amp_raw = pd.read_excel(NON_AMP_EXCEL_PATH).iloc[:, 1].dropna().astype(str).tolist()
    amp_clean = clean_and_filter_seqs(amp_raw)
    non_amp_clean = clean_and_filter_seqs(non_amp_raw)
    # 手动去重
    write_to_fasta(amp_clean, "./temp_data/amp.fasta", "AMP")
    write_to_fasta(non_amp_clean, "./temp_data/nonamp.fasta", "NONAMP")
    run_cdhit("./temp_data/amp.fasta", "./temp_data/amp_clean.fasta")
    run_cdhit("./temp_data/nonamp.fasta", "./temp_data/nonamp_clean.fasta")
    amp_final = read_from_fasta("./temp_data/amp_clean.fasta")
    non_amp_final = read_from_fasta("./temp_data/nonamp_clean.fasta")
    # 构建全量数据池
    all_sequences = amp_final + non_amp_final
    all_labels = [1.0] * len(amp_final) + [0.0] * len(non_amp_final)
    all_labels_tensor = torch.tensor(all_labels, dtype=torch.float32).view(-1, 1)
    # --- 第二步：全量特征预提取 ---
    global_embs, global_masks = extract_global_features(all_sequences, ESM_MODEL_NAME, DEVICE)
    global_embs = global_embs.half()
    gc.collect()
    torch.cuda.empty_cache()
    # --- 第三步：绝对隔离独立测试集 (基于索引) ---
    indices = np.arange(len(all_sequences))
    cv_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=all_labels
    )
    # 保存绝对隔离的独立测试集到 Excel
    test_seqs = [all_sequences[i] for i in test_idx]
    test_lbls = [all_labels[i] for i in test_idx]
    test_set_df = pd.DataFrame({'Sequence': test_seqs, 'True_Label': test_lbls})
    test_set_path = os.path.join(OUTPUT_DIR, "Independent_Test_Set.xlsx")
    test_set_df.to_excel(test_set_path, index=False)
    print(f" 已物理隔离并保存独立测试集 ({len(test_idx)} 条) 至: {test_set_path}")
    # 封装独立测试集
    test_feat_dataset = FeatureDataset(global_embs[test_idx], global_masks[test_idx], all_labels_tensor[test_idx])
    test_loader = DataLoader(test_feat_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=feature_collate_fn)
    print(f"  已隔离独立测试集: {len(test_idx)} 条")

    # --- 第四步：K 折交叉验证 (基于 cv_idx 再次划分) ---
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results_auc = []
    y_cv = [all_labels[i] for i in cv_idx]
    for fold_idx, (train_idx_in_cv, valid_idx_in_cv) in enumerate(skf.split(cv_idx, y_cv)):
        print(f"\n" + "-" * 30 + f" 开始第 {fold_idx + 1} 折 " + "-" * 30)
        # 将局部索引映射回全局索引
        actual_train_idx = cv_idx[train_idx_in_cv]
        actual_valid_idx = cv_idx[valid_idx_in_cv]
        # 极速构建数据加载器 (不重复计算，只是内存切片)
        # 极速构建数据加载器 (开启 pin_memory 高速通道)
        train_loader = DataLoader(
            FeatureDataset(global_embs[actual_train_idx], global_masks[actual_train_idx],
                           all_labels_tensor[actual_train_idx]),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=feature_collate_fn,
            pin_memory=True, num_workers=0  #
        )
        valid_loader = DataLoader(
            FeatureDataset(global_embs[actual_valid_idx], global_masks[actual_valid_idx],
                           all_labels_tensor[actual_valid_idx]),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=feature_collate_fn,
            pin_memory=True, num_workers=0  #
        )
        # 实例化轻量级模型
        model = ELK_Classifier(d_model=1280, kan_hidden_dim=KAN_HIDDEN_DIM).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        # 开始训练
        history, best_model_path = train_model(
            model=model, train_loader=train_loader, valid_loader=valid_loader,
            criterion=criterion, optimizer=optimizer, device=DEVICE,
            num_epochs=NUM_EPOCHS, fold_idx=fold_idx + 1
        )
        # 记录与画图
        plot_save_dir = os.path.join(OUTPUT_DIR, "plots")
        plot_learning_curves(history, fold_idx=fold_idx + 1,save_dir=plot_save_dir)
        model.load_state_dict(torch.load(best_model_path,map_location=DEVICE))

        print(f"\n 正在提取 Fold {fold_idx + 1}【最优模型】在【验证集】上的最终成绩...")

        # 把之前的 _ 全部换成具体的变量名接收
        best_v_loss, best_v_acc, best_v_sn, best_v_sp, best_v_mcc, best_v_auc, all_t, all_p = evaluate(model,
                                                                                                       valid_loader,
                                                                                                       criterion,
                                                                                                       DEVICE,
                                                                                                       "Valid_Best")
        # 打印极其醒目的成绩单面板
        print("┏" + "━" * 46 + "┓")
        print(f"┃  Fold {fold_idx + 1} 最佳验证集黄金指标 (Best Valid)  ┃")
        print("┣" + "━" * 46 + "┫")
        print(f"┃  ➤ Accuracy (ACC) : {best_v_acc:.4f}                   ┃")
        print(f"┃  ➤ Sensitivity(Sn): {best_v_sn:.4f}                   ┃")
        print(f"┃  ➤ Specificity(Sp): {best_v_sp:.4f}                   ┃")
        print(f"┃  ➤ MCC            : {best_v_mcc:.4f}                   ┃")
        print(f"┃  ➤ AUC            : {best_v_auc:.4f}                   ┃")
        print("┗" + "━" * 46 + "┛")
        plot_roc_curve(all_t, all_p, fold_idx + 1, best_v_auc, save_dir=plot_save_dir)
        fold_results_auc.append(best_v_auc)
        #  新增：拿本折选出的最优模型，对【独立测试集】进行大考打分
        print(f"\n 正在使用 Fold {fold_idx + 1} 最优模型测评【独立测试集】...")
        # 如果你想看独立测试集的五小强指标，这里也可以把 _ 换成变量名打印出来！
        test_loss, test_acc, test_sn, test_sp, test_mcc, test_auc, _, test_probs = evaluate(model, test_loader,
                                                                                            criterion, DEVICE,
                                                                                            "Independent_Test")
        print("┏" + "━" * 46 + "┓")
        print(f"┃  Fold {fold_idx + 1} 独立测试集盲测指标 (Blind Test)  ┃")
        print("┣" + "━" * 46 + "┫")
        print(f"┃  ➤ Accuracy (ACC) : {test_acc:.4f}                   ┃")
        print(f"┃  ➤ Sensitivity(Sn): {test_sn:.4f}                   ┃")
        print(f"┃  ➤ Specificity(Sp): {test_sp:.4f}                   ┃")
        print(f"┃  ➤ MCC            : {test_mcc:.4f}                   ┃")
        print(f"┃  ➤ AUC            : {test_auc:.4f}                   ┃")
        print("┗" + "━" * 46 + "┛")

        # 保存用于画 ROC 曲线的分数文件 (包含序列、真实标签、模型打分)
        roc_data_path = os.path.join(OUTPUT_DIR, f"elk_roc_scores_fold_{fold_idx + 1}.csv")
        pd.DataFrame({
            'Sequence': test_seqs,
            'True_Label': test_lbls,
            'ELK_Score': test_probs
        }).to_csv(roc_data_path, index=False)
        print(f" 测评完成！ROC 分数文件已保存至: {roc_data_path}")
        gc.collect()
        torch.cuda.empty_cache()
        print(f" Fold {fold_idx + 1} 进行下一折！")
        if RUN_ONE_FOLD_ONLY: break
    print("\n" + "=" * 50)
    print(f" 训练完成！平均 Valid AUC: {np.mean(fold_results_auc):.4f}")
    print("=" * 50)


# ==========================================
#  3. 加载模型预测新序列 (Predict Mode)
# ==========================================
def run_predict():
    print("\n" + "=" * 50)
    print(" [PREDICT MODE] 启动新序列批量预测任务")
    print("=" * 50)
    if not os.path.exists(NEW_PEPTIDES_PATH):
        print(f"找不到待测文件: {NEW_PEPTIDES_PATH}，请检查路径！")
        return
    print(" 正在读取待测序列...")
    try:
        # 提取全部原始序列
        #raw_sequences = pd.read_excel(NEW_PEPTIDES_PATH).iloc[:, 0].dropna().astype(str).tolist()
        raw_sequences = pd.read_excel(NEW_PEPTIDES_PATH, sheet_name=NEW_PEPTIDES_SHEET).iloc[:, 0].dropna().astype(
            str).tolist()
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # ---  核心升级：带有索引追踪的清洗逻辑 ---
    valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
    valid_seqs = []  # 用来送去模型预测的合格序列
    valid_indices = []  # 记录它们在原始列表中的行号
    for idx, seq in enumerate(raw_sequences):
        seq_clean = seq.upper().strip()
        if 5 <= len(seq_clean) <= 50 and all(char in valid_chars for char in seq_clean):
            valid_seqs.append(seq_clean)
            valid_indices.append(idx)

    print(
        f" 数据检查完成：总计 {len(raw_sequences)} 条 | 合格送检 {len(valid_seqs)} 条 | 过滤抛弃 {len(raw_sequences) - len(valid_seqs)} 条")
    if len(valid_seqs) == 0:
        print(" 没有找到有效的多肽序列，预测终止。")
        return
    # --- 极速提取 ESM-2 特征 (只提取合格的) ---
    print("\n 正在提取 ESM-2 全局特征...")
    embs, masks = extract_global_features(valid_seqs, ESM_MODEL_NAME, DEVICE)
    embs = embs.half()
    gc.collect()
    torch.cuda.empty_cache()
    # --- 构建 DataLoader ---
    dummy_labels = torch.zeros((len(valid_seqs), 1), dtype=torch.float32)
    predict_dataset = FeatureDataset(embs, masks, dummy_labels)
    predict_loader = DataLoader(predict_dataset, batch_size=512, shuffle=False, collate_fn=feature_collate_fn,
                                pin_memory=True)
    # --- 加载模型 ---
    model_path = os.path.join(OUTPUT_DIR, "best_Classifier_model.pth")

    if not os.path.exists(model_path):
        print(f" 找不到模型权重文件: {model_path}，请先运行 train 模式！")
        return
    print(f"\n 正在加载模型权重: {model_path} ...")
    model = ELK_Classifier(d_model=1280, kan_hidden_dim=KAN_HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    # --- 开始推理 ---
    print("开始推理...")
    valid_probs = []
    with torch.no_grad():
        for batch_embs, batch_masks, _ in tqdm(predict_loader, desc=" 预测进度"):
            batch_embs, batch_masks = batch_embs.to(DEVICE), batch_masks.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(batch_embs, batch_masks)
            probs = torch.sigmoid(logits.float()).cpu().numpy().flatten()
            valid_probs.extend(probs)

    # --- 核心升级：还原完整结果列表 ---
    print("\n 正在生成最终预测报告...")
    # 预先创建长度与 raw_sequences 一模一样的空列表
    final_is_amp = ["Invalid (过滤)"] * len(raw_sequences)
    final_probs = ["N/A"] * len(raw_sequences)
    amp_count = 0
    # 把预测出的结果，按照记录好的“座位号(valid_indices)”逐个填回去
    for v_idx, prob in zip(valid_indices, valid_probs):
        pred = 1 if prob >= 0.5 else 0
        final_is_amp[v_idx] = pred
        final_probs[v_idx] = prob
        if pred == 1:
            amp_count += 1
    # 生成最终的 Excel DataFrame
    results_df = pd.DataFrame({
        "Original_Sequence": raw_sequences,
        "Is_AMP": final_is_amp,
        "Probability": final_probs
    })
    # 计算最严谨的占比 (分母是全体原始序列)
    total_count = len(raw_sequences)
    amp_ratio = (amp_count / total_count) * 100
    output_filename = os.path.join(OUTPUT_DIR, "Predicted_AMP_Results.xlsx")
    results_df.to_excel(output_filename, index=False)
    print("=" * 50)
    print(" 预测完成")
    print(f" 结果已保存至: {output_filename}")
    print(f" 统计信息: 输入总序列 {total_count} 条")
    print(f" 识别出 AMP (正样本): {amp_count} 条")
    print(f"占比 (真实 AMP Ratio): {amp_ratio:.2f}%")
    print("=" * 50)

######################main###########################
#  核心开关：在这里切换任务模式！
RUN_MODE = "predict"  # 可选: "train" (训练模型) 或 "predict" (预测新序列)
OUTPUT_DIR = "E:/Users/Mordred/Desktop/ELK_Classifier_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 自动在创建统一输出文件夹

# ---  文件路径配置 ---
AMP_EXCEL_PATH = "E:/Users/Mordred/Desktop/AMP.xlsx"  # 正样本数据
NON_AMP_EXCEL_PATH = "E:/Users/Mordred/Desktop/nonAMP.xlsx"  # 负样本数据
NEW_PEPTIDES_PATH = R"E:\Users\Mordred\Desktop\generated_AMPs_uncond-pfema.xlsx"  # 未来给 predict 模式留的待测数据
NEW_PEPTIDES_SHEET = "Sheet1"  #表格名称

# ---  模型架构配置 ---
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
KAN_HIDDEN_DIM = 64
# ---  训练核心超参数 ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50  #测试设置1
K_FOLDS = 5
# True: 只跑第1折就结束，退化为一次训练集分割
# False: 完整的跑完 5 折交叉验证
RUN_ONE_FOLD_ONLY = True
# 自动检测算力设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#  4
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(f" 当前计算设备: {DEVICE}")
    print(f"⚙ 当前执行模式: {RUN_MODE}")
    print("=" * 60)

    if RUN_MODE == "train":
        run_train()
    elif RUN_MODE == "predict":
        run_predict()
    else:
        print("错误：RUN_MODE 必须是 'train' 或 'predict'！")


