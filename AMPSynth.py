import math
import os
import pandas as pd
import torch
from typing import Callable, Union, Tuple, Optional, List, Dict
from torch import Tensor
from torch.nn import Module
import numpy as np
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import esm
import argparse
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import traceback


### ==================== 数据预处理====================
class ExcelAMPDataLoader:
    """Excel AMP数据加载适配器 - 支持标签读取"""

    def __init__(self,
                 excel_path: str,
                 sequence_column: str = "Sequence",
                 label_column: str = None,  # ### 标签列名，若为None则视为无标签数据
                 fix_null_token: bool = False  # ### 是否强制所有数据为Null标签(预训练模式)
                 ):
        """
        初始化Excel数据加载器
        参数:
            excel_path: Excel文件路径
            sequence_column: 包含AMP序列的列名
            label_column: 包含细菌标签的列名 (例如 "Target_Bacteria")
            fix_null_token: 如果为True，忽略Excel标签，强制指定为Null Token (用于第一阶段预训练)
        """
        self.excel_path = excel_path
        self.sequence_column = sequence_column
        self.label_column = label_column
        self.fix_null_token = fix_null_token
        # ### 标签映射字典
        self.label_to_id = {}
        self.id_to_label = {}
        self.num_classes = 1  # 默认为1 (仅含Null Token)
        # 加载序列和标签
        self.data_pairs = self._load_sequences_and_labels()  # 返回 (seq, label_id) 列表

    def _load_sequences_and_labels(self) -> List[Tuple[str, int]]:  # 返回类型改变
        """从Excel加载AMP序列和标签"""
        try:
            # 读取Excel文件
            df = pd.read_excel(self.excel_path)
            # 检查序列列是否存在
            if self.sequence_column not in df.columns:
                raise ValueError(f"列 '{self.sequence_column}' 不存在。")
            # #处理标签列逻辑
            if self.label_column and self.label_column in df.columns and not self.fix_null_token:
                # 提取并去除空标签的行
                df = df.dropna(subset=[self.sequence_column, self.label_column])
                raw_labels = df[self.label_column].astype(str).tolist()
                # 自动构建标签映射 (排序保证ID固定)
                unique_labels = sorted(list(set(raw_labels)))
                self.label_to_id = {label: i for i, label in enumerate(unique_labels)}
                self.id_to_label = {i: label for label, i in self.label_to_id.items()}
                # 预留最后一个ID给 Null Token
                self.null_token_id = len(unique_labels)
                self.label_to_id['<NULL>'] = self.null_token_id
                self.id_to_label[self.null_token_id] = '<NULL>'
                self.num_classes = len(unique_labels) + 1
                print(f" 检测到 {len(unique_labels)} 种细菌标签: {unique_labels}")
                print(f"   Null Token ID 设置为: {self.null_token_id}")
            else:
                # 无标签模式 (或预训练模式)
                df = df.dropna(subset=[self.sequence_column])
                self.null_token_id = 0
                self.label_to_id = {'<NULL>': 0}
                self.id_to_label = {0: '<NULL>'}
                self.num_classes = 1
                raw_labels = ['<NULL>'] * len(df)
                print(" 未检测到标签列或开启强制Null模式，所有数据将标记为 Null Token")
            sequences = df[self.sequence_column].tolist()
            # 清理序列并配对标签
            cleaned_data = []
            # 如果是无标签模式，raw_labels 长度可能不匹配，需重新生成
            if len(raw_labels) != len(sequences):
                raw_labels = ['<NULL>'] * len(sequences)
            for seq, label_raw in zip(sequences, raw_labels):
                if isinstance(seq, str):
                    if not self._contains_only_standard_amino_acids(seq):
                        continue
                    clean_seq = self._clean_sequence(seq)
                    if clean_seq and self._is_valid_amp_sequence(clean_seq):
                        # 获取对应的ID
                        if self.fix_null_token:
                            label_id = self.null_token_id
                        else:
                            label_id = self.label_to_id.get(label_raw, self.null_token_id)
                        cleaned_data.append((clean_seq, label_id))
            print(f" 成功加载 {len(cleaned_data)} 个有效 (序列, 标签) 对")
            return cleaned_data
        except Exception as e:
            print(f" 加载Excel文件失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _contains_only_standard_amino_acids(self, sequence: str) -> bool:
        """检查序列是否仅包含20种标准氨基酸字符"""
        standard_aa_set = set('ACDEFGHIKLMNPQRSTVWY')
        return all(char.upper() in standard_aa_set for char in sequence)

    def _clean_sequence(self, sequence: str) -> str:
        """清理序列字符串"""
        clean = ''.join(c for c in sequence.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
        return clean

    def _is_valid_amp_sequence(self, sequence: str) -> bool:
        """验证是否为有效AMP序列"""
        if len(sequence) < 5 or len(sequence) > 50:
            return False
        valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
        return all(char in valid_chars for char in sequence)

    def get_data(self) -> List[Tuple[str, int]]:
        """获取所有有效数据对"""
        return self.data_pairs

    def get_sequences(self) -> List[str]:
        """仅获取序列列表"""
        return [item[0] for item in self.data_pairs]

    def get_sequence_stats(self) -> Dict:
        """获取序列统计信息"""
        if not self.data_pairs:
            return {}
        sequences = self.get_sequences()
        lengths = [len(seq) for seq in sequences]
        # ### [新增] 标签统计
        labels = [item[1] for item in self.data_pairs]
        label_counts = {self.id_to_label[label_id]: labels.count(label_id) for label_id in set(labels)}
        return {
            'total_sequences': len(sequences),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_length': sum(lengths) / len(lengths),
            'label_distribution': label_counts
        }


def prepare_training_from_excel(
        excel_path: str = "C:/Users/Mordred/Desktop/AMP-data.xlsx",
        sequence_column: str = "Sequence",
        label_column: str = None,  # ###  支持传入标签列
        batch_size: int = 32,
        fix_null_token: bool = False  # ###  预训练开关
) -> Tuple[DataLoader, int, int]:  # ### 返回值增加 num_classes 和 null_token_id
    """
    从Excel文件准备训练数据加载器 - 支持标签
    """
    # 1. 加载Excel数据
    # ###传递标签参数
    data_loader = ExcelAMPDataLoader(excel_path, sequence_column, label_column, fix_null_token)
    data_pairs = data_loader.get_data()
    if not data_pairs:
        raise ValueError("未找到有效的数据")
    # 2. 显示统计信息
    stats = data_loader.get_sequence_stats()
    print(" 数据集统计:")
    for key, value in stats.items():
        if key == 'label_distribution':
            print(f"   标签分布: {value}")  #
        elif key != 'length_distribution':
            print(f"   {key}: {value}")

    # 3. 创建PyTorch Dataset
    class AMPDataset(torch.utils.data.Dataset):
        def __init__(self, data_pairs):
            self.data_pairs = data_pairs  # List of (seq, label_id)

        def __len__(self):
            return len(self.data_pairs)

        def __getitem__(self, idx):
            # ###返回 (序列, 标签ID)
            return self.data_pairs[idx]

    dataset = AMPDataset(data_pairs)

    # 4. 创建DataLoader
    def amp_collate_fn(batch):
        """
        ### collate函数
        输入: List[(seq_str, label_int)]
        输出: (List[str], Tensor[Batch_Labels])
        """
        sequences = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return sequences, labels  # ###  返回分离的序列列表和标签张量

    # 创建DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=amp_collate_fn
    )
    # ###  返回 DataLoader 以及关于类别的元数据 (供模型初始化使用)返回训练数据，类别数，各个类别个数
    return train_loader, data_loader.num_classes, data_loader.null_token_id


### ==================== 数据预处理====================

## ==================== ESM编码器定义====================
class ESMEncoderWrapper(nn.Module):
    """
    冻结参数的预训练ESM编码器封装
    与您的SGM框架完全兼容
    """

    def __init__(self,
                 model_name: str = "esm2_t33_650M_UR50D",
                 device: str = "cuda",
                 pooling_strategy: str = "none",
                 max_seq_len: int = 1000):  # esm最多处理1024个氨基酸的肽
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

# ==================== 计算σ_data的方差====================
def compute_sigma_data(excel_path: str = "C:/Users/Mordred/AMP-data.xlsx",
                       sequence_column: str = "Sequence",
                       batch_size: int = 32,
                       device: str = "cuda"):
    """
    计算sigma_data - 批次级别计算版本：每个批次计算RMS，最后平均
    避免填充，提高计算效率
    """
    print(" 开始计算sigma_data（批次级别计算版本）...")
    # 1. 加载数据
    data_loader = ExcelAMPDataLoader(excel_path, sequence_column)
    sequences = data_loader.get_sequences()
    if not sequences:
        raise ValueError("未找到有效的AMP序列")
    print(f" 处理 {len(sequences)} 个序列")
    # 2. 初始化ESM编码器
    esm_encoder = ESMEncoderWrapper(
        model_name="esm2_t33_650M_UR50D",
        pooling_strategy="none",
        device=device
    )
    esm_encoder.eval()
    # 3. 存储每个批次的RMS值
    batch_rms_values = []
    batch_std_values = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="处理批次"):
            batch_sequences = sequences[i:i + batch_size]
            try:
                # 编码当前批次（ESM内部会进行批次内填充）
                embeddings = esm_encoder.encode_batch(batch_sequences)
                # 确保是3D张量 [batch_size, seq_len, hidden_dim]
                if embeddings.dim() == 2:
                    embeddings = embeddings.unsqueeze(1)
                # 方法1：计算当前批次的全局RMS（整个批次所有token的RMS）
                batch_rms = torch.sqrt((embeddings ** 2).mean(dim=(1, 2))).mean().item()
                batch_rms_values.append(batch_rms)
                # 方法2：计算当前批次的全局标准差（用于对比）
                batch_std = embeddings.std().item()
                batch_std_values.append(batch_std)
                # 打印批次信息
                batch_info = f"批次 {len(batch_rms_values)}: RMS={batch_rms:.4f}, STD={batch_std:.4f}"
                if len(batch_rms_values) % 20 == 0:  # 每20个批次打印一次
                    print(f"   {batch_info}")
            except Exception as e:
                print(f"️批次处理失败: {e}")
                continue
    if not batch_rms_values:
        raise ValueError("无法计算sigma_data")
    # 4. 计算最终结果（所有批次的平均值）
    sigma_data_rms = np.mean(batch_rms_values)
    sigma_data_std = np.mean(batch_std_values)
    # 5. 输出详细统计信
    print(f"   批次RMS平均法: {sigma_data_rms:.6f} ")
    print(f"   批次STD平均法: {sigma_data_std:.6f}")
    # sigma_data_std
    return sigma_data_rms  #默认使用 批次RMS平均法


# ==================== 计算σ_data的方差====================

# ==================== 计算EDM参数的定义====================
class EDMConfig:
    """EDM训练范式参数配置"""

    def __init__(self,
                 sigma_data: float = 0.283498,
                 P_mean: float = -1.2,
                 P_std: float = 1.2,
                 sigma_min: float = 0.002,
                 sigma_max: float = 80.0,
                 rho: float = 7.0,
                 sigma_sample_strategy: str = "log_normal"):
        """
        EDM参数配置
        Args:
            sigma_data: 数据分布标准差（必须根据训练数据计算）
            P_mean: 对数正态分布的均值参数
            P_std: 对数正态分布的标准差参数
            sigma_min: 最小噪声水平
            sigma_max: 最大噪声水平
            rho: 噪声调度参数
            sigma_sample_strategy: 噪声采样策略
        """
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_sample_strategy = sigma_sample_strategy

    def to_dict(self) -> Dict:
        """将配置转换为字典，便于序列化"""
        return {
            'sigma_data': self.sigma_data,
            'P_mean': self.P_mean,
            'P_std': self.P_std,
            'sigma_min': self.sigma_min,
            'sigma_max': self.sigma_max,
            'rho': self.rho,
            'sigma_sample_strategy': self.sigma_sample_strategy
        }

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'EDMConfig':
        """从字典恢复配置"""
        return cls(**config_dict)

    def __str__(self):
        return f"EDMConfig(sigma_data={self.sigma_data:.4f}, P_mean={self.P_mean}, P_std={self.P_std})"

    def get_sigma_sampling_distribution(self, batch_size: int, device: str):
        """根据策略生成噪声水平采样分布"""
        if self.sigma_sample_strategy == "log_normal":
            # 从对数正态分布采样
            log_sigma = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
            return log_sigma.exp()
        elif self.sigma_sample_strategy == "uniform_log":
            # 在对数空间均匀采样
            log_sigma = torch.rand(batch_size, device=device) * (
                    np.log(self.sigma_max) - np.log(self.sigma_min)) + np.log(self.sigma_min)
            return log_sigma.exp()
        else:
            raise ValueError(f"不支持的采样策略: {self.sigma_sample_strategy}")


# ==================== 计算EDM参数的定义====================

# ==================== 基于幂函数EMA框架====================
class PFEMA:  #暂时没有使用
    """ 基于幂函数EMA框架，使用固定σ_rel"""

    def __init__(self, model: nn.Module, sigma_rel: float = 0.1, device: str = "cuda"):
        """
        Args:
            model: 要应用EMA的模型
            sigma_rel: 相对标准差（控制EMA窗口大小）
            device: 计算设备
        """
        self.model = model
        self.sigma_rel = sigma_rel
        self.device = device
        self.step = 0
        # 计算γ参数
        self.gamma = self._calculate_gamma(sigma_rel)
        # 初始化影子参数
        self.shadow_params = {}
        self._init_shadow_params()
        print(f"初始化简化EMA: σ_rel={sigma_rel:.3f}, γ={self.gamma:.3f}")

    def _calculate_gamma(self, sigma_rel: float) -> float:
        """
        基于Beta分布统计特性
        """
        # 参数范围检查
        if sigma_rel <= 0 or sigma_rel >= 0.2887:
            raise ValueError(f"σ_rel必须在(0, 0.2887)范围内，当前值: {sigma_rel}")
        # 数值求解三次方程
        return self._solve_gamma_equation(sigma_rel)

    def _solve_gamma_equation(self, sigma_rel: float) -> float:
        """数值求解γ的三次方程 """
        sigma_sq = sigma_rel ** 2

        def f(gamma):
            return (sigma_sq * (gamma ** 3 + 7 * gamma ** 2 + 16 * gamma + 12) -
                    (gamma + 1))

        # 二分法求解
        low, high = 0.0, 100.0
        tolerance = 1e-6
        for _ in range(50):
            mid = (low + high) / 2
            f_mid = f(mid)
            # 修正：检查是否找到精确解
            if abs(f_mid) < 1e-12:  # 使用更严格的容差检查接近0
                return mid
            if f_mid > 0:
                high = mid
            else:
                low = mid
            if high - low < tolerance:
                break
        return (low + high) / 2

    def _init_shadow_params(self):
        """初始化影子参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach()

    def update(self):
        """更新EMA参数 - 每个训练步骤调用一次"""
        self.step += 1
        # 计算当前步的衰减率β
        if self.step > 1:  # 从第二步开始更新
            beta = (1.0 - 1.0 / self.step) ** (self.gamma + 1)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.shadow_params and param.requires_grad:
                        # 幂函数EMA更新公式
                        self.shadow_params[name] = (
                                beta * self.shadow_params[name] +
                                (1 - beta) * param.data
                        )

    def apply_to_model(self):
        """将EMA参数应用到模型"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])

    def get_state_dict(self) -> Dict:
        """获取EMA状态（用于保存检查点）"""
        return {
            'shadow_params': {k: v.clone() for k, v in self.shadow_params.items()},
            'sigma_rel': self.sigma_rel,
            'gamma': self.gamma,
            'step': self.step,
            'device': self.device
        }

    def load_state_dict(self, state_dict: Dict):
        """加载EMA状态"""
        self.shadow_params = {k: v.to(self.device) for k, v in state_dict['shadow_params'].items()}
        self.sigma_rel = state_dict.get('sigma_rel', 0.1)
        self.gamma = state_dict.get('gamma', self._calculate_gamma(self.sigma_rel))
        self.step = state_dict.get('step', 0)

    def __repr__(self):
        return f"PFEMA(σ_rel={self.sigma_rel:.3f}, γ={self.gamma:.3f}, step={self.step})"


# ==================== 基于幂函数EMA框架====================


## ==================== ESM解码器定义====================
class ESMLanguageHeadFinetuner(nn.Module):
    """
    ESM语言头微调器
    1. 在SGM训练完成后执行微调
    2. 自动处理维度对齐（支持SGM降维后的升维需求）
    3. 与现有ESMEncoderWrapper完全兼容
    """

    def __init__(self,
                 esm_encoder,  # 您的ESMEncoderWrapper实例
                 edm_output_dim: int = 1280,  # 输出维度
                 max_seq_len: int = 52,
                 device: str = "cuda"):
        super().__init__()
        self.esm_encoder = esm_encoder
        self.edm_output_dim = edm_output_dim
        self.esm_input_dim = esm_encoder.hidden_size  # ESM原始维度（1280）
        self.max_seq_len = max_seq_len
        self.device = device
        # 获取ESM模型和词汇表
        self.esm_model = esm_encoder.model
        self.alphabet = esm_encoder.alphabet

        if hasattr(self.esm_model.lm_head, 'weight'):
            self.vocab_size = self.esm_model.lm_head.weight.size(0)
            print(f" 通过权重矩阵获取词汇表大小: {self.vocab_size}")
            # 方法2：通过alphabet获取（备用方案）
        elif hasattr(self.alphabet, '__len__'):
            self.vocab_size = len(self.alphabet)
            print(f" 通过alphabet获取词汇表大小: {self.vocab_size}")
        else:
            # 方法3：硬编码ESM2的标准词汇表大小（最后手段）
            self.vocab_size = 33  # ESM2标准词汇表大小
            print(f" 使用默认词汇表大小: {self.vocab_size}")

        # 特殊token ID
        self.padding_idx = self.alphabet.padding_idx
        self.cls_idx = getattr(self.alphabet, 'cls_idx', 0)
        self.eos_idx = getattr(self.alphabet, 'eos_idx', 1)
        # 冻结整个ESM编码器
        for param in self.esm_model.parameters():
            param.requires_grad = False
        # 维度适配器：处理SGM输出维度与ESM输入维度的对齐

        self.dimension_adapter = nn.Identity()
        # ESM语言头（lm_head） - 只微调这个部分
        self.lm_head = self.esm_model.lm_head
        self.lm_head.requires_grad_(True)  # 解锁语言头进行微调
        print(f" ESM语言头微调器初始化完成")
        print(f"  输入维度: {edm_output_dim} → ESM维度: {self.esm_input_dim} → 词汇表: {self.vocab_size}")
        print(f"  最大序列长度: {max_seq_len}")
        print(f"  可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self,
                latent_embeddings: Tensor,
                target_tokens: Optional[Tensor] = None,
                attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播：将潜在变量解码为序列logits
        参数:
            latent_embeddings: SGM生成的潜在变量 [batch_size, seq_len, sgm_output_dim]
            target_tokens: 目标序列token（训练时使用）[batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
        返回:
            logits: 每个位置的氨基酸概率 [batch_size, seq_len, vocab_size]
            loss: 交叉熵损失（训练时）
        """
        # 1. 维度适配：将SGM输出维度对齐到ESM输入维度
        adapted_embeddings = self.dimension_adapter(latent_embeddings)
        # 现在形状: [batch_size, seq_len, esm_input_dim]
        # 2. 通过ESM语言头得到logits
        logits = self.lm_head(adapted_embeddings)  # [batch_size, seq_len, vocab_size]
        # 3. 计算损失
        loss = None
        if target_tokens is not None:
            loss = self._compute_loss(logits, target_tokens, attention_mask)
        return logits, loss

    def _compute_loss(self,
                      logits: Tensor,
                      targets: Tensor,
                      attention_mask: Optional[Tensor]) -> Tensor:
        """计算交叉熵损失"""
        # 验证输入形状
        if logits.shape[:2] != targets.shape:
            raise ValueError(f"logits和targets形状不匹配: {logits.shape[:2]} vs {targets.shape}")
        # 重塑为二维
        logits_flat = logits.reshape(-1, self.vocab_size)  # [batch_size * seq_len, vocab_size]
        targets_flat = targets.reshape(-1)  # [batch_size * seq_len]
        # 创建损失掩码（忽略填充token）
        if attention_mask is not None:
            loss_mask = attention_mask.reshape(-1).bool()
        else:
            loss_mask = (targets_flat != self.padding_idx)
        # 只对有效位置计算损失
        if loss_mask.any():
            valid_logits = logits_flat[loss_mask]
            valid_targets = targets_flat[loss_mask]
            if valid_targets.numel() > 0:
                loss = torch.nn.functional.cross_entropy(
                    valid_logits,
                    valid_targets,
                    ignore_index=self.padding_idx,
                    reduction='mean'
                )
            else:
                loss = torch.tensor(0.0, device=logits.device)
        else:
            loss = torch.tensor(0.0, device=logits.device)
        return loss

    @torch.no_grad()
    def generate_sequences(self,
                           latent_embeddings: Tensor,
                           max_length: int = 52,
                           temperature: float = 1.0) -> List[str]:
        """
        自回归生成序列
        """
        if latent_embeddings.dim() == 2:
            latent_embeddings = latent_embeddings.unsqueeze(0)
        batch_size, seq_len, input_dim = latent_embeddings.shape
        self.eval()
        generated_sequences = []
        for i in range(batch_size):
            # 当前样本的潜在嵌入
            sample_latent = latent_embeddings[i]  # [seq_len, input_dim]
            # 初始化生成序列（从CLS token开始）
            current_tokens = [self.cls_idx]
            sequence_tokens = []  # 存储有效的氨基酸token
            for step in range(min(max_length, self.max_seq_len)):
                current_len = len(current_tokens)
                # 准备当前输入
                if current_len > seq_len:
                    # 如果超过潜在变量长度，使用最后一个
                    current_input = sample_latent[-1:].unsqueeze(0)
                else:
                    current_input = sample_latent[:current_len].unsqueeze(0)  # [1, current_len, input_dim]
                # 维度适配
                adapted_input = self.dimension_adapter(current_input)  # [1, current_len, esm_input_dim]
                #1.通过语言头预测下一个token
                with torch.no_grad():
                    logits = self.lm_head(adapted_input)  # [1, current_len, vocab_size]
                    next_token_logits = logits[0, -1, :] / temperature
                    probabilities = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1).item()

                # 2.通过语言头预测下一个token
                # with torch.no_grad():
                #     logits = self.lm_head(adapted_input)  # [1, current_len, vocab_size]
                #     # 贪心算法：永远只选概率最高、最安全的那个氨基酸！
                #     next_token = torch.argmax(logits[0, -1, :], dim=-1).item()

                # 3.通过语言头预测下一个token
                # with torch.no_grad():
                #     logits = self.lm_head(adapted_input)  # [1, current_len, vocab_size]
                #     next_token_logits = logits[0, -1, :] / temperature
                #     # --- 新增: Top-p (Nucleus) 采样截断逻辑 ---
                #     top_p = 0.90  # 核心参数：只保留累计概率占 90% 的头部氨基酸，直接砍掉长尾
                #     sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                #     cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                #     # 找到需要移除的长尾 token 索引
                #     sorted_indices_to_remove = cumulative_probs > top_p
                #     # 向右平移一位，确保哪怕第一个 token 概率就超过 top_p，也能至少保留一个候选项
                #     sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                #     sorted_indices_to_remove[0] = False
                #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
                #     next_token_logits[indices_to_remove] = float('-inf')  # 将长尾氨基酸的概率物理清零
                #     # ----------------------------------------
                #     probabilities = torch.softmax(next_token_logits, dim=-1)
                #     next_token = torch.multinomial(probabilities, num_samples=1).item()

                # 检查结束标记
                if next_token == self.eos_idx:
                    break
                current_tokens.append(next_token)
                # 将token转换为氨基酸字符（过滤特殊token）
                try:
                    token_str = self.alphabet.get_tok(next_token)
                    if token_str in 'ACDEFGHIKLMNPQRSTVWY':  # 标准氨基酸
                        sequence_tokens.append(token_str)
                except:
                    continue
            generated_sequences.append(''.join(sequence_tokens))
        return generated_sequences

    def get_trainable_parameters(self):
        """返回需要训练的参数（用于优化器）"""
        return [p for p in self.parameters() if p.requires_grad]


def prepare_esm_finetuning_data(esm_encoder, sequences: List[str], device: str = "cuda", batch_size: int = 32):
    """
    准备ESM语言头微调数据
   使用 ESM 原生 batch 处理，确保 padding embedding 与 EDM 训练时一致，
           解决生成序列过长(Decoder不识别Padding)的问题。
    """
    z_list = []
    target_tokens_list = []
    mask_list = []

    print(f"正在准备 Decoder 微调数据 (Batch Size={batch_size})...")

    # 按批次处理，利用 ESM 原生的 padding 机制
    # 这样得到的 embedding 中的 padding 部分就不是全0，而是 ESM 特有的 padding embedding
    for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding"):
        batch_seqs = sequences[i: i + batch_size]

        # 1. 准备 Batch 数据 (标签, 序列)
        batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]

        # 2. 获取 Token 和 Mask (这里会自动进行 Padding)
        # batch_tokens: [B, Max_Len]
        _, _, batch_tokens = esm_encoder.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        # 3. 通过 ESM 模型获取 Embeddings
        with torch.no_grad():
            results = esm_encoder.model(
                batch_tokens,
                repr_layers=[esm_encoder.num_layers],
                return_contacts=False
            )
            # token_representations: [B, Max_Len, Hidden_Dim]
            # 这里的 Padding 位置包含了真实的 ESM Padding Embedding
            z = results["representations"][esm_encoder.num_layers]

        # 4. 生成 Attention Mask (用于 Loss 计算忽略 Padding)
        # padding_idx 通常是 1
        mask = (batch_tokens != esm_encoder.alphabet.padding_idx).float()
        z_list.append(z.cpu())  # 先存到 CPU 节省显存
        target_tokens_list.append(batch_tokens.cpu())
        mask_list.append(mask.cpu())
    # 合并数据
    # 注意：不同 batch 的 max_len 可能不同，需要再次统一填充到全局最大长度
    # 但这次填充可以用 0，因为这是 batch 之间的对齐，Decoder 训练时 mask 会屏蔽掉这些区域
    # 找到全局最大长度
    global_max_len = max(t.size(1) for t in target_tokens_list)
    final_z = []
    final_targets = []
    final_masks = []
    for z, t, m in zip(z_list, target_tokens_list, mask_list):
        B, L, D = z.shape
        if L < global_max_len:
            pad_len = global_max_len - L
            # z 补 0
            z = torch.cat([z, torch.zeros(B, pad_len, D)], dim=1)
            # target 补 padding_idx
            t = torch.cat([t, torch.full((B, pad_len), esm_encoder.alphabet.padding_idx)], dim=1)
            # mask 补 0
            m = torch.cat([m, torch.zeros(B, pad_len)], dim=1)
        final_z.append(z)
        final_targets.append(t)
        final_masks.append(m)
    z_tensor = torch.cat(final_z, dim=0).to(device)
    targets_tensor = torch.cat(final_targets, dim=0).to(device)
    masks_tensor = torch.cat(final_masks, dim=0).to(device)
    print(f"微调数据准备完成:")
    print(f"  潜在变量: {z_tensor.shape}")
    print(f"  目标token: {targets_tensor.shape}")
    return z_tensor, targets_tensor, masks_tensor


def train_esm_language_head(
        esm_decoder: ESMLanguageHeadFinetuner,
        train_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50,
        lr: float = 1e-4,
        device: str = "cuda"
) -> Tuple[ESMLanguageHeadFinetuner, List[float]]:
    """
    训练ESM语言头微调器（在SGM训练完成后执行）
    """
    esm_decoder.to(device)
    esm_decoder.train()
    # 只优化可训练参数
    optimizer = torch.optim.AdamW(
        esm_decoder.get_trainable_parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"微调ESM语言头 Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            # 假设batch包含: (潜在变量z, 目标token序列, 注意力掩码)
            if len(batch) == 3:
                z, target_tokens, attention_mask = batch
            else:
                z, target_tokens = batch
                attention_mask = None
            z = z.to(device)
            target_tokens = target_tokens.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            # 前向传播
            logits, loss = esm_decoder(z, target_tokens, attention_mask)
            # 检查损失有效性
            if torch.isnan(loss) or torch.isinf(loss):
                print("警告: 检测到无效损失值，跳过该批次")
                continue
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                esm_decoder.get_trainable_parameters(),
                max_norm=1.0
            )
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
        scheduler.step()
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, 平均损失: {avg_loss:.6f}")
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': esm_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f"esm_language_head_epoch_{epoch + 1}.pth")
            print(f" 检查点已保存: esm_language_head_epoch_{epoch + 1}.pth")
    return esm_decoder, train_losses


# ==================== 时间t(sigma)编码 ====================
class GaussianFourierProjection(nn.Module):
    """ Gaussian random features for encoding time steps. """

    def __init__(self, embed_dim, scale=30.):  # embed_dim 为偶数
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        # \omega \sim \mathcal N(0, s^2 I), s = 30.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1) * math.sqrt(2.0)


# ==================== 时间t编码 ====================


# ==================== 绝对位置编码 ====================
class PositionalEncoding(nn.Module):  # 位置编码
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


# ==================== 绝对位置编码 ====================

# ==================== 余弦多头注意力机制====================
class MultiHeadCosineAttention(nn.Module):
    """多头余弦注意力"""

    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        # 投影层（保持多头结构）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.log_scale = nn.Parameter(torch.zeros(nhead, 1, 1))
        init_val = math.log(math.sqrt(self.head_dim))
        self.batch_first = batch_first
        with torch.no_grad():
            self.log_scale.fill_(init_val)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size, seq_len, _ = query.shape
        # 1. 线性投影 + 多头分割
        Q = self.q_proj(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        # 2. 余弦相似度计算（每个头独立）
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        # 3. 余弦相似度矩阵
        similarity = torch.matmul(Q_norm, K_norm.transpose(-2, -1))
        scale = torch.exp(self.log_scale)
        similarity = similarity * scale
        # 4. 掩码和Softmax
        if key_padding_mask is not None:
            # 扩展维度以匹配 [B, H, Seq_Q, Seq_K]
            # 假设 query 和 key 长度一致 (self-attention)
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Seq_K]
            similarity = similarity.masked_fill(mask_expanded, float('-inf'))
            # attn_mask: [Seq, Seq] (通常用于因果掩码)
        if attn_mask is not None:
            similarity = similarity.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(similarity, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 5. 注意力加权和输出投影
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, D_h]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output), attn_weights


class CosineTransformerEncoderLayer(nn.Module):
    """使用多头余弦注意力的Transformer层 """

    def __init__(self, d_model: int, nhead: int, dim_feedforward=None,
                 dropout: float = 0.1, activation="gelu", batch_first=True):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        # 使用多头余弦注意力
        self.self_attn = MultiHeadCosineAttention(d_model, nhead, dropout)
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 激活函数
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        self.batch_first = batch_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        # 自注意子层 + 残差连接
        attn_output, attn_weights = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask,
                                                   attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        # 前馈子层 + 残差连接
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        return src


# ==================== 余弦多头注意力机制====================


# ==================== transfomer分数网络估计 ====================
### ==================== 模型架构层面: DiT / AdaLN 实现  ====================
# ###AdaLN 核心操作: 对输入 x 进行缩放(scale)和平移(shift)
def modulate(x, shift, scale):
    """
    AdaLN 的核心调制操作
    x: [Batch, Seq, Dim]
    shift, scale: [Batch, Dim] (由条件向量 c 预测得到)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ### 基于 DiT 的 Transformer Block (替代原有的 nn.TransformerEncoderLayer)
class DiTBlock(nn.Module):
    """
    DiT Block: 支持 AdaLN-Zero 的 Transformer 层
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1,
                 cond_dim: int = None, use_cosine_attention: bool = True):
        super().__init__()
        # 1. 自注意力模块
        if use_cosine_attention:
            self.attn = MultiHeadCosineAttention(d_model, nhead, dropout)
        else:
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # 2. 前馈网络 (FFN)
        # 使用 GELU (近似 tanh )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        # 3. AdaLN 调制参数生成器 (核心组件)
        # 输入条件 c，输出 6 个参数控制 Norm (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model, bias=True)
        )
        # 4. 基础 LayerNorm (去掉自带的 weight/bias，因为由 AdaLN 动态生成)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        # ### Zero-Initialization 策略
        # 将最后一层全连接初始化为 0，使得 Block 初始状态近似为 Identity (恒等映射)
        # 这大大加速了深层网络的收敛
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: 序列特征 [Batch, Seq, Dim]
            c: 全局条件向量 [Batch, Cond_Dim] (融合了 Sigma 和 Label)
            padding_mask: 填充掩码 [Batch, Seq]
        """
        # 1. 计算调制参数: 将条件 c 映射为 6 份参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        # 2. Attention Block (Norm -> Modulate -> Attn -> Scale -> Add)
        # (a) AdaLN 归一化
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        # (b) Self-Attention
        if isinstance(self.attn, MultiHeadCosineAttention):
            attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, key_padding_mask=padding_mask)
        else:
            attn_out, _ = self.attn(x_norm1, x_norm1, x_norm1, key_padding_mask=padding_mask)
        # (c) 门控残差连接 (Gate * Attn + Residual)
        x = x + gate_msa.unsqueeze(1) * attn_out
        # 3. FFN Block (Norm -> Modulate -> FFN -> Scale -> Add)
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        ffn_out = self.feed_forward(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * ffn_out
        return x


# ###  DiT 的最后一层 (Final Layer)
class FinalLayer(nn.Module):
    """
    DiT 输出层: AdaLN -> Linear -> Output
    同样使用 AdaLN 对最终特征进行调节，确保输出分布受条件控制
    """

    def __init__(self, d_model: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model, bias=True)
        )
        self.linear = nn.Linear(d_model, out_dim)
        # Zero Init
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # 预测 shift 和 scale
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # 调制 -> 线性投影
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ### 主模型结构: 统一支持预训练和微调
class ScoreNetwork(nn.Module):
    """
    基于 CADiT 的分数网络
    功能:
    1. 预训练兼容: 当 class_labels=None 时，自动使用 Null Token，行为等同于无条件模型。
    2. 微调兼容: 接收真实标签，利用 AdaLN 将细菌特征注入每一层。
    3. 结构统一: 无论哪个阶段，模型结构完全一致，权重可直接加载。
    """

    def __init__(self,
                 esm_input_dim: int = 1280,
                 d_model: int = 1280,
                 nhead: int = 20, #
                 num_layers: int = 6,
                 dim_feedforward: int = None,
                 dropout: float = 0.1,
                 max_seq_len: int = 100,
                 edm_config: EDMConfig = None,
                 use_cosine_attention: bool = True,
                 num_classes: int = 1):  # ### 类别总数，预训练时默认为1
        super().__init__()
        self.d_model = d_model
        self.esm_input_dim = esm_input_dim
        self.edm_config = edm_config
        self.sigma_data = edm_config.sigma_data
        self.num_classes = num_classes
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        # 1. 基础组件
        self.esm_adapter = nn.Identity()
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.input_projection = nn.Linear(esm_input_dim, d_model)  # 将 ESM 维度映射到模型维度
        # 2. 条件嵌入模块 (Condition Embedding)
        # 目标: 将 Sigma(连续) 和 Label(离散) 融合为一个全局条件向量 c
        # (a) Sigma (噪声) 嵌入器
        self.sigma_embedder = nn.Sequential(
            GaussianFourierProjection(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        # (b) Label (细菌) 嵌入器 ###
        # 预训练时虽然不用真实标签，但这一层必须存在以保持架构统一
        self.label_embedder = nn.Embedding(num_classes, d_model)
        # 初始化: 使用较小的 std，确保初始阶段条件对模型影响平滑
        nn.init.normal_(self.label_embedder.weight, std=0.02)
        # 3. 堆叠 DiT Blocks ###
        # 堆叠的 DiTBlock，
        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                cond_dim=d_model,  # 条件向量维度 = Sigma + Label (相加后保持 d_model)
                use_cosine_attention=use_cosine_attention
            )
            for _ in range(num_layers)
        ])
        # 4. 最终输出层 ### [
        self.final_layer = FinalLayer(d_model, esm_input_dim, cond_dim=d_model)
        print(f" DiT架构初始化完成: Layers={num_layers}, Cond=AdaLN-Zero, Classes={num_classes}")

    def forward(self, x: Tensor, sigma: Tensor,
                padding_mask: Optional[Tensor] = None,
                class_labels: Optional[Tensor] = None) -> Tensor:
        """
        前向传播
        Args:
            x: 序列 [B, Seq, Dim]
            sigma: 噪声水平 [B]
            class_labels: 标签 [B] (可选。预训练或无条件生成时为 None)
        """
        batch_size = x.shape[0]
        # --- 1. EDM 预条件处理 (Skip Connection 准备) ---
        if self.edm_config:
            c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
            x_in = x * c_in.unsqueeze(-1).unsqueeze(-1)
        else:
            x_in = x
            c_skip, c_out = 1.0, 1.0
        # --- 2. 准备条件向量 c (核心逻辑) ---
        # (A) 计算 Sigma Embedding
        c_noise = 0.25 * torch.log(sigma + 1e-8)
        sigma_vec = self.sigma_embedder(c_noise)  # [B, D]
        # (B) 计算 Label Embedding
        if class_labels is None:
            # ### 统一预训练和微调接口
            # 如果没传标签 (预训练阶段 或 推理时的无条件分支)，自动使用 Null Token
            # Null Token 是最后一个 ID (num_classes - 1)
            null_id = self.num_classes - 1
            # 创建全为 Null ID 的标签张量
            labels_to_embed = torch.full((batch_size,), null_id, device=x.device, dtype=torch.long)
        else:
            # 微调阶段：使用传入的真实标签 (或被 CFG 随机 mask 后的标签)
            labels_to_embed = class_labels
        label_vec = self.label_embedder(labels_to_embed)  # [B, D]
        # (C) 融合条件: c = Sigma + Label
        # DiT 的标准做法是将它们相加，作为最终的条件向量输入 AdaLN
        c = sigma_vec + label_vec
        # --- 3. 主干网络 Forward ---
        # 投影 + 位置编码
        h = self.input_projection(x_in)
        h = self.pos_encoding(h)
        # 逐层经过 DiT Blocks，注入条件 c
        for block in self.blocks:
            h = block(h, c, padding_mask)
            # --- 4. 最终输出 ---
        # 经过最后一层 AdaLN + Linear
        F_x = self.final_layer(h, c)
        # --- 5.  Denormalize ---
        if self.edm_config:
            output = (c_skip.unsqueeze(-1).unsqueeze(-1) * x +
                      c_out.unsqueeze(-1).unsqueeze(-1) * F_x)
        else:
            output = F_x
        return output


### ==================== 模型架构层面结束 ====================
# ==================== transfomer分数网络估计 ====================

class DynamicLossWeighting(nn.Module):  # 动态加权u(sigma)
    def __init__(self, fourier_dim=128):
        super().__init__()
        # 傅里叶特征编码
        self.fourier_proj = GaussianFourierProjection(embed_dim=fourier_dim)
        # 单层MLP：傅里叶特征 -> 全连接 -> 输出
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        self.softplus = nn.Softplus()

    def forward(self, sigma):
        # 生成傅里叶特征
        log_sigma = torch.log(sigma)
        fourier_features = self.fourier_proj(0.25 * log_sigma)
        u = self.mlp(fourier_features)
        u = self.softplus(u) + 1e-6
        return u.squeeze(-1)


# ==================== loss函数定义====================
def loss_fn(model, data: Tensor, edm_config, loss_weight_net: DynamicLossWeighting,
                 device="cuda",
                 class_labels: Optional[Tensor] = None) -> torch.Tensor:
    """ loss 函数, 其中时间变量是连续数值而非离散的时间步 """
    """
        model 这里指分数网络score_fn
    :   L = E[λ(σ)/e^{u(σ)} * ||D_θ(x; σ) - x||²+ u(σ)]
    """
    batch_size = data.size(0)
    # 从配置的分布中采样噪声水平σ,噪声强度
    sigma = edm_config.get_sigma_sampling_distribution(batch_size, device)  # sigma: [B]
    # 计算EDM权重 λ(σ) = (σ² + σ_data²) / (σ * σ_data)²
    lambda_sigma = (sigma ** 2 + edm_config.sigma_data ** 2) / (sigma * edm_config.sigma_data) ** 2  # lambda_sigma: [B]
    # 添加噪声：x_noisy = x + σ * z
    noise = torch.randn_like(data)
    noisy_data = data + sigma[:, None, None] * noise
    # 模型预测去噪数据 D_θ(x_noisy; σ)
    # 注意：模型现在应该接受sigma而不是时间t
    model_output = model(noisy_data, sigma, class_labels=class_labels)
    # λ(σ)/e^{u(σ)} * ||D_θ(x; σ) - x||²++ u(σ)
    # 5. 计算损失
    target = data
    mse = (model_output - target).pow(2).mean(dim=[1, 2])  # [B]
    # 6. 动态损失加权
    if loss_weight_net is not None:
        u_sigma = loss_weight_net(sigma)
        loss = torch.exp(-u_sigma) * lambda_sigma * mse + u_sigma
    else:
        loss = lambda_sigma * mse
    # 最后返回所有样本的 loss 均值
    return loss.mean()


# ========================================
# ====================  AMPSynth训练过程函数定义====================
# ###  创建带 Warmup 的余弦退火调度器
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """
    创建一个调度器:
    1. Warmup: 学习率从 start_factor * lr 线性增加到 1.0 * lr
    2. Cosine: 学习率从 1.0 * lr 按余弦曲线降到 min_lr
    """
    # 防止 warmup 步数为 0 报错
    if num_warmup_steps == 0:
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=min_lr)
    # 1. Warmup 阶段
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.001,  # 初始学习率只有千分之一
        end_factor=1.0,
        total_iters=num_warmup_steps
    )
    # 2. 余弦退火阶段
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=min_lr
    )
    # 3. 串联
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps]
    )


def train_peptide_AMPSynth(
        model: nn.Module,
        data_loader: DataLoader,
        edm_config: EDMConfig,
        num_epochs: int = 200,
        optimizer: torch.optim.Optimizer = None,
        scheduler: object = None,
        device: str = "cuda",
        esm_encoder: Optional[nn.Module] = None,
        data_from_excel: bool = True,
        use_pf_ema: bool = True,
        loss_weight_net: Optional[nn.Module] = None,
        # ### [CFG 参数]
        use_cfg: bool = False,
        cfg_dropout_prob: float = 0.15,
        null_token_id: int = 0,
        ema_sigma_rel=0.1,
):
    model.to(device)
    model.train()
    if use_pf_ema:
        ema_tracker = PFEMA(model, sigma_rel=ema_sigma_rel, device=device)
        print(f" 启用EMA (σ_rel={ema_sigma_rel:.3f})")
    else:
        ema_tracker = None
        print(" 不使用EMA")
    # 如果传入了 loss_weight_net，也要设为训练模式
    if loss_weight_net is not None:
        loss_weight_net.to(device)
        loss_weight_net.train()
    # 兜底: 如果没传优化器，创建一个包含所有参数的默认优化器
    if optimizer is None:
        print(" 未传入优化器，使用默认配置")
        params = list(model.parameters())
        if loss_weight_net is not None:
            params += list(loss_weight_net.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)
    train_losses = []
    global_step = 0
    print(f"开始训练: Epochs={num_epochs}, CFG={use_cfg}")
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_data in pbar:
            # 1. 解析数据
            # 返回 (seqs, labels)
            if isinstance(batch_data, (tuple, list)):
                sequences, labels = batch_data
            else:
                sequences = batch_data
                labels = None
            # 2. ESM 编码
            if data_from_excel and esm_encoder is not None:
                with torch.no_grad():
                    raw_data = esm_encoder.encode_batch(sequences)  # [B, Seq, Dim]
            else:
                raw_data = sequences.to(device)
            # 3. 标签处理 & CFG Mask
            labels = labels.to(device) if labels is not None else None
            if use_cfg and labels is not None:
                # 生成随机 Mask (True 表示需要丢弃)
                mask = torch.rand(labels.shape[0], device=device) < cfg_dropout_prob
                # 创建副本，将被 mask 的标签设为 Null Token
                train_labels = labels.clone()
                train_labels[mask] = null_token_id
            else:
                # 预训练或无条件模式
                train_labels = labels
            # 4. 优化步骤
            optimizer.zero_grad()
            loss = loss_fn(
                model=model,
                data=raw_data,
                edm_config=edm_config,
                loss_weight_net=loss_weight_net,
                device=device,
                class_labels=train_labels
            )
            if torch.isnan(loss) or torch.isinf(loss):
                print(" Loss NaN/Inf，跳过 Batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if loss_weight_net is not None:
                torch.nn.utils.clip_grad_norm_(loss_weight_net.parameters(), max_norm=1.0)
            optimizer.step()
            # Step-level scheduler (Warmup通常按 step 更新)
            if scheduler:
                scheduler.step()
            if use_pf_ema and ema_tracker is not None:
                ema_tracker.update()
            global_step += 1
            total_loss += loss.item()
            num_batches += 1
            # 更新进度条
            lr_current = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "Loss": f"{loss.item():.5f}",
                "LR": f"{lr_current:.2e}"
            })
        # Epoch 统计
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.6f}")
        # 保存检查点
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            save_name = f"AMPSynth_epoch_{epoch + 1}.pth"
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 如果有 loss_weight_net，也保存它
                'loss_weight_net_state_dict': loss_weight_net.state_dict() if loss_weight_net else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'edm_config': edm_config.to_dict(),
            }
            if use_pf_ema and ema_tracker is not None:
                checkpoint['ema_state_dict'] = ema_tracker.get_state_dict()
                checkpoint['ema_sigma_rel'] = ema_sigma_rel
            # torch.save(checkpoint, save_name) # 根据需要取消注释
    # if use_pf_ema and ema_tracker is not None:
    #     ema_tracker.apply_to_model()
    return model, train_losses


# ==================== AMPSynth 预训练训练过程====================
# ###  1: 预训练执行函数
def setup_pretraining_pipeline(
        excel_path: str = "C:/Users/Mordred/Desktop/AMP-data.xlsx",
        output_dir: str = "C:/Users/Mordred/Desktop/",
        sequence_column: str = "Sequence",
        batch_size: int = 64,
        AMPSynth_epochs: int = 300,
        lr_AMPSynth: float = 1e-5,
        d_model: int = 1280,
        # Decoder 训练参数
        decoder_epochs: int = 50,
        lr_decoder: float = 1e-4,
        device: str = "cuda"
):
    """
    功能:
    1. 强制无条件 (Null Token) 训练AMPSynth (含 ScoreNet + LossNet)
    2. 自动训练 ESM Decoder
    3. 保存所有必要的权重文件
    """
    print(f"\n{'=' * 20} 启动完整预训练流程  {'=' * 20}")
    os.makedirs(output_dir, exist_ok=True)
    # ==================== 第一阶段: 训练 EDM2 ====================
    print("\n>>> 阶段 1/2: 训练AMPSynth 扩散模型")
    # 1. 准备数据 (强制 Null 标签)
    train_loader, num_classes, null_id = prepare_training_from_excel(
        excel_path=excel_path,
        sequence_column=sequence_column,
        label_column=None,
        batch_size=batch_size,
        fix_null_token=True  #预训练强制为 Null
    )
    # 2. 计算并保存配置
    try:
        sigma_data = compute_sigma_data(excel_path, sequence_column)
    except:
        sigma_data = 0.284
        print("使用默认 sigma_data=0.284")
    edm_config = EDMConfig(sigma_data=sigma_data)
    # 保存 Config 供推理使用
    with open(os.path.join(output_dir, "edm_config.json"), 'w') as f:
        json.dump({'edm_config': edm_config.to_dict()}, f, indent=2)
    # 3. 初始化模型 (DiT + LossNet)
    model = ScoreNetwork(
        edm_config=edm_config,
        num_classes=num_classes,  # 通常为1
        d_model=d_model
    )
    loss_weight_net = DynamicLossWeighting(fourier_dim=128).to(device)
    # 4. 优化器与调度器
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': lr_AMPSynth},
        {'params': loss_weight_net.parameters(), 'lr': lr_AMPSynth}
    ], weight_decay=1e-4)
    total_steps = len(train_loader) * AMPSynth_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.05), total_steps)
    # 5. 执行训练
    # 始终冻结 ESM
    esm_encoder = ESMEncoderWrapper(device=device)
    trained_AMPSynth, AMPSynth_losses = train_peptide_AMPSynth(
        model=model,
        data_loader=train_loader,
        edm_config=edm_config,
        num_epochs=AMPSynth_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        esm_encoder=esm_encoder,
        use_cfg=False,  # 预训练不需要随机丢弃
        null_token_id=null_id,
        loss_weight_net=loss_weight_net
    )
    # 6. 保存AMPSynth权重
    torch.save(trained_AMPSynth.state_dict(), os.path.join(output_dir, "pretrained_AMPSynth.pth"))
    torch.save(loss_weight_net.state_dict(), os.path.join(output_dir, "pretrained_lwn.pth"))
    print(" AMPSynth模型权重已保存")
    # ==================== 第二阶段: 训练 ESM Decoder ====================
    print("\n>>> 阶段 2/2: 训练 ESM Decoder (语言头)")
    # 1. 初始化 Decoder
    esm_decoder = ESMLanguageHeadFinetuner(
        esm_encoder=esm_encoder,
        edm_output_dim=d_model,
        max_seq_len=52,  # 可根据数据调整
        device=device
    )
    # 2. 准备 Decoder 数据 (需要原始序列 -> Latent)
    # 重新读取序列列表
    raw_loader = ExcelAMPDataLoader(excel_path, sequence_column)
    sequences = raw_loader.get_sequences()
    print("正在生成 Decoder 训练特征 (Encoding)...")
    z_train, targets_train, masks_train = prepare_esm_finetuning_data(
        esm_encoder, sequences, device
    )
    decoder_dataset = torch.utils.data.TensorDataset(z_train, targets_train, masks_train)
    decoder_loader = DataLoader(decoder_dataset, batch_size=batch_size, shuffle=True)
    # 3. 执行 Decoder 训练
    # 这是一个独立的训练循环，你可以直接用你原来的 train_esm_language_head
    trained_decoder, decoder_losses = train_esm_language_head(
        esm_decoder=esm_decoder,
        train_loader=decoder_loader,
        num_epochs=decoder_epochs,
        lr=lr_decoder,
        device=device
    )
    # 4. 保存 Decoder 权重
    torch.save(trained_decoder.state_dict(), os.path.join(output_dir, "pretrained_decoder.pth"))
    print("Decoder 权重已保存")
    print(f"\n 预训练流程结束。所有文件位于: {output_dir}")
    return trained_AMPSynth, trained_decoder, AMPSynth_losses, decoder_losses, edm_config


### ====================  微调训练管道  ====================

def setup_finetuning_pipeline(
        excel_path: str = "Grampa_mic_filtered.xlsx",  # 你的定向高活性/特定细菌数据集
        output_dir: str = "./checkpoints_finetune",
        sequence_column: str = "Sequence",
        label_column: str = "label",  # 必须指定: Excel中包含细菌名称的列
        # 权重路径 (指向第一阶段产出)
        pretrained_model_path: str = "./checkpoints_pretrain/pretrained_AMPSynth.pth",
        pretrained_lwn_path: str = "./checkpoints_pretrain/pretrained_lwn.pth",
        pretrained_edm_config: str ="E:/Users/Mordred/Desktop/Pretrain_Output",
        # 微调参数
        batch_size: int = 32,
        AMPSynth_epochs: int = 50,  # 微调通常不需要太多轮
        lr_finetune: float = 1e-4,
        d_model: int = 1280,
        device: str = "cuda",
):
    """
    完整微调管道
    功能:
    1. 加载预训练的 EDM2 权重 (含 ScoreNet + LossNet)
    2. 执行层级冻结策略 (Freeze 80%) 和 Embedding 迁移
    3. 开启 CFG (无分类器引导) 进行定向训练
    """
    print(f"\n{'=' * 20} 启动微调流程 (AMPSynth Fine-tuning) {'=' * 20}")
    if not os.path.exists(pretrained_model_path):
        raise FileNotFoundError(f" 找不到预训练权重 {pretrained_model_path}\n请先运行预训练管道！")
    os.makedirs(output_dir, exist_ok=True)
    # ==================== 步骤 1: 数据准备 (真实标签) ====================
    print("\n>>> 步骤 1: 准备微调数据")
    # fix_null_token=False: 告诉 Loader 读取并编码真实的细菌标签
    train_loader, num_classes, null_id = prepare_training_from_excel(
        excel_path=excel_path,
        sequence_column=sequence_column,
        label_column=label_column,  # 传入列名
        batch_size=batch_size,
        fix_null_token=False
    )
    print(f"   - 检测到 {num_classes - 1} 种细菌标签")
    print(f"   - Null Token ID: {null_id}")
    # ==================== 步骤 2: 初始化模型 ====================
    print("\n>>> 步骤 2: 初始化模型架构")
    # 默认配置，sigma_data 保持一致

    config_path = os.path.join(pretrained_edm_config, "edm_config.json")

    # 2. 读取参数
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            saved_config = json.load(f)

        print(f" 成功AMPSynth配置: {saved_config}")
        if "edm_config" in saved_config:
            # 如果配置被包裹在 "edm_config" 键里，就提取里面那一层
            actual_params = saved_config["edm_config"]
        else:
            # 如果是直接保存的扁平结构，就直接使用
            actual_params = saved_config
        # 使用提取出来的真实参数进行解包
        edm_config = EDMConfig(**actual_params)
    else:
        print(f" 未在 {pretrained_edm_config} 找到 edm_config.json！")
        edm_config = EDMConfig(sigma_data=0.2840735597742928)

    # 初始化主模型 (DiT)
    # 注意: 这里的 num_classes (如5) 比预训练时 (1) 要大
    model = ScoreNetwork(
        edm_config=edm_config,
        num_classes=num_classes,
        d_model=d_model
    )
    # 初始化动态权重网络
    loss_weight_net = DynamicLossWeighting(fourier_dim=128).to(device)
    # ==================== 步骤 3: 加载预训练权重====================
    print("\n>>> 步骤 3: 加载预训练权重 & 迁移 Embedding")
    # (A) 加载主模型 (含 Embedding 维度适配)
    state_dict = torch.load(pretrained_model_path, map_location=device)
    model_dict = model.state_dict()
    # 1. 筛选出形状匹配的参数 (DiT Blocks, Final Layer 等绝大多数参数)
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    # 2. 处理 Label Embedding (预训练 [1, D] -> 微调 [N, D])
    if 'label_embedder.weight' in state_dict:
        old_emb = state_dict['label_embedder.weight']
        # 确认预训练确实是单 Null Token 模式
        if old_emb.shape[0] == 1:
            with torch.no_grad():
                # 将预训练学到的 "通用 Null 特征" 复制给微调模型的 "Null Token" 位置
                # 这样微调一开始，Null Token 的表现就和预训练时一致
                model.label_embedder.weight[null_id] = old_emb[0]
            print(f"    已成功迁移 Null Token 向量至 ID {null_id}")

    # 3. 更新权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print("    主模型权重加载完毕")
    # (B) 加载并冻结 Loss Weight Net
    if os.path.exists(pretrained_lwn_path):
        loss_weight_net.load_state_dict(torch.load(pretrained_lwn_path, map_location=device))
        print("    Loss Weight Net 权重加载完毕")
    else:
        print("   未找到 Loss权重，将使用随机初始化 (可能影响 AMPSynth效果)")
    # 冻结它！微调阶段不应该改变损失权重的分布
    loss_weight_net.eval()
    for param in loss_weight_net.parameters():
        param.requires_grad = False
    print("    Loss Weight Net 冻结")
    # ==================== 步骤 4: 层级冻结策略 (80% Rule) ====================
    print("\n>>> 步骤 4: 执行层级冻结策略")
    # 1. 先把所有门关上 (全冻结)
    for param in model.parameters():
        param.requires_grad = False
    # 2. 打开必须训练的门
    # (a) 标签嵌入层: 必须训练，否则学不到细菌特征
    for param in model.label_embedder.parameters():
        param.requires_grad = True
    # (b) 输出层: 必须微调以适应新分布
    for param in model.final_layer.parameters():
        param.requires_grad = True
    # (c) 解冻最后 20% 的 Transformer Block
    total_layers = len(model.blocks)
    start_layer = 4  # 第0,1,2,3层为前4层，第4,5层为最后2层
    # (c) 前 4 层：冻结注意力机制，僅僅解冻 adaLN_modulation (打通條件通道)
    for i in range(0, start_layer):
        for param in model.blocks[i].adaLN_modulation.parameters():
            param.requires_grad = True
    # (d) 最后 2 层：全面解冻
    for i in range(start_layer, total_layers):
        for param in model.blocks[i].parameters():
            param.requires_grad = True
    # 统计一下
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    冻结了前 {start_layer} 层 (0-{start_layer - 1})，仅解冻 adaLN 条件注入通道")
    print(f"    解冻了后 {total_layers - start_layer} 层 ({start_layer}-{total_layers - 1})")
    print(f"    当前可微调参数量: {n_params:,}")

    # ==================== 步骤 5: 开始微调 ====================
    print("\n>>> 步骤 5: 启动训练循环 (CFG Enabled)")
    # 优化器: 只传入解冻的参数
    params_to_optimize = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr_finetune}
    ]
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        weight_decay=1e-4  # 加上权重衰减防止过拟合
    )
    # 调度器: 10% Warmup
    total_steps = len(train_loader) * AMPSynth_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    # ESM 编码器 (始终冻结)
    esm_encoder = ESMEncoderWrapper(device=device)
    # 训练
    trained_model, AMPSynth_losses = train_peptide_AMPSynth(
        model=model,
        data_loader=train_loader,
        edm_config=edm_config,
        num_epochs=AMPSynth_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        esm_encoder=esm_encoder,
        use_cfg=True,  # 开启 CFG 模式
        cfg_dropout_prob=0.15,  # 15% 概率丢弃标签
        null_token_id=null_id,
        loss_weight_net=loss_weight_net  # 传入冻结的网络用于计算 Loss
    )
    # ==================== 步骤 6: 保存结果 ====================
    save_path = os.path.join(output_dir, "finetuned_AMPSynth.pth")
    torch.save(trained_model.state_dict(), save_path)
    print(f"\n 微调流程结束。")
    print(f" 权重已保存至: {save_path}")
    return trained_model, AMPSynth_losses
# ==================== AMPSynth主要训练过程====================




# ==================== AMPSynth采样器 ====================3
class AMPSynthSampler:
    """
    随机采样器
    """
    def __init__(self,
                 edm_config: EDMConfig,
                 device: str = "cuda",
                 S_churn: float = 40.0,  # S_churn一般为step的0.4以下
                 S_tmin: float = 0.002,
                 S_tmax: float = 80.0,  # S_tmin用于限制在S_tmin到S_tmax2范围内，才增添和删除噪声
                 S_noise: float = 1.003):
        """
        Args:
            edm_config: 配置参数
            device: 计算设备
            S_churn: 噪声搅动强度参数（
            S_tmin: 最小噪声水平阈值
            S_tmax: 最大噪声水平阈值
            S_noise: 噪声标准差缩放因子
        """
        self.edm_config = edm_config
        self.device = device
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    def get_time_steps(self, num_steps: int) -> torch.Tensor:
        """
        Args:
            num_steps: 采样步数
        Returns:
            timesteps: 时间步张量 [num_steps + 1]（包含t=0）
        """
        # 推荐的时间步调度（ρ=7）
        step_indices = torch.arange(num_steps, device=self.device)
        # 公式5: σ_i = (σ_max^(1/ρ) + i/(N-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
        sigma_i = (
                          self.edm_config.sigma_max ** (1 / self.edm_config.rho) +
                          step_indices / (num_steps - 1) *
                          (self.edm_config.sigma_min ** (1 / self.edm_config.rho) -
                           self.edm_config.sigma_max ** (1 / self.edm_config.rho))) ** self.edm_config.rho

        # 添加最终步σ=0
        sigma_full = torch.cat([sigma_i, torch.tensor([0.0], device=self.device)])
        return sigma_full

    def sample(self,
               model: nn.Module,
               sample_shape: Tuple[int, ...],
               num_steps: int = 160,  # 采样步数
               denoise: bool = True,
               class_labels: Optional[torch.Tensor] = None,
               guidance_scale: float = 1.0,  # 1.0 表示不使用引导,
               null_token_id: int = 0) -> torch.Tensor:
        """
        EDM  随机采样主函数
        Args:
            model: 分数网络
            sample_shape: 样本形状 [batch_size, seq_len, hidden_dim]
            num_steps: 采样步数
            denoise: 是否在最后去噪
        Returns:
            generated_samples: 生成的样本
        """
        model.eval()
        batch_size = sample_shape[0]
        # 1. 生成时间步序列
        timesteps = self.get_time_steps(num_steps)  # [num_steps + 1]
        print(f" 使用EDM时间步调度: {len(timesteps)}步, σ范围: [{timesteps[0]:.3f}, {timesteps[-2]:.3f}]")
        # 2. 从先验分布采样初始噪声
        x = torch.randn(sample_shape, device=self.device) * self.edm_config.sigma_max
        print(f" 初始噪声采样: {x.shape}, σ={self.edm_config.sigma_max}")

        def get_model_output(x_in, sigma_in_scalar):
            # 扩展 sigma 维度
            if isinstance(sigma_in_scalar, torch.Tensor) and sigma_in_scalar.dim() == 0:
                sigma_batch = sigma_in_scalar.expand(x_in.shape[0])
            else:
                sigma_batch = sigma_in_scalar
            # 情况 A: 无条件 / 无引导 (兼容旧逻辑)
            if class_labels is None or guidance_scale == 1.0:
                return model(x_in, sigma_batch, class_labels=class_labels)
            # 情况 B: CFG 引导
            else:
                # 1. 拼接输入 [Cond, Uncond]
                x_double = torch.cat([x_in, x_in])
                sigma_double = torch.cat([sigma_batch, sigma_batch])
                # 2. 构造标签 [Label, Null]
                null_labels = torch.full_like(class_labels, null_token_id)
                labels_double = torch.cat([class_labels, null_labels])
                # 3. 一次性推理
                out_double = model(x_double, sigma_double, class_labels=labels_double)
                cond_out, uncond_out = out_double.chunk(2)
                # 4. 应用公式: Uncond + w * (Cond - Uncond)
                return uncond_out + guidance_scale * (cond_out - uncond_out)

        # 3. 随机采样循环
        with torch.no_grad():
            for i in tqdm(range(num_steps), desc="Sampling"):
                # 当前和下一个时间步
                sigma_cur = timesteps[i]
                sigma_next = timesteps[i + 1]
                # 只在特定噪声范围内启用随机性
                if self.S_tmin <= sigma_cur <= self.S_tmax:
                    gamma = min(self.S_churn / num_steps, 2 ** 0.5 - 1)
                else:
                    gamma = 0.0
                # 噪声搅动步骤
                if gamma > 0:
                    # 计算增大的噪声水平 σ_hat = σ_cur + γ * σ_cur
                    sigma_hat = sigma_cur * (1 + gamma)
                    # 添加额外噪声
                    eps = torch.randn_like(x) * self.S_noise
                    x_perturbed = x + torch.sqrt(sigma_hat ** 2 - sigma_cur ** 2) * eps
                else:
                    sigma_hat = sigma_cur
                    x_perturbed = x
                # 一阶欧拉步预测（Algorithm 2 第7-8行）
                # 计算得分函数: d = (x - D_θ(x; σ)) / σ

                if sigma_hat.dim() == 0:  # 标量
                    sigma_hat_batch = sigma_hat.expand(batch_size)  # [batch_size]
                else:
                    sigma_hat_batch = sigma_hat
                if sigma_next.dim() == 0:  # 标量
                    sigma_next_batch = sigma_next.expand(batch_size)  # [batch_size]
                else:
                    sigma_next_batch = sigma_next

                denoised_cur = get_model_output(x_perturbed, sigma_hat_batch)
                d_cur = (x_perturbed - denoised_cur) / sigma_hat  # 放入模型sigma必须是二维
                x_euler = x_perturbed + (sigma_next - sigma_hat) * d_cur
                # 二阶Heun校正
                if sigma_next > 0:  # 不是最后一步
                    # 在预测点计算得分
                    denoised_next = get_model_output(x_euler, sigma_next_batch)
                    d_next = (x_euler - denoised_next) / sigma_next  # 放入模型sigma必须是二维
                    # 应用Heun二阶校正
                    x = x_perturbed + (sigma_next - sigma_hat) * (d_cur + d_next) / 2
                else:
                    # 最后一步使用欧拉预测
                    x = x_euler
        # 4. 最终去噪
        if denoise and sigma_next > 0:
            print(" 执行最终去噪步骤...")
            with torch.no_grad():
                # 使用模型进行最终去噪
                x = get_model_output(x, torch.tensor(0.0, device=self.device))
        return x


# ==================== 采样器 ====================



# ==================== 生成函数 ====================
@torch.no_grad()
def generate_antimicrobial_peptides(
        score_model: nn.Module,
        esm_decoder: ESMLanguageHeadFinetuner,
        num_sequences: int = 100,
        max_seq_len: int = 50,
        latent_seq_len: int = 52,  # max_seq_len+2
        batch_size: int = 64,
        num_steps: int = 160,
        temperature: float = 1.0,
        device: str = "cuda",
        output_excel_path: Optional[str] = None,
        edm_config: Optional[EDMConfig] = None,
        target_bacteria_id: Optional[int] = None,  # 目标细菌的整数ID
        guidance_scale: float = 1.0,  # CFG 引导强度 (1.0=关闭, >1.0=开启)
        null_token_id: int = 0  # Null 标签的 ID (需与训练时一致)
) -> List[str]:
    """
    统一的抗菌肽生成
    模式说明:
    1. 无条件生成: 不传 target_bacteria_id (默认为 None)。
    2. 定向生成: 传入 target_bacteria_id (例如 2) 和 guidance_scale (例如 3.0)。
    """

    # 设置模型为评估模式
    score_model.eval()
    esm_decoder.eval()
    # 默认EDM配置
    if edm_config is None:
        edm_config = EDMConfig(
            sigma_data=0.2840735597742928,  # 需要根据实际训练的数据计算
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0
        )
    # 初始化EDM采样器
    sampler = AMPSynthSampler(
        edm_config=edm_config,
        device=device,
        S_churn=40.0,
        S_tmin=0.002,
        S_tmax=80,
        S_noise=1.03)

    all_generated_sequences = []
    # 批量生成支持
    num_batches = (num_sequences + batch_size - 1) // batch_size
    print(f" 开始生成: 总数={num_sequences}, 目标ID={target_bacteria_id}, CFG强度={guidance_scale}")

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_sequences - batch_idx * batch_size)
        # ###  构造条件标签
        if target_bacteria_id is not None:
            # 创建一个填满目标ID的张量 [Batch]
            batch_labels = torch.full(
                (current_batch_size,),
                target_bacteria_id,
                device=device,
                dtype=torch.long
            )
        else:
            batch_labels = None
        # 4. 采样潜在变量
        sample_shape = (current_batch_size, latent_seq_len, score_model.d_model)
        latent_vectors = sampler.sample(
            model=score_model,
            sample_shape=sample_shape,
            num_steps=num_steps,
            denoise=True,
            # 传入新增参数
            class_labels=batch_labels,
            guidance_scale=guidance_scale,
            null_token_id=null_token_id
        )
        # 5. 解码为序列 (Latent -> Sequence)
        batch_sequences = esm_decoder.generate_sequences(
            latent_embeddings=latent_vectors,
            max_length=max_seq_len,
            temperature=temperature
        )
        all_generated_sequences.extend(batch_sequences)
        # 简单的进度打印
        print(f"  Batch {batch_idx + 1}/{num_batches} 完成")
        # 6. 保存结果
    if output_excel_path:
        label_suffix = f"_Target{target_bacteria_id}" if target_bacteria_id is not None else ""
        save_sequences_to_excel(all_generated_sequences, output_excel_path, suffix=label_suffix)
    return all_generated_sequences


def save_sequences_to_excel(sequences: List[str], file_path: str, suffix: str = ""):
    """
    保存序列到 Excel
    """
    # 1. 验证输入
    if not sequences:
        print("没有序列需要保存")
        return False
    # 2. 处理文件路径
    # 如果给的是目录 (例如 "./results")，自动生成文件名
    if os.path.isdir(file_path) or not file_path.endswith(('.xlsx', '.xls')):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_AMPs{suffix}_{timestamp}.xlsx"

        if os.path.isdir(file_path):
            file_path = os.path.join(file_path, filename)
        else:
            # 如果输入是 "my_result"，变成 "my_result.xlsx"
            file_path = file_path + ".xlsx"

    # 3. 创建父目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 4. 创建 DataFrame
    df = pd.DataFrame({
        'Sequence': sequences,
        'Length': [len(seq) for seq in sequences],
        'Type': 'Generated_AMP',
        'Target_ID': suffix.replace("_Target", "") if suffix else "Uncond",
        'Timestamp': pd.Timestamp.now()
    })
    # 5. 保存
    try:
        df.to_excel(file_path, index=False, engine='openpyxl')
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"成功保存 {len(sequences)} 条序列至: {file_path}")
            return True
    except Exception as e:
        print(f" 保存失败: {e}")
        return False


# ==================== 生成函数 ====================


### ==================== 主程序 A: 预训练与无条件生成 (Pretrain & Uncond Gen) ====================
def main_pretrain_uncond():
    """
    脚本 A: 抗菌肽生成模型 - 预训练与无条件生成全流程
    功能:
    1. 从 Excel 读取数据，进行 AMPSynth预训练 (无条件) 和 ESM Decoder 训练。
    2. 绘制并保存训练 Loss 曲线。
    3. 立即使用训练好的模型进行无条件生成。
    4. 保存生成结果和模型权重。
    """
    # ==================== 1. 配置参数 ====================
    config = {
        # --- 数据路径 ---
        'excel_path': "C:/Users/Mordred/Desktop/AMP.xlsx",
        'output_dir': "C:/Users/Mordred/Desktop/Pretrain_Output",  # 建议分开文件夹
        'sequence_column': "Sequence",
        # --- 硬件与训练参数 ---
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'batch_size': 64,
        'd_model': 1280,  # 模型维度
        # --- EDM 预训练参数 ---
        'AMPSynth_epochs': 200,  # 演示用20，原本设置为200
        'edm_lr': 1e-5,  # EDM2 学习率
        # --- Decoder 训练参数 ---
        'decoder_epochs': 50,  # Decoder 训练轮数
        'decoder_lr': 1e-4,
        # --- 生成参数 (无条件) ---
        'num_sequences_to_generate': 5000,
        'num_steps': 160,  # 采样步数
        'max_seq_len': 50,
        'latent_seq_len': 52,  # 通常是 max_seq_len + 2 (BOS/EOS)
        'temperature': 1.0,
        'guidance_scale': 1.0  # 无条件生成不需要引导
    }
    print("=" * 60)
    print("  预训练 + 无条件生成")
    print("=" * 60)
    os.makedirs(config['output_dir'], exist_ok=True)

    try:
        # ==================== 阶段 1: 完整预训练管道 ====================
        print("\n>>> 阶段 1: 执行预训练管道")

        # 调用我们写好的 setup_pretraining_pipeline
        # 注意: 该函数内部已经保存了 .pth 权重文件
        # 我们修改了它的返回值，让它返回训练好的对象以便直接使用
        trained_AMPSynth, trained_decoder, AMPSynth_losses, decoder_losses, edm_config = setup_pretraining_pipeline(
            excel_path=config['excel_path'],
            output_dir=config['output_dir'],
            sequence_column=config['sequence_column'],
            batch_size=config['batch_size'],
            AMPSynth_epochs=config['AMPSynth_epochs'],
            AMPSynth_edm=config['AMPSynth_lr'],
            d_model=config['d_model'],
            # Decoder 参数
            decoder_epochs=config['decoder_epochs'],
            lr_decoder=config['decoder_lr'],
            device=config['device']
        )
        print(" 预训练管道执行完毕!")
        # ==================== 阶段 2: 绘制 Loss 曲线 ====================
        print("\n>>> 阶段 2: 绘制训练 Loss")
        if len(AMPSynth_losses) > 0 and len(decoder_losses) > 0:
            try:
                plt.figure(figsize=(12, 5))
                # EDM Loss
                plt.subplot(1, 2, 1)
                plt.plot(AMPSynth_losses, label='AMPGen Loss')
                plt.title('AMPSynth Pre-training')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True, alpha=0.3)
                plt.legend()
                # Decoder Loss
                plt.subplot(1, 2, 2)
                plt.plot(decoder_losses, label='Decoder Loss', color='orange')
                plt.title('ESM Decoder Training ')
                plt.xlabel('Epoch')
                plt.ylabel('loss')
                plt.grid(True, alpha=0.3)
                plt.legend()
                loss_plot_path = os.path.join(config['output_dir'], 'training_losses.png')
                plt.tight_layout()
                plt.savefig(loss_plot_path, dpi=300)
                plt.close()
                print(f"  Loss 曲线已保存: {loss_plot_path}")
            except Exception as e:
                print(f" ⚠ 绘图失败: {e}")
        # ==================== 阶段 3: 无条件生成 ====================
        print("\n>>> 阶段 3: 执行无条件生成 (Unconditional Generation)")
        output_excel_path = os.path.join(config['output_dir'], "generated_AMPs_uncond.xlsx")
        generated_sequences = generate_antimicrobial_peptides(
            score_model=trained_AMPSynth,
            esm_decoder=trained_decoder,
            num_sequences=config['num_sequences_to_generate'],
            num_steps=config['num_steps'],
            max_seq_len=config['max_seq_len'],
            latent_seq_len=config['latent_seq_len'],
            batch_size=config['batch_size'],
            temperature=config['temperature'],
            device=config['device'],
            output_excel_path=output_excel_path,
            edm_config=edm_config,
            # ★ 关键: 无条件生成参数
            target_bacteria_id=None,  # 不指定细菌
            guidance_scale=1.0,  # 不引导
            null_token_id=0  # 预训练时的 Null ID (通常是0，具体看 prepare_training 返回)
        )
        # ==================== 阶段 4: 结果统计 ====================
        valid_sequences = [s for s in generated_sequences if 5 <= len(s) <= 50]
        print("\n" + "=" * 60)
        print(f"  任务完成 Summary")
        print("=" * 60)
        print(f"  输出目录: {config['output_dir']}")
        print(f"  生成文件: {output_excel_path}")
        print(f"  生成统计: 总数 {len(generated_sequences)}, 有效 {len(valid_sequences)}")
        if valid_sequences:
            # 1. 先提取所有长度，避免重复计算
            seq_lengths = [len(s) for s in valid_sequences]
            # 2. 计算各项统计指标
            avg_len = sum(seq_lengths) / len(valid_sequences)
            min_len = min(seq_lengths)
            max_len = max(seq_lengths)
            # 3. 打印结果
            print(f"  平均长度: {avg_len:.1f}")
            # 注意: 长度通常是整数，所以这里去掉了 .1f，当然加上也可以
            print(f"  长度范围: {min_len}-{max_len}")
            print("  示例序列 (前10条):")
            for i, seq in enumerate(valid_sequences[:10]):
                print(f"   {i + 1}. {seq}")
        return {'status': 'success'}
    except Exception as e:
        print(f"\n❌ 程序崩溃: {e}")
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


# if __name__ == "__main__":
#     main_pretrain_uncond()


### ==================== 主程序 B: 微调与条件生成 (Finetune & Cond Gen) ====================

### ==================== 主程序 B: 微调 + 多目标条件生成 (含Loss图与有效率统计) ====================

def main_finetune_condition():
    """
微调 + 多目标条件生成
    功能:
    1. 执行AMPSynth微调 & 绘制 Loss 曲线。
    2. 加载预训练 Decoder。
    3. 循环生成不同细菌序列，统计有效数量。
    4. 结果保存到同一个 Excel 的不同 Sheet。
    """
    # ==================== 1. 配置参数 ====================
    config = {
        # --- 路径配置 ---
        'pretrained_dir': "E:/Users/Mordred/Desktop/Pretrain_Output",
        'pretrained_AMPSynth': "pretrained_AMPSynth.pth",
        'pretrained_lwn': "pretrained_lwn.pth",
        'pretrained_decoder': "pretrained_decoder.pth",
        'excel_path': "E:/Users/Mordred/Desktop/grampa_gen.xlsx",
        'output_dir': "E:/Users/Mordred/Desktop/Finetune_Output",
        'sequence_column': "Sequence",
        'label_column': "label",
        # --- 训练参数 ---
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'batch_size': 64,
        'd_model': 1280,
        'finetune_epochs': 200,
        'finetune_lr': 1e-4,
        # --- 全局生成参数 ---
        'num_steps': 160,
        'max_seq_len': 50,
        'latent_seq_len': 52,
        'temperature': 1.0,
        'null_token_id': 4,
        # --- 多目标生成配置 ---
        'targets': [
            {'name': 'E.coli', 'id': 0, 'scale': 4, 'num': 2000},
            {'name': 'S.aureus', 'id': 1, 'scale': 4, 'num': 2000},
            {'name': 'P.aeruginosa', 'id': 2, 'scale': 4.5, 'num': 2000},
            {'name': 'B.subtilis', 'id': 3, 'scale': 4.5, 'num': 2000},
        ]
    }

    # 路径拼接
    path_AMPSynth = os.path.join(config['pretrained_dir'], config['pretrained_AMPSynth'])
    path_lwn = os.path.join(config['pretrained_dir'], config['pretrained_lwn'])
    path_dec = os.path.join(config['pretrained_dir'], config['pretrained_decoder'])

    print("=" * 60)
    print("  微调 + 多目标条件生成 ")
    print("=" * 60)

    os.makedirs(config['output_dir'], exist_ok=True)

    try:
        # ==================== 阶段 1: 执行微调 ====================
        print("\n>>> 阶段 1: 执行 AMPSynth 微调")
        finetuned_AMPSynth, finetune_losses = setup_finetuning_pipeline(
            excel_path=config['excel_path'],
            output_dir=config['output_dir'],
            sequence_column=config['sequence_column'],
            label_column=config['label_column'],
            pretrained_model_path=path_edm,
            pretrained_lwn_path=path_lwn,
            batch_size=config['batch_size'],
            AMPSynth_epochs=config['finetune_epochs'],
            lr_finetune=config['finetune_lr'],
            d_model=config['d_model'],
            device=config['device']
        )
        print(" 微调完成")
        # 绘制微调 Loss 曲线
        if len(finetune_losses) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(finetune_losses, label='Finetune Loss', color='green')
            plt.title('Diff Fine-tuning Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            loss_path = os.path.join(config['output_dir'], 'finetune_loss.png')
            plt.savefig(loss_path, dpi=300)
            plt.close()
            print(f"  Loss 曲线已保存: {loss_path}")

        # ==================== 阶段 2: 加载 Decoder ====================
        print("\n>>> 阶段 2: 加载 Decoder")
        esm_encoder = ESMEncoderWrapper(device=config['device'])
        esm_decoder = ESMLanguageHeadFinetuner(
            esm_encoder=esm_encoder,
            edm_output_dim=config['d_model'],
            max_seq_len=52,
            device=config['device']
        )

        if os.path.exists(path_dec):
            checkpoint = torch.load(path_dec, map_location=config['device'])
            # 2. 判断它是不是一个包含了 'model_state_dict' 的大字典
            if 'model_state_dict' in checkpoint:
                # 提取真正的模型权重
                actual_state_dict = checkpoint['model_state_dict']
            else:
                # 如果已经是纯权重，就直接用
                actual_state_dict = checkpoint

            # 3. 再把纯权重喂给解码器
            esm_decoder.load_state_dict(actual_state_dict)
            esm_decoder.eval()
        else:
            raise FileNotFoundError(f"找不到 Decoder: {path_dec}")

        # ==================== 阶段 3: 多目标生成 & 统计 ====================
        print("\n>>> 阶段 3: 开始多目标生成任务")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        final_excel_path = os.path.join(config['output_dir'], f"MultiTarget_Gen_{timestamp}.xlsx")

        with pd.ExcelWriter(final_excel_path, engine='openpyxl') as writer:

            for target in config['targets']:
                bac_name = target['name']
                bac_id = target['id']
                scale = target['scale']
                num_seq = target['num']

                print(f"\n--- 正在生成: {bac_name} (ID={bac_id}) ---")
                print(f"    Target: {num_seq} 条 | CFG Scale: {scale}")

                # 1. 生成
                edm_config = EDMConfig(sigma_data=0.2840735597742928)
                seqs = generate_antimicrobial_peptides(
                    score_model=finetuned_AMPSynth,
                    esm_decoder=esm_decoder,
                    num_sequences=num_seq,
                    num_steps=config['num_steps'],
                    max_seq_len=config['max_seq_len'],
                    latent_seq_len=config['latent_seq_len'],
                    batch_size=config['batch_size'],
                    temperature=config['temperature'],
                    device=config['device'],
                    output_excel_path=None,
                    edm_config=edm_config,
                    target_bacteria_id=bac_id,
                    guidance_scale=scale,
                    null_token_id=config['null_token_id']
                )
                #  统计有效性 (5 <= len <= 50)
                valid_seqs = [s for s in seqs if 5 <= len(s) <= config['max_seq_len']]
                valid_count = len(valid_seqs)
                valid_rate = (valid_count / len(seqs)) * 100
                seq_lengths = [len(s) for s in valid_seqs]
                min_len = min(seq_lengths)
                max_len = max(seq_lengths)
                print(f"     生成完毕")
                print(f"     统计: 总数 {len(seqs)} | 有效 {valid_count} ({valid_rate:.1f}%)")
                print(f"  长度范围: {min_len}-{max_len}")
                if valid_seqs:
                    print(f"     示例: {valid_seqs[0]}")
                # 2. 保存 Sheet
                df = pd.DataFrame({
                    'Sequence': seqs,
                    'Length': [len(s) for s in seqs],
                    'Is_Valid': [5 <= len(s) <= config['max_seq_len'] for s in seqs],  # 标记是否有效
                    'Bacterium': bac_name,
                    'CFG_Scale': scale
                })
                sheet_name = bac_name[:30]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"    已写入 Sheet: {sheet_name}")
        print("\n" + "=" * 60)
        print(f"  全部任务完成")
        print(f"  结果保存: {final_excel_path}")
        print("=" * 60)
        return {'status': 'success'}
    except Exception as e:
        print(f"\n 程序崩溃: {e}")
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


# if __name__ == "__main__":
#     main_finetune_condition()


### ==================== 主程序 C: 纯推理 - 导入预训练模型进行无条件生成 AMPSynth====================
def main_inference_unconditional():
    """
    脚本 C: 纯推理 (Inference Only) - 无条件生成
    功能:
    1. 不进行训练。
    2. 加载预训练好的 EDM 和 Decoder 权重。
    3. 批量生成通用抗菌肽序列 (无条件)。
    4. 结果保存为 Excel。
    """
    # ==================== 1. 推理配置 ====================
    config = {
        # --- 权重路径 (指向预训练产出) ---
        'weights_dir': "E:/Users/Mordred/Desktop/Pretrain_Output",
        'AMPSynth_weight': "pretrained_AMPSynth.pth",
        'decoder_weight': "pretrained_decoder.pth",
        # --- 输出配置 ---
        'output_dir': "E:/Users/Mordred/Desktop/Inference_Results1",
        'output_name': "Unconditional_Generated_AMPs.xlsx",
        # --- 模型参数 (必须与训练时一致) --- 
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'd_model': 1280,
        # 预训练如果是强制 Null Token，num_classes 通常为 1 或数据集中类别的最大数
        # 如果报错 size mismatch，请尝试修改此值为预训练时的类别数
        'num_classes': 1,
        # --- 生成参数 ---
        'num_sequences': 2000,  # 想要生成的数量
        'batch_size': 64,  # 显存够大可以调大
        'num_steps': 30,  # 采样步数
        'max_seq_len': 50,
        'latent_seq_len': 52,
        'temperature': 1,
    }
    # 构造路径
    path_AMPSynth = os.path.join(config['weights_dir'], config['AMPSynth_weight'])
    path_dec = os.path.join(config['weights_dir'], config['decoder_weight'])
    print("=" * 60)
    print("  导入预训练模型 -> 无条件生成")
    print("=" * 60)
    os.makedirs(config['output_dir'], exist_ok=True)
    try:
        # ==================== 阶段 1: 加载模型 ====================
        print("\n>>> 阶段 1: 加载预训练模型")
        # 1. 初始化 EDM Config
        config_path = os.path.join(config['weights_dir'], "edm_config.json")
        with open(config_path, 'r') as f:
            # 提取保存时的字典
            saved_config_dict = json.load(f)['edm_config']

        # 召唤 EDMConfig 的 from_dict 方法，完美复活全精度对象！
        edm_config = EDMConfig.from_dict(saved_config_dict)
        # 2. 初始化并加载AMPSynth
        print(f"   正在加载 AMPSynth 权重: {config['AMPSynth_weight']} ...")
        score_model = ScoreNetwork(
            edm_config=edm_config,
            num_classes=config['num_classes'],
            d_model=config['d_model']
        ).to(config['device'])
        try:
            score_model.load_state_dict(torch.load(path_edm, map_location=config['device']))
            print("    AMPSynth 加载成功")
        except RuntimeError as e:
            print(f"    AMPSynth权重加载失败! 可能是 num_classes 不匹配。")
            print(f"      错误信息: {e}")
            print("      检查 config['num_classes'] 是否与预训练时一致。")
            return
        # 3. 初始化并加载 Decoder
        print(f"   正在加载 Decoder 权重: {config['decoder_weight']} ...")
        esm_encoder = ESMEncoderWrapper(device=config['device'])  # 依赖项
        esm_decoder = ESMLanguageHeadFinetuner(
            esm_encoder=esm_encoder,
            edm_output_dim=config['d_model'],
            max_seq_len=config['max_seq_len'] + 2,  # 通常是 52
            device=config['device']
        ).to(config['device'])

        checkpoint = torch.load(path_dec, map_location=config['device'], weights_only=False)
        # 2. 剥去外套，只提取纯净权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(" 检测到 Checkpoint 格式，正在提取纯净权重...")
            esm_decoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            esm_decoder.load_state_dict(checkpoint)


        print("    Decoder 加载成功")
        # ==================== 阶段 2: 执行生成 ====================
        print(f"\n>>> 阶段 2: 开始生成 (数量: {config['num_sequences']})")
        output_path = os.path.join(config['output_dir'], config['output_name'])
        generated_sequences = generate_antimicrobial_peptides(
            score_model=score_model,
            esm_decoder=esm_decoder,
            num_sequences=config['num_sequences'],
            num_steps=config['num_steps'],
            max_seq_len=config['max_seq_len'],
            latent_seq_len=config['latent_seq_len'],
            batch_size=config['batch_size'],
            temperature=config['temperature'],
            device=config['device'],
            output_excel_path=output_path,  # 直接保存
            edm_config=edm_config,
            # 无条件生成参数
            target_bacteria_id=None,
            guidance_scale=1.0,
            null_token_id=0
        )
        # ==================== 阶段 3: 统计与展示 ====================
        valid_seqs = [s for s in generated_sequences if 5 <= len(s) <= config['max_seq_len']]
        print("\n" + "=" * 60)
        print(f"  推理任务完成")
        print("=" * 60)
        print(f"  结果已保存: {output_path}")
        print(f"  统计: 总计 {len(generated_sequences)} | 有效 {len(valid_seqs)}")
        if valid_seqs:
            print(f"  平均长度: {sum(len(s) for s in valid_seqs) / len(valid_seqs):.1f}")
            print("  示例序列:")
            for i, seq in enumerate(valid_seqs[:3]):
                print(f"   {i + 1}. {seq}")
        return {'status': 'success'}
    except Exception as e:
        print(f"\n 推理过程出错: {e}")
        traceback.print_exc()
        return {'status': 'error'}


# if __name__ == "__main__":
#     main_inference_unconditional()


### ==================== 主程序 D: 纯推理 - 导入微调模型进行多目标条件生成 ====================

def main_inference_conditional_multi_target():
    """
    脚本 D: 纯推理 (Inference Only) - 多目标条件生成
    功能:
    1. 不进行训练。
    2. 加载 [微调后的 EDM] 和 [预训练 Decoder]。
    3. 针对多个目标细菌，使用指定的 CFG 强度批量生成序列。
    4. 结果保存为多 Sheet Excel。
    """

    # ==================== 1. 推理配置 ====================
    config = {
        # --- 权重路径 (指向微调产出) ---
        'weights_dir': "E:/Users/Mordred/Desktop/Finetune_Output",
        'finetuned_AMPSynth': "finetuned_AMPSynth.pth",

        # --- Decoder 路径 (通常还在预训练文件夹里) ---
        'decoder_dir': "E:/Users/Mordred/Desktop/Pretrain_Output",
        'pretrained_decoder': "pretrained_decoder.pth",

        # --- 输出配置 ---
        'output_dir': "E:/Users/Mordred/Desktop/Inference_Results",

        # --- 模型参数 (必须与微调时完全一致!) ---
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'd_model': 1280,

        # ★★★ 极重要: 必须等于微调时的类别数 (细菌数 + 1个Null) ★★★
        # 如果微调时有4种细菌，这里填5; 如果有3种，这里填4
        'num_classes': 5,
        'null_token_id': 4,  # 通常等于 num_classes - 1

        # --- 生成参数 ---
        'num_steps': 160,
        'max_seq_len': 50,
        'latent_seq_len': 52,
        'temperature': 1.0,

        # --- 目标任务列表 (在此处定义你的设计需求) ---
        'targets': [
            # { 细菌名, ID, 引导强度, 生成数量 }
            # {'name': 'E.coli', 'id': 0, 'scale': 0, 'num': 2000},
            # {'name': 'E.coli', 'id': 0, 'scale': 1, 'num': 2000},
            # {'name': 'E.coli', 'id': 0, 'scale': 2, 'num': 2000},
            # {'name': 'E.coli', 'id': 0, 'scale': 3, 'num': 2000},
            # {'name': 'E.coli', 'id': 0, 'scale': 4, 'num': 2000},
            # {'name': 'E.coli', 'id': 0, 'scale': 5, 'num': 2000},
            # {'name': 'S.aureus', 'id': 1, 'scale': 0, 'num': 2000},
            # {'name': 'S.aureus', 'id': 1, 'scale': 1, 'num': 2000},
            # {'name': 'S.aureus', 'id': 1, 'scale': 2, 'num': 2000},
            # {'name': 'S.aureus', 'id': 1, 'scale': 3, 'num': 2000},
            # {'name': 'S.aureus', 'id': 1, 'scale': 4, 'num': 2000},
            # {'name': 'S.aureus', 'id': 1, 'scale': 5, 'num': 2000},
            {'name': 'P.aeruginosa', 'id': 2, 'scale': 0, 'num': 2000},
            {'name': 'P.aeruginosa', 'id': 2, 'scale': 1, 'num': 2000},
            {'name': 'P.aeruginosa', 'id': 2, 'scale': 2, 'num': 2000},
            {'name': 'P.aeruginosa', 'id': 2, 'scale': 3, 'num': 2000},
            {'name': 'P.aeruginosa', 'id': 2, 'scale': 4, 'num': 2000},
            {'name': 'P.aeruginosa', 'id': 2, 'scale': 5, 'num': 2000},
            {'name': 'B.subtilis', 'id': 3, 'scale': 0, 'num': 2000},
            {'name': 'B.subtilis', 'id': 3, 'scale': 1, 'num': 2000},
            {'name': 'B.subtilis', 'id': 3, 'scale': 2, 'num': 2000},
            {'name': 'B.subtilis', 'id': 3, 'scale': 3, 'num': 2000},
            {'name': 'B.subtilis', 'id': 3, 'scale': 4, 'num': 2000},
            {'name': 'B.subtilis', 'id': 3, 'scale': 5, 'num': 2000},

            # 你甚至可以为同一种细菌尝试不同的力度
            # {'name': 'S.aureus_Strong', 'id': 1, 'scale': 7.0, 'num': 50},
        ]
    }
    # 构造完整路径
    path_AMPSynth = os.path.join(config['weights_dir'], config['finetuned_AMPSynth'])
    path_dec = os.path.join(config['decoder_dir'], config['pretrained_decoder'])
    print("=" * 60)
    print("  启动脚本 D: 导入微调模型 -> 多目标条件生成")
    print("=" * 60)
    os.makedirs(config['output_dir'], exist_ok=True)
    try:
        # ==================== 阶段 1: 加载模型 ====================
        print("\n>>> 阶段 1: 加载模型权重")
        # 1. 加载微调后的 AMPSynth
        print(f"   正在加载 AMPSynth: {config['finetuned_AMPSynth']} ...")

        config_path = os.path.join(config['weights_dir'], "edm_config.json")
        with open(config_path, 'r') as f:
            # 提取保存时的字典
            saved_config_dict = json.load(f)['edm_config']
        # 召唤 EDMConfig 的 from_dict 方法，完美复活全精度对象！
        edm_config = EDMConfig.from_dict(saved_config_dict)

        score_model = ScoreNetwork(
            edm_config=edm_config,
            num_classes=config['num_classes'],  # 关键参数
            d_model=config['d_model']
        ).to(config['device'])
        try:
            score_model.load_state_dict(torch.load(path_edm, map_location=config['device']))
            print("    AMPSynth 加载成功")
        except Exception as e:
            print(f"   ❌ AMPSynth 加载失败! 请检查 config['num_classes'] 是否正确。")
            print(f"      错误详情: {e}")
            return
        # 2. 加载 Decoder
        print(f"   正在加载 Decoder: {config['pretrained_decoder']} ...")
        esm_encoder = ESMEncoderWrapper(device=config['device'])
        esm_decoder = ESMLanguageHeadFinetuner(
            esm_encoder=esm_encoder,
            edm_output_dim=config['d_model'],
            max_seq_len=config['max_seq_len'] + 2,
            device=config['device']
        ).to(config['device'])

        checkpoint = torch.load(path_dec, map_location=config['device'], weights_only=False)
        # 2. 提取权重
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("检测到 Checkpoint 格式，正在提取权重...")
            esm_decoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            esm_decoder.load_state_dict(checkpoint)

        print("    Decoder 加载成功")
        # ==================== 阶段 2: 循环生成 ====================
        print(f"\n>>> 阶段 2: 开始执行 {len(config['targets'])} 个生成任务")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        final_filename = f"Conditional_Inference_{timestamp}.xlsx"
        final_path = os.path.join(config['output_dir'], final_filename)
        # 使用 ExcelWriter 实现多 Sheet
        with pd.ExcelWriter(final_path, engine='openpyxl') as writer:
            for task in config['targets']:
                bac_name = task['name']
                bac_id = task['id']
                scale = task['scale']
                num_seq = task['num']
                print(f"\n--- 任务: {bac_name} (ID={bac_id}) ---")
                print(f"    数量: {num_seq} | CFG Scale: {scale}")
                # 调用生成函数
                seqs = generate_antimicrobial_peptides(
                    score_model=score_model,
                    esm_decoder=esm_decoder,
                    num_sequences=num_seq,
                    num_steps=config['num_steps'],
                    max_seq_len=config['max_seq_len'],
                    latent_seq_len=config['latent_seq_len'],
                    batch_size=64,
                    temperature=config['temperature'],
                    device=config['device'],
                    output_excel_path=None,  # 不保存单文件
                    edm_config=edm_config,
                    # 条件参数
                    target_bacteria_id=bac_id,
                    guidance_scale=scale,
                    null_token_id=config['null_token_id']
                )
                # 统计有效性
                valid_seqs = [s for s in seqs if 5 <= len(s) <= config['max_seq_len']]
                valid_rate = len(valid_seqs) / len(seqs) * 100
                print(f"     统计: 有效 {len(valid_seqs)}/{len(seqs)} ({valid_rate:.1f}%)")
                # 写入 Sheet
                df = pd.DataFrame({
                    'Sequence': seqs,
                    'Length': [len(s) for s in seqs],
                    'Is_Valid': [5 <= len(s) <= config['max_seq_len'] for s in seqs],
                    'Target_Name': bac_name,
                    'Target_ID': bac_id,
                    'CFG_Scale': scale
                })
                sheet_name = f"{bac_name}_CFG_{scale}"[:31]
                invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
                for ch in invalid_chars:
                    sheet_name = sheet_name.replace(ch, '_')
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"     已写入 Sheet: {sheet_name}")
        print("\n" + "=" * 60)
        print(f"  所有推理任务完成")
        print("=" * 60)
        print(f"  结果文件: {final_path}")
        return {'status': 'success'}
    except Exception as e:
        print(f"\n 推理中断: {e}")
        traceback.print_exc()
        return {'status': 'error'}


# if __name__ == "__main__":
#     main_inference_conditional_multi_target()


#  核心开关：在这里切换任务模式！
RUN_MODE = "inference_unconditional"
# ==========================================5
if __name__ == "__main__":
    print("=" * 60)
    print(f" ⚙ 当前执行模式: {RUN_MODE}")
    print("=" * 60)
    if RUN_MODE == "pretrain_uncond":
        main_pretrain_uncond()
    elif RUN_MODE == "finetune_condition":
        main_finetune_condition()
    elif RUN_MODE == "inference_unconditional":
        main_inference_unconditional()
    elif RUN_MODE == "inference_conditional":
        main_inference_conditional_multi_target()
    else:
        print("输入错误，请重新输入合理模式")




