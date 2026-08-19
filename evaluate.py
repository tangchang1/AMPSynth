import os
import pandas as pd
import numpy as np
import Levenshtein
import re
import random
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import esm
from collections import Counter
from scipy.spatial.distance import jensenshannon
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from tqdm import tqdm
from modlamp.descriptors import GlobalDescriptor
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm


# 1. 严格的数据清洗函数 (Quality Control)
def clean_sequences(seq_list, min_len=5, max_len=50):
    valid_aa_pattern = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
    cleaned_seqs = []
    for seq in seq_list:
        seq = str(seq).strip().upper()
        if min_len <= len(seq) <= max_len:
            if valid_aa_pattern.match(seq):
                cleaned_seqs.append(seq)
    return cleaned_seqs


# 2. 精简版 Excel 读取函数
def load_seqs_from_excel(file_path, column_indicator=0,sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if isinstance(column_indicator, int):
        seqs = df.iloc[:, column_indicator].dropna().astype(str).tolist()
    else:
        seqs = df[column_indicator].dropna().astype(str).tolist()
    return seqs

# def load_seqs_from_excel(file_path, column_indicator=0):
#     df = pd.read_excel(file_path)
#     if isinstance(column_indicator, int):
#         seqs = df.iloc[:, column_indicator].dropna().astype(str).tolist()
#     else:
#         seqs = df[column_indicator].dropna().astype(str).tolist()
#     return seqs0



# 3. 核心指标计算函数
# --- 替换：基于 BLOSUM62 的生物学内部多样性 ---
def calculate_esm_intdiv(sequences, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    基于 ESM2 模型计算多肽序列的内部多样性 (Internal Diversity)

    参数:
        sequences (list of str): 生成的多肽序列列表
        batch_size (int): 提取特征时的批处理大小，防显存溢出
        device (str): 运行设备 (cuda 或 cpu)

    返回:
        float: ESM IntDiv 得分 (范围 0 到 1，越大越好)
    """
    if not sequences:
        return 0.0
    print(f"Loading ESM-2 model on {device}...")
    # 加载 ESM2 轻量级模型 (8M 参数，速度极快，维度 320)
    # 如果想追求极致精度，可以换成 'esm2_t33_650M_UR50D' (维度 1280)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    all_embeddings = []
    # 1. 分批提取序列的 ESM2 Embedding (防止把显存撑爆)
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="提取 ESM2 语义特征"):
            batch_seqs = sequences[i: i + batch_size]
            # ESM2 API 要求的输入格式: [(id, seq), ...]
            data = [(str(idx), seq) for idx, seq in enumerate(batch_seqs)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            # 提取特征字典 (第 6 层)
            results = model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]
            # 2. Mean Pooling: 提取真正的序列句向量 (Sequence Embedding)
            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                # 切片去除开头的 <cls> (索引0) 和结尾的 <eos>/<pad>
                # 只保留真实氨基酸的特征进行平均
                seq_emb = token_representations[j, 1: seq_len + 1].mean(dim=0)
                all_embeddings.append(seq_emb.cpu())  # 存回 CPU，防 GPU 溢出
    # 堆叠成一个大的特征矩阵 (N, D)，例如 (10000, 320)
    embeddings = torch.stack(all_embeddings)
    N = embeddings.shape[0]
    if N <= 1:
        return 0.0
    print("计算全局 Cosine 相似度与 IntDiv 分数...")
    # 3. 矩阵化光速计算两两余弦相似度 (Cosine Similarity)
    # 先对所有向量进行 L2 归一化
    embeddings_normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # 归一化后，矩阵乘法的结果即为 Cosine 相似度矩阵 (N x N)
    sim_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
    # 4. 计算 IntDiv 公式
    # 矩阵所有元素之和，减去对角线元素之和 (即 N，因为自己跟自己相似度为 1)
    sum_off_diagonal = sim_matrix.sum().item() - N
    # 计算非对角线元素的平均相似度
    mean_sim = sum_off_diagonal / (N * (N - 1))
    # IntDiv = 1 - 平均相似度
    intdiv = 1.0 - mean_sim
    return intdiv


def calculate_average_plddt(sequences, sample_size=500, device="cuda" if torch.cuda.is_available() else "cpu"):
    if not sequences:
        return 0.0

    print("只保留纯洁的 20 种氨基酸...")
    valid_sequences = []
    for seq in sequences:
        seq = str(seq).strip().upper()
        # 严苛过滤：只允许 20 种标准氨基酸。遇到带有 X, U, B 等未知字母的直接抛弃！
        if re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq) and 3 <= len(seq) <= 150:
            valid_sequences.append(seq)

    if len(valid_sequences) == 0:
        print("❌ 数据全军覆没！请检查输入数据！")
        return 0.0

    if len(valid_sequences) > sample_size:
        print(f"为了加速评估，从 {len(valid_sequences)} 条中随机抽样 {sample_size} 条...")
        eval_sequences = random.sample(valid_sequences, sample_size)
    else:
        eval_sequences = valid_sequences

    print(f"Loading ESMFold model on {device} in FP16 (Half Precision)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    ).eval().to(device)

    total_plddt = 0.0
    valid_seq_count = 0

    print(" 开始结构预测...")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # 取消了 batch_size，直接一条一条遍历
        for seq in tqdm(eval_sequences, desc="ESMFold 结构预测"):

            # 单条输入，不需要 padding！完美避开 CUDA 越界 Bug！
            inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            try:
                outputs = model(**inputs)
                # 输出形状直接是 [1, seq_len]，直接取平均即可
                seq_plddt = outputs.plddt[0].mean().item()

                # 统一换算为百分制
                if seq_plddt <= 1.0:
                    seq_plddt *= 100.0

                total_plddt += seq_plddt
                valid_seq_count += 1
            except Exception as e:
                print(f"\n⚠ {seq} 预测失败，已跳过。报错: {e}")
                continue

    macro_avg_plddt = total_plddt / valid_seq_count if valid_seq_count > 0 else 0.0

    del model
    torch.cuda.empty_cache()

    return macro_avg_plddt







# --- 新增：BLOSUM62 比对器初始化 ---
def get_blosum_aligner():
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.mode = 'global'
    return aligner
def compute_bio_similarity(seq1, seq2, aligner):
    score_ab = aligner.score(seq1, seq2)
    score_aa = aligner.score(seq1, seq1)
    score_bb = aligner.score(seq2, seq2)
    denominator = max(score_aa, score_bb)
    if denominator <= 0: return 0.0
    return max(0.0, score_ab / denominator)
# --- 替换：统合计算 Similarity 和 Novelty ---
def calculate_blosum_similarity_and_novelty(generated_seqs, train_seqs, sample_size=1000, novelty_threshold=0.8):
    gen_set_unique = list(set(generated_seqs))
    train_set_unique = list(set(train_seqs))
    if not gen_set_unique or not train_set_unique:
        return 0.0, 0.0
    actual_sample = min(sample_size, len(gen_set_unique))
    # 随机抽样
    gen_sample = np.random.choice(gen_set_unique, actual_sample, replace=False)
    aligner = get_blosum_aligner()
    print(f"   [BLOSUM62 相似度 & 新颖度] 正在为 {len(gen_sample)} 条生成序列寻找训练集中的‘最相似序列’...")
    max_similarities = []
    novel_count = 0
    # 加入 tqdm 进度条
    for gen_seq in tqdm(gen_sample, desc="比对进度"):
        best_sim_for_this_seq = 0.0
        for train_seq in train_set_unique:
            sim = compute_bio_similarity(gen_seq, train_seq, aligner)
            if sim > best_sim_for_this_seq:
                best_sim_for_this_seq = sim
            if best_sim_for_this_seq >= 1.0:
                break  # 已经找到完全一样的，提前结束当前序列的比对
        max_similarities.append(best_sim_for_this_seq)
        # 核心逻辑：最大相似度小于 0.8 才算 Novel
        if best_sim_for_this_seq < novelty_threshold:
            novel_count += 1
    avg_similarity = np.mean(max_similarities) if max_similarities else 0.0
    novelty_ratio = novel_count / len(gen_sample) if len(gen_sample) > 0 else 0.0
    return avg_similarity, novelty_ratio





# 4. 新增：ESM-2 伪困惑度计算类
class ESM2PerplexityCalculator:
    def __init__(self, model_name="esm2_t6_8M_UR50D"):
        print(f"正在加载 ESM-2 模型: {model_name} ...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f" 模型已就绪 ({self.device})")

    def compute_ppl(self, sequence):
        seq_len = len(sequence)
        # 针对每个位置进行 Mask
        batch_data = []
        for i in range(seq_len):
            masked_seq = list(sequence)
            masked_seq[i] = "<mask>"
            batch_data.append((f"pos_{i}", "".join(masked_seq)))

        _, _, batch_tokens = self.batch_converter(batch_data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"]  # (L, L+2, Alphabet_Size)

        log_probs = []
        for i in range(seq_len):
            true_aa_idx = self.alphabet.get_idx(sequence[i])
            # 这里的 i+1 是因为 ESM2 自动添加了 <CLS> 标记
            token_logits = logits[i, i + 1, :]
            prob = torch.nn.functional.softmax(token_logits, dim=-1)
            log_probs.append(torch.log(prob[true_aa_idx]).item())

        return np.exp(-np.mean(log_probs))

    def evaluate(self, seq_list, sample_size=100):
        unique_seqs = list(set(seq_list))
        actual_sample = min(sample_size, len(unique_seqs))
        random.seed(42)
        sample_seqs = random.sample(unique_seqs, actual_sample)

        ppl_values = []
        for i, seq in enumerate(sample_seqs):
            if (i + 1) % 100 == 0: print(f"   -> PPL 进度: {i + 1}/{actual_sample}")
            ppl_values.append(self.compute_ppl(seq))
        return np.mean(ppl_values)









# =================  主执行程序 =================
if __name__ == "__main__":
    train_excel_path = "E:/Users/Mordred/Desktop/AMP.xlsx"
    gen_excel_path = r"E:\Users\Mordred\Desktop\generated_AMPs_uncond-pfema.xlsx"
    gen_sheet_name = "Sheet1"

    try:
        print("1. 正在读取并清洗数据...")
        raw_train_seqs = load_seqs_from_excel(train_excel_path, column_indicator=1)
        #raw_gen_seqs = load_seqs_from_excel(gen_excel_path, column_indicator=0)
        raw_gen_seqs = load_seqs_from_excel(gen_excel_path, column_indicator=0, sheet_name=gen_sheet_name)
        clean_train = clean_sequences(raw_train_seqs)
        clean_gen = clean_sequences(raw_gen_seqs)
        #clean_train=raw_train_seqs
        #clean_gen=raw_gen_seqs
        # print(f"-> 训练集: 保留 {len(clean_train)} 条")
        # print(f"-> 生成集: 保留 {len(clean_gen)} 条\n")
        #
        print("2. 正在计算序列统计指标...")
        # 统合计算 BLOSUM62 相似度和新颖性 (抽样 1000 条足以保证统计学精度，且防卡死)
        avg_sim, _ = calculate_blosum_similarity_and_novelty(
            clean_gen, clean_train, sample_size=2000)

        #
        print(f" BLOSUM62 Similarity (相似度): {avg_sim:.2%}")
        #
        diversity = calculate_esm_intdiv(clean_gen)
        print(f"\n最终 ESM-IntDiv 得分: {diversity:.4f}")
        #
        pLDDT = calculate_average_plddt(clean_gen,sample_size=2000)
        print(f"\n 最终平均 pLDDT 得分: {pLDDT:.2f} / 100")
        # # 3. 计算 ESM-2 困惑度
        print("\n3. 正在计算 ESM-2 伪困惑度 (PPL)...")
        ppl_calc = ESM2PerplexityCalculator()
        # #计算生成集 PPL
        gen_ppl = ppl_calc.evaluate(clean_gen, sample_size=2000)
        print(f" 生成集 ESM-2 PPL: {gen_ppl:.4f} (越低越接近天然序列)")
        # # (可选) 计算训练集 PPL 作为基准对比
        #train_ppl = ppl_calc.evaluate(clean_train, sample_size=1000)
        # print(f" 训练集 PPL 基准: {train_ppl:.4f}")






    except Exception as e:
        print(f"运行出错: {e}")