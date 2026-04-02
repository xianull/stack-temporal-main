
# stack-temporal-main-0402

scTFM的基座模型
- [x] 模块拆解+quick start
- [ ] 对比ZINB和NB分布
- [ ] 建模连续的生物流形隐空间
- [ ] 实现stack-bio-aware + flow matching
- [ ] 封装模块和测试脚本
## 模型架构与数据流

### 总体数据流

```
原始计数矩阵 (B, n_cells, n_genes)
       │
       ▼ log1p + 矩形掩码 (mask_rate 0.2~0.8)
掩码后特征 (B, n_cells, n_genes)
       │
       ▼ 可训练 Tokenization (Linear → GELU → Dropout)
Token 嵌入 (B, n_cells, n_hidden, token_dim)
       │
       ▼ × n_layers 个 TabularAttentionLayer
  ┌─────────────────────────────────────┐
  │  ① 细胞内注意力 (Cell-wise Attn)     │  reshape → (B×n_cells, n_hidden, token_dim)
  │     每个细胞内 n_hidden 个 token     │  基因间信息交互 + 位置编码
  │              ↓                      │
  │  ② 细胞间注意力 (Gene-wise Attn)     │  flatten → (B, n_cells, n_hidden×token_dim)
  │     n_cells 个细胞的完整表征互相关注  │  细胞间信息交互
  │              ↓                      │
  │  ③ 前馈网络 (MLP)                    │  token_dim → 4×token_dim → token_dim
  │     非线性特征提炼                   │
  └─────────────────────────────────────┘
       │
       ▼ reshape → (B, n_cells, n_hidden × token_dim)
最终细胞嵌入 final_cell_embeddings
       │
       ▼ 输出 MLP (Linear → GELU → Linear)
NB 分布参数 (nb_mean, nb_dispersion, px_scale)
```

### 最终 x 的特点：双重注意力机制

模型的核心创新在于 **`TabularAttentionLayer`**（`src/stack/modules/attention.py`），每一层交替执行两种注意力：

| 阶段 | 输入形状 | 注意力方向 | 语义含义 |
|------|---------|-----------|---------|
| **细胞内注意力** `(cell_attn)` | `(B×n_cells, n_hidden, token_dim)` | n_hidden 个 token 之间 | 同一细胞内，基因模块（gene tokens）之间的共表达关系 |
| **细胞间注意力** `(gene_attn)` | `(B, n_cells, n_hidden×token_dim)` | n_cells 个细胞之间 | 不同细胞的完整表达谱之间的互作关系 |

- **细胞内注意力**：把每个细胞视为一个"句子"，n_hidden 个 token 是"词"，自注意力捕捉基因间的共调控模式
- **细胞间注意力**：把每个细胞的完整表征（flatten 后的 n_hidden×token_dim 维向量）作为输入，自注意力捕捉细胞间的上下文关系

每个阶段后都接 LayerNorm + 残差连接，最后经过 MLP 非线性变换。

### 训练目标

```
L_pre = L_recon + λ × L_sw
```

- **L_recon**：负二项分布（NB）负对数似然重构损失，仅在被掩码基因上计算
- **L_sw**：Sliced Wasserstein 距离，将潜在空间正则化为以批次为中心的多变量高斯分布
- **λ**（`sw_weight`）：默认 0.01

### 推理输出

输出 MLP 为每个细胞中的每个基因预测两个参数：
- **px_scale**（softmax 后）：基因表达比例，同一细胞内所有基因求和为 1
- **nb_dispersion**（softplus 后）：NB 分布的离散度 θ
- **nb_mean** = px_scale × observed_lib_size：预测的原始计数期望值

## 安装

```bash
# pip 安装
pip install arc-stack

# 或从源码安装（开发模式）
git clone https://github.com/ArcInstitute/stack.git
cd stack
pip install -e .

# uv 安装
uv pip install arc-stack
```

## Quick Start

### 1. 提取细胞嵌入

```bash
stack-embedding \
    --checkpoint "./checkpoints/pretrained.ckpt" \
    --adata "data.h5ad" \
    --genelist "hvg_genes.pkl" \
    --output "embeddings.h5ad" \
    --batch-size 32
```

也可使用交互式 Notebook：[tutorial-embed.ipynb](notebooks/tutorial-embed.ipynb)

### 2. 零样本预测（上下文生成）

```bash
stack-generation \
    --checkpoint "./checkpoints/pretrained.ckpt" \
    --base-adata "base_data.h5ad" \
    --test-adata "test_data.h5ad" \
    --genelist "hvg_genes.pkl" \
    --output-dir "./generations" \
    --split-column "donor_id"
```

也可使用交互式 Notebook：[tutorial-predict.ipynb](notebooks/tutorial-predict.ipynb)

### 3. 从零训练

```bash
stack-train \
    --dataset_configs "/path/to/data:false:gene_symbols" \
    --genelist_path "hvg_genes.pkl" \
    --save_dir "./checkpoints" \
    --sample_size 256 \
    --batch_size 32 \
    --n_hidden 100 \
    --token_dim 16 \
    --n_layers 9 \
    --max_epochs 10
```

### 4. 冻结教师微调

```bash
stack-finetune \
    --checkpoint_path "./checkpoints/pretrained.ckpt" \
    --dataset_configs "human:/path/to/data:donor_id:cell_type:false" \
    --genelist_path "hvg_genes.pkl" \
    --save_dir "./finetuned_checkpoints" \
    --sample_size 512 \
    --batch_size 8 \
    --replacement_ratio 0.75 \
    --max_epochs 8
```

### 5. 使用配置文件

```bash
# 使用预设配置训练
stack-train --config configs/training/bc_large.yaml

# 命令行参数覆盖配置文件中的值
stack-train --config configs/training/bc_large.yaml --learning_rate 5e-5

# 微调
stack-finetune --config configs/finetuning/ft_parsecg.yaml
```

> YAML 配置需要 `pip install pyyaml`，也可直接使用 JSON 格式。

## 数据准备

### 计算高变基因（HVG）

```python
from stack.data.datasets import DatasetConfig, compute_hvg_union

configs = [DatasetConfig(path="/data/path", filter_organism=True)]
hvg_genes = compute_hvg_union(configs, n_top_genes=1000, output_path="hvg.pkl")
```

### 数据集配置格式

- **人类数据集**：`human:/path:donor_col:cell_type_col[:filter_organism[:gene_col]]`
- **药物数据集**：`drug:/path:condition_col:cell_line_col:control_condition[:filter_organism[:gene_col]]`

## 模块更换指南

Stack 的架构采用模块化设计，各组件可以独立替换。以下是关键模块及其文件位置：

### 项目结构

```
src/stack/
├── modules/                    # 可复用的基础构建块
│   ├── attention.py            # MultiHeadAttention, TabularAttentionLayer
│   └── regularizers.py         # SlicedWassersteinDistance
├── models/
│   ├── core/
│   │   ├── base.py             # StateICLModelBase（主模型定义 + forward）
│   │   ├── losses.py           # LossComputationMixin（重构损失 + SW 正则化）
│   │   └── inference.py        # 推理逻辑
│   └── finetune/
│       ├── model.py            # 微调模型
│       └── mixins.py           # 微调混入类
├── data/
│   ├── training/datasets.py    # 训练数据集 + DataLoader
│   ├── finetuning/datasets.py  # 微调数据集
│   ├── gene_processing.py      # 基因名称处理
│   ├── h5_manager.py           # HDF5 文件管理
│   └── hvg.py                  # 高变基因计算
└── cli/                        # 命令行入口
```

### 替换注意力层

**文件**：`src/stack/modules/attention.py`

`TabularAttentionLayer` 包含两个注意力子模块：

```python
# 细胞内注意力：处理 (B×n_cells, n_hidden, token_dim)
self.cell_attn = MultiHeadAttention(cell_attn_dim, cell_attn_heads, dropout)

# 细胞间注意力：处理 (B, n_cells, n_hidden×token_dim)
self.gene_attn = MultiHeadAttention(gene_attn_dim, n_heads, dropout)
```

**替换方式**：只要新注意力层的输入输出形状与 `MultiHeadAttention` 一致（输入 `(B, S, D)` → 输出 `(B, S, D)`），即可直接替换。例如换用 FlashAttention、线性注意力等。

### 替换 Tokenization 模块

**文件**：`src/stack/models/core/base.py` 的 `gene_reduction`

```python
# 默认：Linear(n_genes, n_hidden × token_dim) → GELU → Dropout
self.gene_reduction = nn.Sequential(...)
```

**替换方式**：任何将 `(B, n_cells, n_genes)` 映射到 `(B, n_cells, n_hidden × token_dim)` 的模块均可替换此处。例如换用 1D CNN、PCA 投影等。

### 替换输出解码器

**文件**：`src/stack/models/core/base.py` 的 `output_mlp`

```python
# 默认：Linear → GELU → Dropout → Linear，输出 n_genes×2 个参数
self.output_mlp = nn.Sequential(...)
```

**替换方式**：任何将 `(B×n_cells, n_hidden×token_dim)` 映射到 `(B×n_cells, n_genes×2)` 的模块。如需换用其他分布（如 ZINB、Poisson），需同时修改 `losses.py` 中的 `_compute_reconstruction_loss`。

### 替换正则化项

**文件**：`src/stack/modules/regularizers.py`

默认使用 Sliced Wasserstein Distance 作为潜在空间正则化。可替换为 VAE 的 KL 散度、MMD 等其他分布对齐方法。需同步修改 `losses.py` 中的 `_compute_sw_loss`。

### 替换损失函数

**文件**：`src/stack/models/core/losses.py`

`LossComputationMixin` 提供三个方法：
- `_compute_reconstruction_loss`：NB 负对数似然
- `_compute_sw_loss`：Sliced Wasserstein 正则化
- `_compute_eval_metrics`：MAE 和相关系数评估指标

可独立替换任意一个，不影响其他部分。
