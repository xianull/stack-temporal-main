"""Attention building blocks for the StateICL models."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 这边input的x是src/stack/models/core/base.py中的第74行

class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation used across the project."""

    # 初始化函数，定义模型参数
    # d_model: input x的维度，这里是(batch_size * n_cells, n_hidden, token_dim) --> (64 * 128, 100, 16) --> (8192,100,16)
    # n_heads: 注意力头数
    # head_dim: 每个注意力头的维度
    # scale: 缩放因子，用于避免梯度爆炸
    # qkv: 线性变换，将输入x映射到qkv。这是一个三合一的投影层，将输入x映射到query、key、value三个向量。
        # Query (问题)、Key (答案标签)、Value (具体内容)
    # proj: 线性变换，将输出映射到d_model。将多头注意力拼接后的结果映射回原始维度d_model。
    # dropout:  dropout比例
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model # $d\_model = n\_heads \times head\_dim$
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply multi-head self-attention."""

        batch_size, seq_len, _ = x.shape 
        # cell-wise self-attention的输入：x_cell = (B, n_cells, n_hidden, token_dim) --> (B*n_cells, n_hidden,token_dim) --> (64 * 128, 100, 16) --> (8192, 100, 16) (样本数为8192，也就是8192个细胞，每个细胞内部的100个token都要进行一次基因间的attention计算)
        # gene-wise self-attention的输入：x_gene = (B, n_cells, n_genes * token_dim) --> (B,n_cells, n_genes * token_dim) --> (64, 128, 100 * 16) --> (64, 128, 1600) （样本数为64，这里的64是batch_size，也就是64个样本，每个样本内部的128个细胞都要进行一次细胞间的attention计算）
        # qkx(x_cell) : 
            # self.qkv = nn.Linear(d_model, d_model * 3, bias=False) pytorch中的linear层只计算最后一个维度
            # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model * 3) --> (8192, 100, 16 * 3) --> (8192, 100, 48) 
            # 然后reshape成(batch_size, seq_len, 3, self.n_heads, self.head_dim) --> (8192,100,3,8,16) --> (8192,100,3,8,2) [样本, 序列位置, 身份(Q/K/V), 哪个头, 头维度]

        # qkv(x_gene) : 
            # (64, 128, 1600) --> (64, 128, 1600 * 3) --> (64, 128, 4800) --> (64, 128, 3, 1600) --> (64, 128, 3, 8, 200)
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim) 

        # 将qkv的维度从(batch_size, seq_len, 3, self.n_heads, self.head_dim) -> (3, batch_size, self.n_heads, seq_len, self.head_dim)
        # 一次性算出来q,k,c；multi-head就是这里的
        q, k, v = qkv.permute(2, 0, 3, 1, 4) 
        # 变换后：(3, batch_size, n_heads, seq_len, head_dim) --> (3, 8192, 8, 100, 2)
        # x_gene的q,k,v: (3, 64, 8, 128, 200)
        # 直接解包就可以得到q,k,v
        # self.qkv.weight：$$(3 \times d\_model, d\_model)$$
        
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        # 计算注意力分数：$$\text{Score} = Q \times K^T$$
        # @：矩阵乘法,只对tensor的最后两个维度进行运算
        # k.transpose(-2, -1) : 交换k的最后两个维度，(8192, 8, 100, 2) --> (8192, 8, 2, 100)，相当于 $$K^T$
        # scores:(8192, 8, 100, 100)
        # self.scale = self.head_dim**-0.5 $\frac{1}{\sqrt{head\_dim}}$
        # 在一个细胞中，(_, _, 100, 100)的矩阵表示100个基因之间的两两相关性

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            mask = attn_mask
            # apply_mask
            while mask.ndim < 4:
                # attn_scores = (Batch, Heads, Seq, Seq)，四维
                # 维度补齐，(1, 1, Seq, Seq)
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1:
                mask = mask.expand(batch_size, -1, -1, -1)
            if mask.shape[1] == 1:
                mask = mask.expand(-1, self.n_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            # masked_fill: 将mask为True的位置填充为float("-inf")
            # 这样在softmax时，这些位置的权重会趋近于0，从而实现masked的效果
            # $$\text{Attention Weight} = \text{Softmax}(\text{Score})$$
            
            # $e^{-\infty} = 0$

        attn = self.dropout(F.softmax(attn_scores, dim=-1))
        # F.softmax(attn_scores, dim=-1)：对最后一维进行softmax操作，得到注意力权重
        # 在每一行（即每一个 Query 对应的所有 Key）上做 Softmax。这确保了每个基因（或细胞）看出去的总权重之和等于 1
        # 以x_cell举例，现在这里的attn为(batch_size,n_head,seq_len,seq_len) --> (8192, 8, 100, 100)
        out = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        # (attn @ v)：将注意力权重与value相乘，得到加权后的value (batch_size,n_head,L,L) @ (batch_size, n_heads, seq_len, head_dim)，此时信息还是在8个头中分开的
        # (8192, 8, 100, 100) @ (8192, 8, 100, 2) --> (8192, 8, 100, 2)，现在，每个基因的特征向量（维度为 2）不再是它自己原始的样子了，而是根据权重混合了其他基因特征后的新向量
        # transpose(1, 2): (batch_size, n_heads, seq_len, head_dim) --> (batch_size, seq_len, n_heads, head_dim) --> (8192, 100, 8, 2),让同一个基因在 8 个头里的特征挨在一起
        # reshape(batch_size, seq_len, self.d_model)：将注意力权重和value的维度 reshape 成(batch_size, seq_len, self.d_model)
        # 这里注意，x_cell的d_model是16，x_gene的d_model是1600
        
        out = self.proj(out)
        # $W_O$矩阵，投影层融合信息
        if return_attn:
            return out, attn
        return out, None
        # 如果要查看注意力图，就返回结果和权重矩阵
        # 否则只返回计算后的特征，节省内存

class TabularAttentionLayer(nn.Module):
    """Single layer of tabular attention for gene expression modelling."""

    def __init__(
        self,
        token_dim: int,
        n_cells: int,
        n_hidden: int,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        cell_attn_dim = token_dim # 16，这里就是把n_genes reduce到1600维，然后拆成100*16
        cell_attn_heads = 8
        # The model dimension must be divisible by the number of attention heads.
        # If not, find the largest valid number of heads smaller than or equal to n_heads.
        if cell_attn_dim % cell_attn_heads != 0:
            for h in range(min(n_heads, cell_attn_dim), 0, -1):
                if cell_attn_dim % h == 0:
                    cell_attn_heads = h
                    break
            else:  # This fallback should ideally not be reached if cell_attn_dim >= 1
                cell_attn_heads = 1

        # 实例化第一个注意力层，用于计算细胞内部基因与基因的互作
        self.cell_attn = MultiHeadAttention(cell_attn_dim, cell_attn_heads, dropout)
        # 这里LayerNorm的作用是把神经元的输出拉回到一个均值为 0、方差为 1 的标准分布，避免梯度消失或爆炸
        self.cell_norm = nn.LayerNorm(cell_attn_dim)

        # 实例化第二个注意力层，用于计算细胞间的互作
        # 这里就需要表示一个细胞，所以把n_hidden * token_dim --> 100 * 16 --> 1600维
        gene_attn_dim = n_hidden * token_dim
        self.gene_attn = MultiHeadAttention(gene_attn_dim, n_heads, dropout)
        # gene_attn = (batch_size*n_cells, n_genes, token_dim) --> (8192, 100, 16)
        self.gene_norm = nn.LayerNorm(gene_attn_dim)

        # 一般来说，FNN映射的维度是原维度的4倍 (Attention is all you need)
        hidden_dim = token_dim * mlp_ratio
        
        # FNN，out_dim --> out_dim * mlp_ratio --> out_dim，特征压缩和提炼
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        x: torch.Tensor,
        gene_pos_emb: torch.Tensor,
        gene_attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, n_cells, n_genes, token_dim = x.shape # batch_size: 64, n_cells: 128, n_genes: 100, token_dim: 16（注意，这里的n_genes是base中的n_hidden）
        # 将x的维度从(batch_size, n_cells, n_genes, token_dim) --> (batch_size * n_cells, n_genes, token_dim) --> (8192, 100, 16)
        x_cell = x.reshape(batch_size * n_cells, n_genes, token_dim)

        # 这里可以看src/stack/models/core/base.py，这里的gene_pos_emb是(n_hidden, token_dim) --> (100, 16)
        x_cell_with_pos = x_cell + gene_pos_emb.unsqueeze(0)

        # 在我们计算的x的基础上，广播位置编码信息
        cell_attn_out, _ = self.cell_attn(x_cell_with_pos)
       
        # 残差连接
        x_cell = self.cell_norm(x_cell + cell_attn_out)
        
        x = x_cell.reshape(batch_size, n_cells, n_genes, token_dim)

        # 细胞间注意力的输入：(B, n_cells, n_genes, token_dim) --> (B, n_cells, n_genes * token_dim) --> (64, 128, 100 * 16) --> (64, 128, 1600)
        x_gene = x.reshape(batch_size, n_cells, n_genes * token_dim)

        if return_attn:
            gene_attn_out, attn = self.gene_attn(x_gene, attn_mask=gene_attn_mask, return_attn=True)
        else:
            gene_attn_out, attn = self.gene_attn(x_gene, attn_mask=gene_attn_mask)
        
        # 残差连接
        x_gene = self.gene_norm(x_gene + gene_attn_out)
        

        x = x_gene.reshape(batch_size, n_cells, n_genes, token_dim)

        # 特征压缩和提炼，(64, 128, 100, 16) --> (64 * 128 * 100, 16) --> (8192 * 100, 16) --> (819200, 16)
        mlp_input = x.reshape(-1, token_dim)
        
        # 对每个基因的 16 维特征进行非线性映射（先扩充到 64 维再缩回 16 维）
        mlp_out = self.mlp(mlp_input)

        # 残差连接
        x = self.mlp_norm(mlp_input +  mlp_out).reshape(batch_size, n_cells, n_genes, token_dim)

        # 这里的 x 是经过一个layers“每个 cell 的 token 表征”（已经融合了 cell 内 token 关系 + cell 间关系 + MLP 非线性变换）
        return x, attn # x: (batch_size, n_cells, n_genes, token_dim)，attn: (batch_size, n_head, seq_len, seq_len)
