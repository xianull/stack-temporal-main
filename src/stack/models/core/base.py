"""stack的模型架构"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import SlicedWassersteinDistance, TabularAttentionLayer
# TabularAttentionLayer: 注意力层，参考Transformer架构
# Sliced Wasserstein Distance: 模型生成的经验分布与以批次为中心的多变量高斯先验分布之间的距离

class StateICLModelBase(nn.Module):
    """stack的基类，继承自pytorch的nn.Module，定义模型架构和foward方法"""

    # 初始化函数，定义模型参数，
    # n_genes: 基因数量，n_hidden: 隐藏层维度，token_dim: 每个基因token（论文中的gene module）的维度
    # n_cells: 细胞数量，n_layers: 注意力层数，n_heads: 注意力头数，mlp_ratio: FNN中间层宽度相对token_dim的倍数，参考Transformer架构
    # dropout:  dropout比例
    # mask_rate_min: 最小mask比例，mask_rate_max: 最大mask比例，论文中的Gene_level r^2MAE mask
    # sw_weight: Sliced Wasserstein距离权重 --》 The pre-training objective combines a reconstruction loss with a latent space regularization term. stack预训练的目标函数是重构损失和潜在空间正则化项的组合：$$\mathcal{L}_{pre} = \mathcal{L}_{recon} + \lambda \mathcal{L}_{sw}$$，这里的lambda就是sw_weight，论文取值为0.01
    # 在post-training阶段，sw_weight也被保留
    # Sliced Wasserstein Distance：模型生成的经验分布与以批次为中心的多变量高斯先验分布之间的距离
    # n_proj: 投影维度
    # 假设训练时batchsize为64

    def __init__(
        self,
        n_genes: int,
        n_hidden: int = 100,
        token_dim: int = 8,
        n_cells: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        mask_rate_min: float = 0.2,
        mask_rate_max: float = 0.8,
        sw_weight: float = 1.0,
        n_proj: int = 64,
    ) -> None:
        super().__init__()
    # 初始化父类，定义模型参数
        self.n_genes = n_genes
        self.n_hidden = n_hidden
        self.token_dim = token_dim
        self.n_cells = n_cells
        self.n_layers = n_layers
        self.mask_rate_min = mask_rate_min
        self.mask_rate_max = mask_rate_max
        self.sw_weight = sw_weight
        self.n_proj = n_proj
        self.sw_distance = SlicedWassersteinDistance(n_proj=n_proj)

        # gene_reduction: 基因特征降维，将基因数量从n_genes降维到n_hidden * token_dim，fig1B中的Trainable tokenization
        # 使用GELU激活函数，GELU激活函数的特点是：在0附近是线性的，在远离0的地方是非线性的，类似于ReLU函数，但比ReLU函数更平滑，更易于优化
        #GELU激活函数公式：$$GELU(x) = x * \Phi(x)$$，其中$$\Phi(x) = 0.5 * (1 + erf(x / \sqrt{2}))$$，erf是误差函数，$$erf(x) = 2 / \sqrt{\pi} * \int_0^x e^{-t^2} dt$$
        # 在输入接近 0 的区域，GELU 的梯度更加丰富
        # 这里只是先进行基因维度降维，后续还会进行tokenization
        self.gene_reduction = nn.Sequential(
            nn.Linear(n_genes, n_hidden * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )


        # gene_pos_embedding: 基因位置嵌入，在 cell 内、沿 n_hidden 个slot 做 cell-wise self-attention 之前，给每个槽加上一个可学习的偏置向量，让不同下标 (0,\ldots,n_{\text{hidden}}-1) 的槽在特征空间里可区分。
        # 广播 (B * n_cells, n_hidden, token_dim) + (1, n_hidden, token_dim) -> (B * n_cells, n_hidden, token_dim)
        # 这里的广播是在attention中进行的，x(B, n_cells, n_hidden, token_dim) --> x_cell (B * n_cells, n_hidden, token_dim)
        self.gene_pos_embedding = nn.Parameter(torch.randn(n_hidden, token_dim))

        # 跳转至src/stack/modules/attention.py阅读TabularAttentionLayer的定义
        # 目前为止，x = (B, n_cells, n_hidden, token_dim) (64, 128, 100, 16)

        # 定义n_layers个注意力层，每个注意力层都包含一个TabularAttentionLayer，TabularAttentionLayer的定义在src/stack/modules/attention.py中
        self.layers = nn.ModuleList(
            [
                TabularAttentionLayer(
                    token_dim=token_dim,
                    n_cells=n_cells,
                    n_hidden=n_hidden,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # 输入 -> Linear -> GELU -> Dropout -> Linear
        # 用于compute_nb_parameters()
        self.output_mlp = nn.Sequential(
            # (8192, 1600) --> (8192,3200)
            nn.Linear(n_hidden * token_dim, n_hidden * token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # (8192,3200) --> (8192,n_genes * 2)
            nn.Linear(n_hidden * token_dim * 2, n_genes * 2),
        )

        self.apply(self._init_weights)

    # 初始化权重，用于初始化模型参数
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为0，缩放项为1
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布，均值为0，方差为0.2
            nn.init.normal_(module.weight, mean=0, std=0.2)

    # 将n_genes降维到n_hidden * token_dim，然后reshape为(batch_size, n_cells, n_hidden, token_dim)
    def _reduce_and_tokenize(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, n_cells, _ = features.shape
        reduced = self.gene_reduction(features)
        return reduced.reshape(batch_size, n_cells, self.n_hidden, self.token_dim)

    # tokens: (batch_size, n_cells, n_hidden, token_dim)
    def _run_attention_layers(
        self,
        tokens: torch.Tensor,
        gene_attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Sequence[torch.Tensor]]]:
        attn_maps: List[torch.Tensor] = []
        x = tokens # x: (batch_size, n_cells, n_hidden, token_dim)
        for layer in self.layers: # layer: TabularAttentionLayer，参考src/stack/modules/attention.py中的TabularAttentionLayer的foward过程，同时计算cell-wise self-attention和gene-wise self-attention
            x, attn = layer(x, self.gene_pos_embedding, gene_attn_mask, return_attn)
            if return_attn:
                attn_maps.append(attn)
                # attn: (batch_size, n_heads, n_cells, n_cells)
        if return_attn: # 如果需要返回注意力图，则返回注意力图
            return x, attn_maps # x: (batch_size, n_cells, n_hidden, token_dim)，attn_maps: List[torch.Tensor]，每个元素是(batch_size, n_heads, n_cells, n_cells) --> gene_attn（细胞间的注意力）
        return x # 如果不需要返回注意力图，则返回计算后的特征

    # 计算NB参数，输入为(batch_size, n_cells, n_hidden, token_dim)，输出为(batch_size, n_cells, n_genes, 2)，分别是mean, dispersion, px_scale
    def _compute_nb_parameters(
        self,
        final_cell_embeddings: torch.Tensor,
        observed_lib_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # final_cell_embeddings: (batch_size, n_cells, n_hidden * token_dim) --> (64, 128, 1600)
        batch_size, n_cells, _ = final_cell_embeddings.shape
        flat_embeddings = final_cell_embeddings.reshape(batch_size * n_cells, -1) # flat_embeddings: (64 * 128, 1600) --> (8192, 1600)
        # decoder，为每个gene输出两个参数：mean, dispersion
        output = self.output_mlp(flat_embeddings) # (8192,1600) --> (8192,n_genes * 2)
        output = output.reshape(batch_size, n_cells, self.n_genes, 2)  # output: (64, 128, n_genes, 2)

        # 基因 $g$ 在该细胞中表达的相对强
        px_scale_logits = output[..., 0] # output[b, c, g, 0]

        # 基因 $g$ 在该细胞中表达的离散度 $\theta$ 
        # 使用 softplus 是为了确保 $\theta$ 永远为正数
        
        # NB分布的方差：$\sigma^2 = \mu + \frac{\mu^2}{\theta}$
        nb_dispersion = F.softplus(output[..., 1])

        # 表达比例，同一个细胞内所有基因的 px_scale 相加等于 1
        px_scale = F.softmax(px_scale_logits, dim=-1)

        # NB分布的均值，代表模型预测的该基因真实的原始计数（Raw Counts）期望值
        nb_mean = px_scale * observed_lib_size
        return nb_mean, nb_dispersion, px_scale

    
    
    def apply_mask(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_cells, n_genes = features.shape # (B, n_cells, n_genes)
        device = features.device

        # 随机生成一个mask比例，范围在mask_rate_min和mask_rate_max之间
        mask_rate = torch.empty(1, device=device).uniform_(
            self.mask_rate_min, self.mask_rate_max
        ).item()
        n_genes_to_mask = int(n_genes * mask_rate) # 计算需要mask的基因数量

        mask_indices = torch.randperm(n_genes, device=device)[:n_genes_to_mask] # 随机生成需要mask的基因索引

        mask = torch.zeros(batch_size, n_cells, n_genes, dtype=torch.bool, device=device)
        mask[:, :, mask_indices] = True # 将需要mask的基因索引位置设置为True

        masked_features = features.clone()
        masked_features[mask] = 0.0
        return masked_features, mask

    def forward(
        self,

        # features: (batch_size, n_cells, n_genes_原始) --> (64, 128, n_genes)
        features: torch.Tensor,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, n_cells, _ = features.shape
        device = features.device

        original_features = features.clone()

        # observed_lib_size: (batch_size, n_cells, 1) --> (64, 128, 1)
        # 求library size，即每个细胞的表达量之和
        observed_lib_size = original_features.sum(dim=-1, keepdim=True)

        features = torch.log1p(features) # 对features进行log1p变换

        masked_features, mask = self.apply_mask(features) # 应用mask，得到masked_features和mask

        tokens = self._reduce_and_tokenize(masked_features) # 将masked_features降维到n_hidden * token_dim，然后reshape为(batch_size, n_cells, n_hidden, token_dim)

        # 运行注意力层，输入为(batch_size, n_cells, n_hidden, token_dim)，输出为(batch_size, n_cells, n_hidden, token_dim)
        x = self._run_attention_layers(tokens)

        final_cell_embeddings = x.reshape(batch_size, n_cells, -1) # final_cell_embeddings: (batch_size, n_cells, n_hidden * token_dim) --> (64, 128, 100 * 16) --> (64, 128, 1600)

        nb_mean, nb_dispersion, px_scale = self._compute_nb_parameters(
            final_cell_embeddings, observed_lib_size
        )

        result = {
            "nb_mean": nb_mean,
            "nb_dispersion": nb_dispersion,
            "px_scale": px_scale,
            "observed_lib_size": observed_lib_size,
            "mask": mask,
            "cell_embeddings": final_cell_embeddings,
            "masked_features": masked_features,
            "original_features": original_features,
        }

        if return_loss:
            recon_loss, _ = self._compute_reconstruction_loss(
                nb_mean, nb_dispersion, original_features, mask
            )
            sw_loss = self._compute_sw_loss(final_cell_embeddings)
            total_loss = recon_loss + self.sw_weight * sw_loss

            result.update(
                {
                    "loss": total_loss,
                    "recon_loss": recon_loss,
                    "sw_loss": sw_loss,
                }
            )

            if not self.training:
                metrics = self._compute_eval_metrics(nb_mean, original_features, mask)
                result.update(metrics)
            else:
                zero = torch.tensor(0.0, device=device, dtype=nb_mean.dtype)
                result.update(
                    {
                        "masked_mae": zero,
                        "masked_corr": zero,
                        "mask_rate": zero,
                    }
                )

        return result


__all__ = ["StateICLModelBase"]
