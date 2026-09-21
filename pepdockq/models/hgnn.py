import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import ones, reset
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_scatter import scatter_add


_EDGE_TYPES = [
    ('peptide', 'near', 'peptide'),
    ('protein', 'near', 'protein'),
    ('peptide', 'near', 'protein'),
    ('protein', 'near', 'peptide'),
]
_NODE_TYPES = ['peptide', 'protein']


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly drop the entire residual branch per node."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    mask = x.new_empty(x.size(0), 1).bernoulli_(keep_prob)
    return x * mask / keep_prob



class HGNNlayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4,
                 attn_dropout=0.15, hidden_dropout=0.15, drop_path_rate=0.1):
        super().__init__()
        assert out_channels % heads == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.d_head = out_channels // heads
        self.attn_dropout = attn_dropout
        self.hidden_dropout = hidden_dropout
        self.drop_path_rate = drop_path_rate

        # Per-edge-type edge update (300k samples can afford some per-type capacity)
        self.edge_attr_proj = nn.ModuleDict()
        self.edge_norm = nn.ModuleDict()
        for et in _EDGE_TYPES:
            key = '__'.join(et)
            self.edge_attr_proj[key] = nn.Sequential(
                nn.Linear(in_channels * 3, in_channels),
                nn.GELU(),
                nn.Dropout(hidden_dropout),
            )
            self.edge_norm[key] = nn.LayerNorm(in_channels)

        # Shared Q/o per node type, K/V per edge type. This keeps the
        # cross-type interaction expressive but avoids the original's
        # full per-type matrices.
        self.q_lin = nn.ModuleDict({
            nt: Linear(in_channels, out_channels) for nt in _NODE_TYPES
        })
        self.o_lin = nn.ModuleDict({
            nt: Linear(out_channels, out_channels) for nt in _NODE_TYPES
        })
        self.k_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        for et in _EDGE_TYPES:
            key = '__'.join(et)
            self.k_lin[key] = Linear(in_channels, out_channels)
            self.v_lin[key] = Linear(in_channels, out_channels)

        self.skip = nn.ParameterDict({
            nt: Parameter(torch.ones(1)) for nt in _NODE_TYPES
        })
        self.node_norm = nn.ModuleDict({
            nt: nn.LayerNorm(out_channels) for nt in _NODE_TYPES
        })

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_attr_proj)
        reset(self.edge_norm)
        reset(self.q_lin)
        reset(self.k_lin)
        reset(self.v_lin)
        reset(self.o_lin)
        reset(self.node_norm)
        ones(self.skip)

    def forward(self, data):
        H = self.heads
        D = self.d_head

        for et in _EDGE_TYPES:
            edge_index = data[et].edge_index
            src_t, _, dst_t = et
            edge_attr_stack = torch.cat([
                data[et].edge_attr,
                data[src_t].x[edge_index[0]],
                data[dst_t].x[edge_index[1]],
            ], dim=1)
            key = '__'.join(et)
            data[et].edge_attr = self.edge_norm[key](
                data[et].edge_attr + self.edge_attr_proj[key](edge_attr_stack)
            )

        q_dict = {nt: self.q_lin[nt](data[nt].x).view(-1, H, D)
                  for nt in _NODE_TYPES}

        out_dict = {nt: [] for nt in _NODE_TYPES}
        for et in _EDGE_TYPES:
            edge_index = data[et].edge_index
            edge_attr = data[et].edge_attr
            _, _, dst_t = et
            key = '__'.join(et)

            q_dst = q_dict[dst_t][edge_index[1]]
            k_edge = self.k_lin[key](edge_attr).view(-1, H, D)
            v_edge = self.v_lin[key](edge_attr).view(-1, H, D)

            score = (q_dst * k_edge).sum(dim=-1) / math.sqrt(D)
            alpha = pyg_softmax(score, edge_index[1],
                                num_nodes=q_dict[dst_t].size(0))
            alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training)

            o = alpha.unsqueeze(-1) * v_edge
            num_dst = q_dict[dst_t].size(0)
            agg = scatter_add(o.reshape(-1, H * D), edge_index[1],
                              dim=0, dim_size=num_dst)
            out_dict[dst_t].append(agg)

        for nt in _NODE_TYPES:
            out = torch.stack(out_dict[nt], dim=0).sum(dim=0)
            out = self.o_lin[nt](out)
            out = F.dropout(out, p=self.hidden_dropout, training=self.training)
            out = drop_path(out, self.drop_path_rate, self.training)

            alpha_skip = self.skip[nt].sigmoid()
            out = alpha_skip * out + (1 - alpha_skip) * data[nt].x
            data[nt].x = self.node_norm[nt](out)

        return data


class HGNN(nn.Module):
    """HGNN — OOD-aware version targeting low protein-identity test sets.

    Key OOD-targeted design choices:
      - ESM bottleneck: 1280 -> 64 (vs original 128) to limit ESM-specific fitting
      - ESM noise augmentation: per-residue mask + Gaussian noise during training
      - Structural features (dihedrals, edge geometry) given proportionally more
        capacity so the model can rely on transferable signals
      - Stronger feat_dropout on ESM path specifically
      - Standard transformer-style regularization (DropPath, attn dropout, etc.)
    """

    def __init__(self, num_layers, hidden_channels,
                 features, heads, num_error_bins, 
                 # ESM-specific regularization (OOD core)
                 esm_seq_dropout: float = 0.15,
                 esm_noise_std: float = 0.1,
                 esm_path_dropout: float = 0.3,
                 # General regularization
                 attn_dropout: float = 0.15,
                 hidden_dropout: float = 0.15,
                 drop_path_rate: float = 0.1,
                 feat_dropout: float = 0.15):
        super().__init__()

        feature_dim = {'esm': 1280, 'zscale': 5, 'dihedrals': 6, 'esmif_mutation_score': 20}
        _feat_hidden = {'esm': 128, 'esmif_mutation_score': 88, 'zscale': 8, 'dihedrals': 32}
        print(f"Feature hidden dims: {_feat_hidden}")

        self.esm_seq_dropout = esm_seq_dropout
        self.esm_noise_std = esm_noise_std
        self.features = features
        assert 'esm' in features

        self.feat_mlps = nn.ModuleDict()
        feat_concat_dim = 0
        for feat in self.features:
            h = _feat_hidden[feat]
            # ESM path gets heavier dropout than other features
            drop_p = esm_path_dropout if feat == 'esm' else feat_dropout
            self.feat_mlps[feat] = nn.Sequential(
                nn.Linear(feature_dim[feat], h),
                nn.GELU(),
                nn.Dropout(drop_p),
            )
            feat_concat_dim += h
        assert feat_concat_dim == hidden_channels, \
            f"Sum of per-feature hidden dims ({feat_concat_dim}) must equal hidden_channels ({hidden_channels})"

        self.node_input_norm = nn.LayerNorm(hidden_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(416, hidden_channels),
            nn.GELU(),
            nn.Dropout(feat_dropout),
        )

        dp_rates = [drop_path_rate * i / max(num_layers - 1, 1)
                    for i in range(num_layers)] if num_layers > 1 else [drop_path_rate]

        self.convs = nn.ModuleList([
            HGNNlayer(hidden_channels, hidden_channels, heads=heads,
                        attn_dropout=attn_dropout,
                        hidden_dropout=hidden_dropout,
                        drop_path_rate=dp_rates[i])
            for i in range(num_layers)
        ])

        self.num_error_bins = num_error_bins

        self.edge_interact_proj = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        self.error_head = nn.Linear(hidden_channels, num_error_bins, bias=False)

        self.pool_norm = nn.LayerNorm(hidden_channels * 2)
        self.dockq_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _augment_esm(self, esm: torch.Tensor) -> torch.Tensor:
        """Per-residue mask + additive Gaussian noise on ESM embeddings.

        Both are training-time only. The mask zeroes out the entire embedding
        of a residue with probability esm_seq_dropout; the noise perturbs the
        embedding to make the downstream model robust to ESM micro-variation.
        """
        if not self.training:
            return esm
        # per-residue mask (whole-vector dropout, like the original)
        if self.esm_seq_dropout > 0.0:
            mask = (torch.rand(esm.size(0), 1, device=esm.device) > self.esm_seq_dropout).float()
            esm = esm * mask / max(1.0 - self.esm_seq_dropout, 1e-6)  # inverted dropout scaling
        # additive Gaussian noise — scale to the local std so it's calibrated
        if self.esm_noise_std > 0.0:
            # estimate per-residue std on the fly; cheap and avoids global stats
            local_std = esm.std(dim=-1, keepdim=True).clamp(min=1e-3)
            noise = torch.randn_like(esm) * (self.esm_noise_std * local_std)
            esm = esm + noise
        return esm

    def forward(self, data):
        batchsize = int(data['peptide'].batch.max().item()) + 1

        pep_x, rec_x = [], []
        if 'esm' in self.features:
            pep_esm = self._augment_esm(data['peptide'].seq_feat[:, :1280])
            rec_esm = self._augment_esm(data['protein'].seq_feat[:, :1280])
            pep_x.append(self.feat_mlps['esm'](pep_esm))
            rec_x.append(self.feat_mlps['esm'](rec_esm))
        if 'zscale' in self.features:
            pep_x.append(self.feat_mlps['zscale'](data['peptide'].seq_feat[:, 1280:1285]))
            rec_x.append(self.feat_mlps['zscale'](data['protein'].seq_feat[:, 1280:1285]))
        if 'esmif_mutation_score' in self.features:
            pep_x.append(self.feat_mlps['esmif_mutation_score'](
                data['peptide'].seq_feat[:, 1286:1306] - data['peptide'].seq_feat[:, 1306:]))
            rec_x.append(self.feat_mlps['esmif_mutation_score'](
                data['protein'].seq_feat[:, 1286:1306] - data['protein'].seq_feat[:, 1306:]))
        if 'dihedrals' in self.features:
            pep_x.append(self.feat_mlps['dihedrals'](data['peptide'].node_s))
            rec_x.append(self.feat_mlps['dihedrals'](data['protein'].node_s))

        data['peptide'].x = self.node_input_norm(torch.cat(pep_x, dim=1))
        data['protein'].x = self.node_input_norm(torch.cat(rec_x, dim=1))

        for et in _EDGE_TYPES:
            data[et].edge_attr = self.edge_mlp(data[et].edge_s)

        for conv in self.convs:
            data = conv(data)

        pred_index = data['peptide', 'near', 'protein'].edge_index
        pred_attr = data['peptide', 'near', 'protein'].edge_attr
        rev_pred_attr = data['protein', 'near', 'peptide'].edge_attr

        interact_emb = torch.cat([
            0.5 * pred_attr + 0.5 * rev_pred_attr,
            data['peptide'].x[pred_index[0]],
            data['protein'].x[pred_index[1]],
        ], dim=1)
        interact_emb = self.edge_interact_proj(interact_emb)

        batch = data['peptide'].batch[pred_index[0]]
        pool_mean = global_mean_pool(interact_emb, batch, size=batchsize)
        pool_max = global_max_pool(interact_emb, batch, size=batchsize)
        pool_cat = torch.cat([pool_mean, pool_max], dim=1)
        pool_cat = self.pool_norm(pool_cat)
        pred_global = self.dockq_head(pool_cat).squeeze(1).sigmoid()

        pred_error = self.error_head(interact_emb)

        pair_error_label = None
        if hasattr(data['peptide', 'near', 'protein'], 'edge_label'):
            edge_label = data['peptide', 'near', 'protein'].edge_label
            if edge_label is not None and edge_label.numel() > 0:
                # edge_label[:, 1] is decoy dist, edge_label[:, 0] is exp dist
                pair_error_label = edge_label[:, 1] - edge_label[:, 0]

        return pred_error, pair_error_label, pred_global

