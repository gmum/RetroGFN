"""
The code is partially adapted from LocalRetro repository (https://github.com/kaist-amsg/LocalRetro).
"""

import math
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import dgl
import torch
from dgllife.model import MPNNGNN
from dgllife.utils import (
    BaseAtomFeaturizer,
    CanonicalBondFeaturizer,
    WeaveAtomFeaturizer,
    mol_to_bigraph,
)
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch
from torchtyping import TensorType

from gflownet.gfns.retro.api.data_structures import Molecule, Pattern
from gflownet.gfns.retro.policy.featurizers import JointFeaturizer
from gflownet.utils.helpers import to_indices


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    """

    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Attention mechanism.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.size(-1), 1)
            mask = mask.unsqueeze(1).repeat(1, scores.size(1), 1, 1)
            scores = torch.masked_fill(scores, ~mask, -float("inf"))
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, output = self.attention(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)


class FeedForward(nn.Module):
    """
    Feed forward module.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        output = self.net(x)
        return self.layer_norm(x + output)


class GlobalReactivityAttention(nn.Module):
    """
    A transformer-like attention module.
    """

    def __init__(self, d_model: int, heads: int, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x


class EmbeddingGNN(nn.Module):
    """
    A GNN model that embeds the input graph into a fixed-size vector. It consists of `num_layers` GNN layers
    optionally followed by a global attention mechanism.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        node_featurizer: BaseAtomFeaturizer | WeaveAtomFeaturizer | JointFeaturizer,
        cache: bool = True,
        cache_dict: Dict[Union[Molecule, Pattern], Any] | None = None,
        use_attention: bool = True,
    ):
        """
        Initialize the model.

        Args:
            hidden_dim: a hidden dimension.
            num_layers: number of GNN layers.
            num_attention_heads: number of attention heads.
            node_featurizer: a featurizer for the nodes.
            cache: whether to cache the graph encodings (initial graph features obtained with a featurizer).
            cache_dict: an optional cache dictionary for shared graph encodings.
            use_attention: whether to use the global attention mechanism.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_featurizer = node_featurizer
        self.use_attention = use_attention
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.featurize_fn = partial(
            mol_to_bigraph,
            add_self_loop=True,
            canonical_atom_order=False,
            node_featurizer=self.node_featurizer,
            edge_featurizer=self.edge_featurizer,
        )
        self.gnn = MPNNGNN(
            node_in_feats=self.node_featurizer.feat_size(),
            edge_in_feats=self.edge_featurizer.feat_size(),
            edge_hidden_feats=hidden_dim,
            node_out_feats=hidden_dim,
            num_step_message_passing=num_layers,
        )
        if self.use_attention:
            self.attention = GlobalReactivityAttention(
                heads=num_attention_heads, d_model=hidden_dim, dropout=0.0
            )
        self.cache = cache
        if cache_dict is None:
            self.cache_dict: Dict[Union[Molecule, Pattern], Any] = {}
        else:
            self.cache_dict = cache_dict
        self.device = "cpu"

    def set_device(self, device: str):
        self.device = device
        super().to(device)

    def featurize(self, items: List[Molecule] | List[Pattern]) -> List[dgl.DGLGraph]:
        """
        Featurize the input items.
        Args:
            items: a list of molecules or patterns to featurize.

        Returns:
            a list of DGLGraphs.
        """
        if self.cache:
            graphs = []
            for item in items:
                cached_graph = self.cache_dict.get(item, None)
                if cached_graph is None:
                    cached_graph = self.featurize_fn(item.rdkit_mol)
                    self.cache_dict[item] = cached_graph
                graphs.append(cached_graph)
            return graphs
        return [self.featurize_fn(item.rdkit_mol) for item in items]

    def forward(
        self, items: List[Molecule] | List[Pattern]
    ) -> TensorType[float] | Tuple[TensorType[float], dgl.DGLGraph]:
        """
        Forward pass.

        Args:
            items: a list of molecules or patterns to embed.
        Returns:
            if `use_attention` is True, returns the nodes embeddings in the dense format of shape
                (batch_size, max_num_nodes, hidden_dim)
            Otherwise, returns the embeddings in the sparse format of shape (num_nodes, hidden_dim) along with
                a batched DGLGraph.
        """
        graphs = self.featurize(items)
        batch = dgl.batch(graphs).to(self.device)
        x = self.gnn(batch, batch.ndata["h"], batch.edata["e"])
        if self.use_attention:
            graph_indices = to_indices(batch.batch_num_nodes())
            x, mask = to_dense_batch(x=x, batch=graph_indices)
            _, x = self.attention.forward(x, mask=mask)
            x = torch.masked_fill(x, ~mask.unsqueeze(-1), 0.0)
            return x
        else:
            return x, batch
