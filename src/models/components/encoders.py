import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class SocialContextExtractor(nn.Module):
    """Extract social context from graph structure via engagement patterns.

    This module measures user engagement patterns and influence by computing
    engagement-weighted pooled graph representations. It uses an MLP-based
    scoring mechanism to weight nodes based on their engagement signals.

    Args:
        hidden_dim (int): Dimension of hidden embeddings. Default: 256.
        dropout (float): Dropout probability. Default: 0.3.

    Shape:
        - Input: node_emb (num_nodes, hidden_dim), batch (num_nodes,)
        - Output: (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # MLP to compute per-node engagement scores
        self.engagement_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_emb, batch, graph_sizes):
        """Compute engagement-weighted graph representation.

        Args:
            node_emb (torch.Tensor): Node embeddings of shape (num_nodes, hidden_dim).
            batch (torch.Tensor): Batch indices of shape (num_nodes,).
            graph_sizes (list or torch.Tensor): Number of nodes per graph.

        Returns:
            torch.Tensor: Engagement-weighted representations of shape
                (batch_size, hidden_dim).
        """
        # Compute engagement scores for each node
        engagement_scores = self.engagement_scorer(node_emb)  # (num_nodes, 1)

        # Perform weighted pooling by engagement strength
        weighted_emb = node_emb * engagement_scores
        social_context = global_mean_pool(weighted_emb, batch)

        return social_context


class ExogenousContextEncoder(nn.Module):
    """Multi-layer GAT encoder with unified architecture for all datasets.

    Encodes exogenous context (user engagement and social influence) using
    a stacked Graph Attention Network with consistent architecture across all datasets.
    
    Uses LayerNorm for stable training with varying graph sizes, adaptive residual
    connections to prevent over-smoothing, and attention-based edge pruning.

    Args:
        in_dim (int): Input feature dimension.
        hidden_dim (int): Hidden representation dimension. Default: 256.
        dropout (float): Dropout probability. Default: 0.3.
        num_layers (int): Number of GAT layers. Default: 4.
        edge_prune_threshold (float): Attention threshold for edge pruning
            (0.0-1.0). Default: 0.05.
        dataset_name (str): Dataset name (kept for compatibility, not used for branching).
            Default: 'gossipcop'.

    Shape:
        - Input: x (num_nodes, in_dim), edge_index (2, num_edges)
        - Output: (batch_size, hidden_dim)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=256,
        dropout=0.3,
        num_layers=4,
        edge_prune_threshold=0.05,
        dataset_name="gossipcop",
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.edge_prune_threshold = edge_prune_threshold
        self.dataset_name = dataset_name

        # Multi-layer GAT with attention-based pruning capability
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()  # LayerNorm for stability across all datasets

        # Residual projection layers for dimension matching
        self.residual_projs = nn.ModuleList()

        # Multi-head attention for diverse pattern learning on small graphs
        num_heads = 4  # Use 4 heads to capture diverse attention patterns
        head_dim = hidden_dim // num_heads  # Dimension per head
        # Actual output dimension after multi-head concatenation
        actual_out_dim = num_heads * head_dim  # = hidden_dim

        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim
            out_channels = head_dim  # Output per head

            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_heads,  # Use 4 heads instead of 1
                    concat=True,  # Concatenate head outputs: 4 * head_dim = hidden_dim
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            
            # LayerNorm for all datasets (works better with varying graph sizes)
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

            # Residual projection: match actual output dimension (accounts for multi-head concat)
            # After concat: actual output is num_heads * head_dim = hidden_dim
            if in_channels != actual_out_dim:
                self.residual_projs.append(nn.Linear(in_channels, actual_out_dim))
            else:
                self.residual_projs.append(None)

        # Adaptive residual scaling per layer (unconditionally initialized for all datasets)
        self.residual_alphas = nn.ParameterList(
            [nn.Parameter(torch.ones(1) * 0.5) for _ in range(num_layers)]
        )

        self.social_context = SocialContextExtractor(hidden_dim, dropout)

    def _prune_edges_by_attention(self, edge_index, attention_weights):
        """Prune edges based on GAT attention weight threshold.

        Removes edges whose attention weights fall below the pruning threshold,
        reducing noise and keeping only high-confidence graph connections.

        Args:
            edge_index (torch.Tensor): Edge indices of shape (2, num_edges).
            attention_weights (torch.Tensor or None): Attention weights from
                GATConv, either shape (num_edges,) or (num_edges, heads).

        Returns:
            torch.Tensor: Pruned edge indices of shape (2, num_pruned_edges).
        """
        if attention_weights is None:
            return edge_index

        # Normalize attention weights across heads if multi-head
        if len(attention_weights.shape) > 1:
            edge_attn = attention_weights.mean(dim=1)
        else:
            edge_attn = attention_weights

        # Validate shape compatibility
        if edge_attn.shape[0] != edge_index.shape[1]:
            return edge_index

        # Keep edges with attention weight above threshold
        mask = edge_attn > self.edge_prune_threshold

        # Safe fallback: prevent crash when all edges are pruned
        if mask.sum() == 0:
            return edge_index

        pruned_edge_index = edge_index[:, mask]

        return pruned_edge_index

    def forward(self, x, edge_index, batch, graph_sizes):
        """Forward pass with unified architecture for all datasets.

        Stacks GAT layers with LayerNorm, edge pruning, and residual connections.
        This unified pipeline works optimally across PolitiFact, GossipCop, and
        other datasets without any dataset-specific branching.

        Args:
            x (torch.Tensor): Node features of shape (num_nodes, in_dim).
            edge_index (torch.Tensor): Edge indices of shape (2, num_edges).
            batch (torch.Tensor): Batch indices of shape (num_nodes,).
            graph_sizes (list or torch.Tensor): Number of nodes per graph.

        Returns:
            torch.Tensor: Exogenous context representation of shape
                (batch_size, hidden_dim).
        """
        current_edge_index = edge_index

        # ============= UNIFIED PIPELINE FOR ALL DATASETS =============
        # Single architecture: GATConv -> LayerNorm -> ReLU -> Dropout -> Residual
        for i in range(self.num_layers):
            x_input = x

            # Apply GAT layer with optional attention weights retrieval
            try:
                x, attn_tuple = self.gat_layers[i](
                    x, current_edge_index, return_attention_weights=True
                )
                attn_weights = (
                    attn_tuple[1] if attn_tuple and len(attn_tuple) == 2 else None
                )
            except TypeError:
                x = self.gat_layers[i](x, current_edge_index)
                attn_weights = None

            # Attention-based edge pruning (after first layer to stabilize)
            if i > 0 and attn_weights is not None:
                try:
                    current_edge_index = self._prune_edges_by_attention(
                        current_edge_index, attn_weights
                    )
                except Exception:
                    pass

            # LayerNorm for stability (works well with varying graph sizes)
            x = self.norm_layers[i](x)
            
            # Activation and dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Adaptive residual connection (unconditional for all datasets)
            if i > 0:
                if self.residual_projs[i] is not None:
                    x_input = self.residual_projs[i](x_input)
                x = x + self.residual_alphas[i] * x_input

        # Extract social context with engagement weighting
        exogenous_context = self.social_context(x, batch, graph_sizes)

        return exogenous_context


class EndogenousPreferenceEncoder(nn.Module):
    """Learn text representations and user preference weights from news content.

    Encodes endogenous context by learning deep text representations from
    frozen embeddings instead of using them directly. The module:

    - Transforms frozen text embeddings into learned representations
    - Models user preferences as feature importance weights
    - Refines news textual embeddings via preference weighting
    - Uses batch normalization for training stability

    Args:
        text_dim (int): Dimension of input frozen embeddings. Default: 384.
        hidden_dim (int): Hidden representation dimension. Default: 256.
        dropout (float): Dropout probability. Default: 0.3.

    Shape:
        - Input: text_emb (batch_size, text_dim)
        - Output: news_textual_emb (batch_size, hidden_dim),
                  pref_weights (batch_size, hidden_dim)
    """

    def __init__(self, text_dim=384, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Multi-layer text encoder: learn from frozen embeddings
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        # User preference modeling: learns to weight text features
        self.preference_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )

        # News textual embedding refinement
        self.news_refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, text_emb):
        """Learn endogenous text representations and preference weights.

        Processes frozen text embeddings through multiple transformation layers
        to learn both content representations and user preference patterns.

        Args:
            text_emb (torch.Tensor): Frozen text embeddings of shape
                (batch_size, text_dim).

        Returns:
            tuple: A tuple containing:
                - news_textual_emb (torch.Tensor): Learned news textual
                    embedding of shape (batch_size, hidden_dim).
                - pref_weights (torch.Tensor): User preference feature weights
                    of shape (batch_size, hidden_dim).
        """
        # Learn text representation from frozen embeddings
        learned_text = self.text_encoder(text_emb)

        # Model user preferences as feature importance weights
        pref_weights = self.preference_weight(learned_text)

        # Apply preference weighting to learned representation
        weighted_text = learned_text * pref_weights

        # Refine news textual embedding
        news_textual_emb = self.news_refiner(weighted_text)

        return news_textual_emb, pref_weights


class MultiModalFusion(nn.Module):
    """Attention-based fusion with learnable gating and residual connections.

    Fuses endogenous (text) and exogenous (graph) context representations
    using learned attention weights. This module dynamically balances the
    importance of text features and social structure through a gating mechanism.

    Args:
        hidden_dim (int): Hidden representation dimension. Default: 256.
        dropout (float): Dropout probability for regularization. Default: 0.3.

    Shape:
        - Input: endogenous_emb (batch_size, hidden_dim),
                 exogenous_emb (batch_size, hidden_dim)
        - Output: (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable gating network: computes normalized attention weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
            nn.Softmax(dim=-1),
        )

        # Refinement layer: projects fused representation for downstream tasks
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, endogenous_emb, exogenous_emb):
        """Fuse endogenous and exogenous embeddings with attention gating.

        Args:
            endogenous_emb (torch.Tensor): Text representation of shape
                (batch_size, hidden_dim).
            exogenous_emb (torch.Tensor): Graph representation of shape
                (batch_size, hidden_dim).

        Returns:
            torch.Tensor: Fused representation of shape (batch_size, hidden_dim).
        """
        # Concatenate embeddings for joint processing
        combined = torch.cat([endogenous_emb, exogenous_emb], dim=-1)

        # Compute normalized attention weights via gating mechanism
        attn_weights = self.attention(combined)

        # Extract per-modality weights: shape (batch_size, 1)
        w_text = attn_weights[:, 0].unsqueeze(1)
        w_graph = attn_weights[:, 1].unsqueeze(1)

        # Fuse embeddings: weighted combination of both modalities
        fused_emb = (w_text * endogenous_emb) + (w_graph * exogenous_emb)

        # Refine fused representation via projection layer
        return self.project(fused_emb)

    def gate(self, combined_features):
        """Compute gating weights from combined features.

        Args:
            combined_features (torch.Tensor): Concatenated embeddings of shape
                (batch_size, hidden_dim * 2).

        Returns:
            torch.Tensor: Attention weights of shape (batch_size, 2).
        """
        return self.attention(combined_features)


class EnhancedNewsClassifier(nn.Module):
    """Deep news classifier for fake news detection.

    Classifies news into fake or real categories using a deep multi-layer
    network. Takes fused embeddings from exogenous and endogenous context
    and outputs probabilistic predictions via softmax over two classes.

    Args:
        hidden_dim (int): Input feature dimension. Default: 256.
        dropout (float): Dropout probability. Default: 0.3.

    Shape:
        - Input: fused_emb (batch_size, hidden_dim)
        - Output: (batch_size, 2) - logits for [real, fake]
    """

    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),
        )

    def forward(self, fused_emb):
        """Classify news as fake or real.

        Args:
            fused_emb (torch.Tensor): Fused representation from exogenous and
                endogenous encoders of shape (batch_size, hidden_dim).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, 2).
                Index 0 corresponds to real news, index 1 to fake news.
        """
        return self.classifier(fused_emb)