import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm


class SocialContextExtractor(nn.Module):
    """
    Extract social context from graph structure.
    Measures user engagement patterns and influence.
    """
    
    def __init__(self, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # MLP to compute engagement scores
        self.engagement_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, node_emb, batch, graph_sizes):
        """
        Compute engagement-weighted graph representation.
        
        Args:
            node_emb: Node embeddings (num_nodes, hidden_dim)
            batch: Batch indices (num_nodes,)
            graph_sizes: Number of nodes per graph
        
        Returns:
            engagement_scores: (batch_size, hidden_dim) - engagement-weighted repr
        """
        engagement_scores = self.engagement_scorer(node_emb)  # (num_nodes, 1)
        
        # Weighted pooling by engagement
        weighted_emb = node_emb * engagement_scores
        social_context = global_mean_pool(weighted_emb, batch)  # (batch_size, hidden_dim)
        
        return social_context


class ExogenousContextEncoder(nn.Module):
    """
    GNN encoder for exogenous context (user engagement and social influence).
    Implements multi-layer GAT with batch normalization, residual connections, and edge pruning.
    
    Improvements:
    - Attention-based edge pruning to remove low-relevance connections
    - Residual connections between all consecutive layers to prevent over-smoothing
    - Adaptive residual scaling for stable training
    """
    
    def __init__(self, in_dim, hidden_dim, dropout=0.3, num_layers=4, edge_prune_threshold=0.05):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.edge_prune_threshold = edge_prune_threshold
        
        # Multi-layer GAT with attention-based pruning capability
        self.gat_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        # Residual projection layers for dimension matching (input -> hidden)
        self.residual_projs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim
            out_channels = hidden_dim
            
            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )
            self.bn_layers.append(BatchNorm(out_channels))
            
            # Residual projection: handle dimension mismatch
            if in_channels != out_channels:
                self.residual_projs.append(nn.Linear(in_channels, out_channels))
            else:
                self.residual_projs.append(None)
        
        # Adaptive residual scaling per layer
        self.residual_alphas = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(num_layers)
        ])
        
        self.social_context = SocialContextExtractor(hidden_dim, dropout)
    
    def _prune_edges_by_attention(self, edge_index, attention_weights):
        """
        Prune edges based on attention weights from GAT.
        
        Args:
            edge_index: (2, num_edges)
            attention_weights: Attention weights returned from GATConv
                - Could be shape (num_edges,) or (num_edges, heads)
        
        Returns:
            pruned_edge_index: (2, num_pruned_edges) - edges above threshold
        """
        if attention_weights is None:
            return edge_index
        
        # Handle different attention weight shapes
        if len(attention_weights.shape) > 1:
            # Shape is (num_edges, heads) - take mean across heads
            edge_attn = attention_weights.mean(dim=1)
        else:
            # Shape is (num_edges,) - use directly
            edge_attn = attention_weights
        
        # Verify shapes match
        if edge_attn.shape[0] != edge_index.shape[1]:
            # Shape mismatch - don't prune
            return edge_index
        
        # Keep edges with attention weight above threshold
        mask = edge_attn > self.edge_prune_threshold
        pruned_edge_index = edge_index[:, mask]
        
        return pruned_edge_index
    
    def forward(self, x, edge_index, batch, graph_sizes):
        """
        Forward pass with edge pruning and residual connections.
        
        Args:
            x: Node features (num_nodes, in_dim)
            edge_index: Edge indices (2, num_edges)
            batch: Batch indices (num_nodes,)
            graph_sizes: List of graph sizes
        
        Returns:
            exogenous_context: (batch_size, hidden_dim)
        """
        current_edge_index = edge_index
        
        for i in range(self.num_layers):
            # Save input for residual connection
            x_input = x
            
            # GAT layer with optional attention weights
            try:
                # Try with return_attention_weights (PyG 2.3+)
                x, attn_tuple = self.gat_layers[i](x, current_edge_index, return_attention_weights=True)
                # attn_tuple is (edge_index_out, attention_weights)
                if attn_tuple is not None and len(attn_tuple) == 2:
                    _, attn_weights = attn_tuple
                else:
                    attn_weights = None
            except TypeError:
                # Fallback for older PyG versions
                x = self.gat_layers[i](x, current_edge_index)
                attn_weights = None
            
            # Optional: Prune edges by attention (after first layer to stabilize)
            if i > 0 and attn_weights is not None:
                try:
                    current_edge_index = self._prune_edges_by_attention(current_edge_index, attn_weights)
                except Exception:
                    # If pruning fails for any reason, continue without it
                    pass
            
            # Batch norm
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection with dimension matching and adaptive scaling
            if i > 0:
                # Project input if needed
                if self.residual_projs[i] is not None:
                    x_input = self.residual_projs[i](x_input)
                
                # Adaptive residual: alpha controls contribution strength
                x = x + self.residual_alphas[i] * x_input
        
        # Extract social context with engagement weighting
        exogenous_context = self.social_context(x, batch, graph_sizes)
        
        return exogenous_context


class EndogenousPreferenceEncoder(nn.Module):
    """
    Learn text representations from news content and user preferences.
    Implements deep text representation learning instead of frozen embeddings.
    """
    
    def __init__(self, text_dim=384, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Multi-layer text encoder with batch normalization
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
        
        # User preference modeling (learns to weight text features)
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
        """
        Learn endogenous text representations.
        
        Args:
            text_emb: Frozen text embeddings (batch_size, text_dim)
        
        Returns:
            news_textual_emb: Learned news textual embedding
            pref_weights: User preference weights
        """
        # Initial text representation learning
        learned_text = self.text_encoder(text_emb)  # (batch_size, hidden_dim)
        
        # Learn user preference weights
        pref_weights = self.preference_weight(learned_text)  # (batch_size, hidden_dim)
        
        # Apply preference weighting
        weighted_text = learned_text * pref_weights
        
        # Refine news textual embedding
        news_textual_emb = self.news_refiner(weighted_text)
        
        return news_textual_emb, pref_weights


class EnhancedNewsClassifier(nn.Module):
    """
    News classifier with deep representation learning.
    Takes fused embedding and outputs fake news probability.
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
            
            nn.Linear(hidden_dim // 4, 2),  # 2 classes
        )
    
    def forward(self, fused_emb):
        """
        Classify news.
        
        Args:
            fused_emb: Fused representation (batch_size, hidden_dim)
        
        Returns:
            logits: Classification logits (batch_size, 2)
        """
        return self.classifier(fused_emb)
class MultiModalFusion(nn.Module):
    """Attention-based fusion with learnable gating and residual connections.

    Fuses endogenous (text) and exogenous (graph) context representations
    using learned attention weights. This module dynamically balances the
    importance of text features and social structure through a gating mechanism.

    Args:
        hidden_dim (int): Hidden representation dimension. Default: 256.
        dropout (float): Dropout probability for regularization. Default: 0.5.

    Shape:
        - Input: endogenous_emb (batch_size, hidden_dim),
                 exogenous_emb (batch_size, hidden_dim)
        - Output: (batch_size, hidden_dim)
    """

    def __init__(self, hidden_dim=256, dropout=0.5):
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
        """Fuse endogenous and exogenous embeddings with attention gating and residual.

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

        # Restore gated residual fusion: add graph structure signal
        fused_emb = fused_emb + exogenous_emb

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