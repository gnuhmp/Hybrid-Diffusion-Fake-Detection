import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from src.training.losses import FocalLoss
from src.models.components.encoders import (
    ExogenousContextEncoder,
    EndogenousPreferenceEncoder,
    EnhancedNewsClassifier,
)
from src.models.components.fusion import MultiModalFusion


class HyDFakeModel(nn.Module):
    """
    HyDFakeModel combining exogenous and endogenous context encoding.
    
    Architecture:
    1. Exogenous Context Encoder: Multi-layer GAT with social context extraction
    2. Endogenous Preference Encoder: Deep text representation learning
    3. Multi-Modal Fusion: Attention-based fusion with learnable weights
    4. News Classifier: Deep classifier for fake news detection
    
    Expected improvements:
    - Better social context modeling (+5-7% F1)
    - Learned text representations (+3-5% F1)
    - Attention-based fusion (+2-3% F1)
    - Total: +10-15% F1 over baseline
    """
    
    def __init__(self, text_frozen_dim=384, graph_in_dim=11, hidden_dim=256, 
                 dropout=0.3, mode='graph_text', num_gat_layers=4, edge_prune_threshold=0.05,
                 text_encoder_name=None):
        super().__init__()
        
        self.text_frozen_dim = text_frozen_dim
        self.graph_in_dim = graph_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.mode = mode
        self.num_gat_layers = num_gat_layers
        self.edge_prune_threshold = edge_prune_threshold
        self.text_encoder_name = text_encoder_name
        
        # ============= EXOGENOUS CONTEXT ENCODER =============
        # Multi-layer GNN for user engagement and social influence
        self.exogenous_encoder = ExogenousContextEncoder(
            in_dim=graph_in_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_gat_layers,
            edge_prune_threshold=edge_prune_threshold,
        )
        
        # ============= ENDOGENOUS PREFERENCE ENCODER =============
        # Deep text representation learning for user preferences
        self.endogenous_encoder = None
        if mode == 'graph_text':
            self.endogenous_encoder = EndogenousPreferenceEncoder(
                text_dim=text_frozen_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                encoder_name=text_encoder_name,
            )
        
        # ============= MULTI-MODAL FUSION =============
        # Attention-based fusion of exogenous and endogenous contexts
        self.fusion = None
        if mode == 'graph_text':
            self.fusion = MultiModalFusion(
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        
        # ============= NEWS CLASSIFIER =============
        # Deep classifier for final prediction
        self.classifier = EnhancedNewsClassifier(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    
    def forward(self, data):
        """
        Forward pass combining exogenous and endogenous contexts.
        
        Args:
            data: PyG Data object with:
                - x: Node features (num_nodes, graph_in_dim)
                - edge_index: Edge indices (2, num_edges)
                - batch: Batch indices (num_nodes,)
                - text_x: Text embeddings (batch_size, text_frozen_dim)
        
        Returns:
            logits: Classification logits (batch_size, 2)
        """
        batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
        graph_sizes = torch.unique(data.batch, return_counts=True)[1].tolist()
        
        # ============= EXOGENOUS CONTEXT =============
        # Extract social context from graph structure
        exogenous_context = self.exogenous_encoder(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
            graph_sizes=graph_sizes,
        )  # (batch_size, hidden_dim)
        
        # ============= ENDOGENOUS PREFERENCE =============
        endogenous_context = None
        if self.mode == 'graph_text' and hasattr(data, 'text_x'):
            # Pool text embeddings to graph level (node_level → graph_level)
            text_graph_level = global_mean_pool(data.text_x, data.batch)  # (batch_size, text_dim)
            # Learn text representations from frozen embeddings
            endogenous_context, pref_weights = self.endogenous_encoder(text_graph_level)
            # (batch_size, hidden_dim), (batch_size, hidden_dim)
        
        # ============= MULTI-MODAL FUSION =============
        if self.mode == 'graph_text' and endogenous_context is not None:
            # Fuse exogenous and endogenous using attention
            fused_emb = self.fusion(exogenous_context, endogenous_context)
        else:
            fused_emb = exogenous_context
        
        # ============= CLASSIFICATION =============
        logits = self.classifier(fused_emb)  # (batch_size, 2)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, data):
        """
        Get attention weights for interpretability.
        Shows how much exogenous vs endogenous context contributes.
        """
        with torch.no_grad():
            batch_size = data.batch.max().item() + 1
            exo = self.exogenous_encoder(data.x, data.edge_index, data.batch, 
                                         torch.unique(data.batch, return_counts=True)[1].tolist())
            
            if self.mode == 'graph_text' and hasattr(data, 'text_x'):
                endo, _ = self.endogenous_encoder(data.text_x)
                # Get fusion weights
                combined = torch.cat([exo, endo], dim=1)
                weights = self.fusion.gate(combined)  # (batch_size, 2)
                return weights  # [exogenous_weight, endogenous_weight]
        return None
