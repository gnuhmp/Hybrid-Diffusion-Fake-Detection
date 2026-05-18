import argparse
import os
from collections import deque

import numpy as np
import torch
from tqdm import tqdm


class GraphAugmentor:
    """
    Augment node features from 3D to 11D with diffusion-aware signals.
    
    Original 3D features:
        - in_degree
        - out_degree
        - root_flag
    
    Added 8D diffusion signals:
        - depth (BFS distance from root)
        - breadth_rank (percentile at same level)
        - structural_virality (out_degree * depth ratio)
        - early_propagation_flag (depth <= 3)
        - temporal_decay (unused if no timestamps)
        - user_credibility (unused if not available)
        - degree_centrality (normalized out_degree)
        - clustering_coefficient (local clustering)
    
    These features capture cascade structure important for fake news detection.
    """
    
    @staticmethod
    def compute_depths_and_breadth(edge_index, num_nodes, root_idx=0):
        """
        BFS from root to compute depths and breadth at each level.
        
        Returns:
            depths: List of depths for each node
            breadth_at_depth: Dict of node counts at each depth
        """
        depths = [-1] * num_nodes
        depths[root_idx] = 0
        breadth_at_depth = {}
        
        queue = deque([root_idx])
        
        # Build adjacency list
        adj = [[] for _ in range(num_nodes)]
        if edge_index.numel() > 0:
            src_list = edge_index[0].tolist()
            dst_list = edge_index[1].tolist()
            for src, dst in zip(src_list, dst_list):
                adj[src].append(dst)
        
        # BFS
        while queue:
            node = queue.popleft()
            d = depths[node]
            
            if d not in breadth_at_depth:
                breadth_at_depth[d] = 0
            breadth_at_depth[d] += 1
            
            for neighbor in adj[node]:
                if depths[neighbor] == -1:
                    depths[neighbor] = d + 1
                    queue.append(neighbor)
        
        return depths, breadth_at_depth
    
    @staticmethod
    def augment_features(data, root_idx=0):
        """
        Augment node features to 11D.
        
        Args:
            data: PyG Data object with x (num_nodes, 3)
            root_idx: Root node index
        
        Returns:
            augmented_x: (num_nodes, 11) augmented features
        """
        x = data.x  # (num_nodes, 3): [in_degree, out_degree, root_flag]
        num_nodes = x.shape[0]
        edge_index = data.edge_index
        
        # Compute depths
        depths, breadth_at_depth = GraphAugmentor.compute_depths_and_breadth(
            edge_index, num_nodes, root_idx
        )
        depths = torch.tensor(depths, dtype=torch.float32)
        
        # Compute out_degree for each node
        out_degree = torch.zeros(num_nodes, dtype=torch.float32)
        if edge_index.numel() > 0:
            src_list = edge_index[0].tolist()
            for src in src_list:
                out_degree[src] += 1
        
        # Compute features
        max_depth = depths[depths >= 0].max().item() if (depths >= 0).any() else 0
        
        # 1. Depth (normalized)
        depth_norm = depths.clone()
        if max_depth > 0:
            depth_norm = depth_norm / max_depth
        depth_norm[depths < 0] = 0  # unreachable nodes
        
        # 2. Breadth rank (percentile at same depth) - IMPROVED for PolitiFact
        breadth_rank = torch.zeros(num_nodes, dtype=torch.float32)
        depth_nodes = {}  # nodes grouped by depth
        for i, d in enumerate(depths):
            if d >= 0:
                d_int = int(d)
                if d_int not in depth_nodes:
                    depth_nodes[d_int] = []
                depth_nodes[d_int].append((i, out_degree[i].item()))
        
        # Compute actual percentile ranks
        for d, nodes in depth_nodes.items():
            if len(nodes) > 0:
                # Sort by out_degree descending
                nodes_sorted = sorted(nodes, key=lambda x: x[1], reverse=True)
                for rank, (node_id, _) in enumerate(nodes_sorted):
                    breadth_rank[node_id] = rank / max(1, len(nodes) - 1)
        
        # 3. Structural virality - ENHANCED with in_degree weighting
        virality = torch.zeros(num_nodes, dtype=torch.float32)
        if max_depth > 0:
            in_degree = torch.zeros(num_nodes, dtype=torch.float32)
            if edge_index.numel() > 0:
                dst_list = edge_index[1].tolist()
                for dst in dst_list:
                    in_degree[dst] += 1
            
            for i, d in enumerate(depths):
                if d >= 0:
                    # Virality: (out_degree + in_degree) * (depth_ratio)
                    virality[i] = (out_degree[i] * 0.7 + in_degree[i] * 0.3) * (d / max_depth)
        
        # 4. Early propagation flag (depth <= 2 for PolitiFact) - more aggressive
        early_prop = (depths <= 2).float()
        early_prop[depths < 0] = 0
        
        # 5-7. Degree centrality (normalized out_degree)
        max_out_degree = max(1.0, out_degree.max().item())
        degree_centrality = out_degree / max_out_degree
        
        # 8. Leaf flag (out_degree == 0)
        is_leaf = (out_degree == 0).float()
        
        # 9. Clustering coefficient - IMPROVED: use actual neighbor connections
        clustering_coeff = torch.zeros(num_nodes, dtype=torch.float32)
        if edge_index.numel() > 0:
            # Build adjacency for clustering coefficient calculation
            adj = [set() for _ in range(num_nodes)]
            src_list = edge_index[0].tolist()
            dst_list = edge_index[1].tolist()
            for src, dst in zip(src_list, dst_list):
                adj[src].add(dst)
                adj[dst].add(src)  # Treat as undirected for clustering
            
            # Compute clustering coefficient
            for i in range(num_nodes):
                neighbors = list(adj[i])
                if len(neighbors) > 1:
                    # Count edges between neighbors
                    edges_between = 0
                    for j in range(len(neighbors)):
                        for k in range(j + 1, len(neighbors)):
                            if neighbors[k] in adj[neighbors[j]]:
                                edges_between += 1
                    max_edges = len(neighbors) * (len(neighbors) - 1) / 2
                    clustering_coeff[i] = edges_between / max_edges if max_edges > 0 else 0
                else:
                    clustering_coeff[i] = 0
        
        # Stack all features: original (3D) + new (8D) = 11D
        augmented_x = torch.cat([
            x,                          # 3D: in_degree, out_degree, root_flag
            depth_norm.unsqueeze(1),    # 1D
            breadth_rank.unsqueeze(1),  # 1D
            virality.unsqueeze(1),      # 1D
            early_prop.unsqueeze(1),    # 1D
            degree_centrality.unsqueeze(1),  # 1D
            out_degree.unsqueeze(1) / (max_out_degree + 1),  # 1D (log-like)
            is_leaf.unsqueeze(1),       # 1D
            clustering_coeff.unsqueeze(1),  # 1D
        ], dim=1)
        
        return augmented_x  # (num_nodes, 11)


class GraphPruner:
    """
    Prune graphs to remove noise and reduce memory.
    
    Keep:
        - Root node (always)
        - Nodes with depth <= max_depth
        - Nodes with out_degree >= min_degree
    
    Remove:
        - Deep nodes (depth > max_depth) - likely noise
        - Leaf nodes with out_degree < min_degree
    
    Expected: -40-60% nodes removed, ±0-2% accuracy impact
    """
    
    @staticmethod
    def prune_graph(data, root_idx=0, max_depth=5, min_degree=1):
        """
        Prune graph by removing deep/isolated nodes.
        
        Args:
            data: PyG Data object
            root_idx: Root node index
            max_depth: Maximum depth to keep
            min_degree: Minimum out-degree to keep (except root)
        
        Returns:
            pruned_data: New Data object with pruned nodes/edges
        """
        from torch_geometric.data import Data
        
        edge_index = data.edge_index
        num_nodes = data.x.shape[0]
        
        # Compute depths
        depths, _ = GraphAugmentor.compute_depths_and_breadth(
            edge_index, num_nodes, root_idx
        )
        
        # Compute out_degree
        out_degree = torch.zeros(num_nodes, dtype=torch.long)
        if edge_index.numel() > 0:
            src_list = edge_index[0].tolist()
            for src in src_list:
                out_degree[src] += 1
        
        # Nodes to keep
        keep_mask = torch.zeros(num_nodes, dtype=torch.bool)
        keep_mask[root_idx] = True  # Always keep root
        
        for i in range(num_nodes):
            if i == root_idx:
                continue
            d = depths[i]
            deg = out_degree[i].item()
            
            # Keep if depth valid and (has children or is within max_depth)
            if 0 <= d <= max_depth:
                keep_mask[i] = True
            elif deg >= min_degree:  # Allow high-degree nodes even if deep
                keep_mask[i] = True
        
        # Create node mapping
        old_to_new = {}
        new_node_id = 0
        for old_id in range(num_nodes):
            if keep_mask[old_id]:
                old_to_new[old_id] = new_node_id
                new_node_id += 1
        
        # Filter edges
        new_edges = []
        if edge_index.numel() > 0:
            src_list = edge_index[0].tolist()
            dst_list = edge_index[1].tolist()
            
            for src, dst in zip(src_list, dst_list):
                if keep_mask[src] and keep_mask[dst]:
                    new_src = old_to_new[src]
                    new_dst = old_to_new[dst]
                    new_edges.append([new_src, new_dst])
        
        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
        else:
            new_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create new data
        new_x = data.x[keep_mask]
        new_num_nodes = keep_mask.sum().item()
        
        pruned_data = Data(
            x=new_x,
            edge_index=new_edge_index,
            y=data.y,
        )
        
        # =========================================================
        # ĐÂY LÀ ĐOẠN ĐÃ ĐƯỢC SỬA LỖI (Cắt tỉa đồng bộ text_x)
        # =========================================================
        if hasattr(data, 'text_x'):
            pruned_data.text_x = data.text_x[keep_mask]
        if hasattr(data, 'struc_x'):
            pruned_data.struc_x = data.struc_x[keep_mask]
        if hasattr(data, 'graph_id'):
            pruned_data.graph_id = data.graph_id
            
        return pruned_data


def augment_dataset(dataset, root_idx=0):
    """
    Augment all graphs in dataset (3D → 11D features).
    """
    augmentor = GraphAugmentor()
    augmented = []
    
    for data in tqdm(dataset, desc='Augmenting features'):
        data.x = augmentor.augment_features(data, root_idx)
        augmented.append(data)
    
    return augmented


def prune_dataset(dataset, max_depth=5, min_degree=1):
    """
    Prune all graphs in dataset.
    """
    pruner = GraphPruner()
    pruned = []
    
    for data in tqdm(dataset, desc='Pruning graphs'):
        pruned_data = pruner.prune_graph(data, max_depth=max_depth, min_degree=min_degree)
        pruned.append(pruned_data)
    
    return pruned