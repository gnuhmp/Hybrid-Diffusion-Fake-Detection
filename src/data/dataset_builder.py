import os
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def load_raw_parquet_data(dataset_name, raw_dir='data/raw', interim_dir='data/interim', encoder='sbert'):
    dataset_dir = os.path.normpath(os.path.join(raw_dir, dataset_name))
    labels_path = os.path.join(dataset_dir, 'graph_labels.npy')
    labels = np.load(labels_path)
    
    node_graph_id_path = os.path.join(dataset_dir, 'node_graph_id.npy')
    node_graph_id = np.load(node_graph_id_path)
    
    edges_path = os.path.join(dataset_dir, 'A.txt')
    if os.path.getsize(edges_path) > 0:
        edges_data = np.loadtxt(edges_path, delimiter=',', dtype=int)
        if edges_data.ndim == 1:
            edges_data = edges_data.reshape(1, -1)
    else:
        edges_data = np.array([])
    
    feature_file = f'new_{encoder}_feature.npz'
    features_path = os.path.normpath(os.path.join(interim_dir, f'embeddings_{encoder}', dataset_name, feature_file))
    
    if features_path.endswith('.npz'):
        node_features = sp.load_npz(features_path).toarray()
    elif features_path.endswith('.npy'):
        node_features = np.load(features_path)
    else:
        raise ValueError(f"Unsupported feature file format: {features_path}")
        
    return {
        'labels': labels,
        'edges_data': edges_data,
        'node_features': node_features,
        'node_graph_id': node_graph_id, 
        'dataset_dir': dataset_dir
    }


def create_pyg_graphs_from_raw(data_dict):
    labels = data_dict['labels']
    edges_data = data_dict['edges_data']
    node_features = data_dict['node_features']
    node_graph_id = data_dict['node_graph_id']
    
    num_graphs = len(labels)
    pyg_graphs = []
    
    graph_edges = {g_id: [] for g_id in range(num_graphs)}
    if edges_data.size > 0:
        for row in edges_data:
            src, dst = row[0], row[1]
            g_id = node_graph_id[src]
            graph_edges[g_id].append((src, dst))
            
    for g_id in tqdm(range(num_graphs), desc="Building PyG Graphs"):
        y = torch.tensor([labels[g_id]], dtype=torch.long)
        
        global_nodes = np.where(node_graph_id == g_id)[0]
        num_nodes = len(global_nodes)
        node_mapping = {global_id: local_id for local_id, global_id in enumerate(global_nodes)}
        
        if len(graph_edges[g_id]) == 0:
            struc_x = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            # ĐÃ THÊM: Tạo cạnh đảo ngược (Bottom-Up) và root_index cho BiGCN
            BU_edge_index = torch.zeros((2, 0), dtype=torch.long)
            root_index = torch.tensor([0], dtype=torch.long)
            
            local_text_features = np.zeros((num_nodes, node_features.shape[1]))
            for global_id, local_id in node_mapping.items():
                if global_id < len(node_features):
                    local_text_features[local_id] = node_features[global_id]
            x = torch.tensor(local_text_features, dtype=torch.float32)
            
            # ĐÃ ĐỔI: Gán biến x = vector chữ, struc_x = cấu trúc 3D
            data = Data(x=x, text_x=x, struc_x=struc_x, edge_index=edge_index, 
                        BU_edge_index=BU_edge_index, root_index=root_index, y=y, graph_id=g_id)
            pyg_graphs.append(data)
            continue
            
        edges = np.array(graph_edges[g_id])
        src_nodes = edges[:, 0]
        dst_nodes = edges[:, 1]
        
        local_src = [node_mapping[src] for src in src_nodes]
        local_dst = [node_mapping[dst] for dst in dst_nodes]
        
        edge_index = torch.tensor([local_src, local_dst], dtype=torch.long)
        
        # ĐÃ THÊM: Tạo cạnh đảo ngược (Bottom-Up) cho BiGCN
        BU_edge_index = torch.tensor([local_dst, local_src], dtype=torch.long)
        
        in_degree = torch.zeros(num_nodes, dtype=torch.float32)
        out_degree = torch.zeros(num_nodes, dtype=torch.float32)
        root_flag = torch.zeros(num_nodes, dtype=torch.float32)
        
        for src, dst in zip(local_src, local_dst):
            out_degree[src] += 1
            in_degree[dst] += 1
            
        root_idx = (in_degree == 0).nonzero(as_tuple=True)[0]
        if len(root_idx) > 0:
            root_flag[root_idx[0]] = 1.0
            root_index = torch.tensor([root_idx[0]], dtype=torch.long)
        elif num_nodes > 0:
            root_flag[0] = 1.0
            root_index = torch.tensor([0], dtype=torch.long)
        else:
            root_index = torch.tensor([0], dtype=torch.long)
            
        struc_x = torch.stack([in_degree, out_degree, root_flag], dim=1)
        
        local_text_features = np.zeros((num_nodes, node_features.shape[1]))
        for global_id, local_id in node_mapping.items():
            if global_id < len(node_features):
                local_text_features[local_id] = node_features[global_id]
                
        x = torch.tensor(local_text_features, dtype=torch.float32)
        
        # ĐÃ ĐỔI: Bơm đầy đủ tham số để nuôi Baseline
        data = Data(x=x, text_x=x, struc_x=struc_x, edge_index=edge_index, 
                    BU_edge_index=BU_edge_index, root_index=root_index, y=y, graph_id=g_id)
        pyg_graphs.append(data)
        
    return pyg_graphs


def split_datasets_strict(graphs, dataset_name='politifact', raw_dir='data/raw'):
    dataset_dir = os.path.join(raw_dir, dataset_name)
    try:
        train_idx = np.load(os.path.join(dataset_dir, 'train_idx.npy'))
        val_idx = np.load(os.path.join(dataset_dir, 'val_idx.npy'))
        test_idx = np.load(os.path.join(dataset_dir, 'test_idx.npy'))
    except FileNotFoundError:
        raise Exception(f"Missing index files in {dataset_dir}.")

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    train_graphs, val_graphs, test_graphs = [], [], []
    
    for graph in graphs:
        g_id = graph.graph_id.item() if isinstance(graph.graph_id, torch.Tensor) else graph.graph_id
        if g_id in train_set:
            train_graphs.append(graph)
        elif g_id in val_set:
            val_graphs.append(graph)
        elif g_id in test_set:
            test_graphs.append(graph)
            
    return train_graphs, val_graphs, test_graphs