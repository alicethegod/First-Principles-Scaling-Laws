# -*- coding: utf-8 -*-
"""
Core Experiment Logic Module (Pre-trained CNN + MLP Head Version)

This file encapsulates the core, reusable components of the experiment.
This version has been modified to implement the experimental setup of 
"freezing the feature extractor and training the fully connected head."
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import networkx as nx
import copy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Model Definition ---

# MLP Head for classification on top of fixed features
class MLP_Head(nn.Module):
    def __init__(self, input_features, num_classes=10):
        super(MLP_Head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- Theory Analyzer ---
# This is the original analyzer, adapted only for fully connected models
class TheoryAnalyzer:
    def __init__(self, model):
        model_copy = copy.deepcopy(model)
        model_copy.eval()
        self.model = model_copy.to('cpu')
        self.graph = self._build_graph()
        self.hidden_nodes = self._get_hidden_nodes()
        self.memoized_paths = {}

    def _build_graph(self):
        G = nx.DiGraph()
        node_counter = 0
        layer_map = {}
        in_features = self.model.layers[0].in_features
        layer_map[0] = list(range(node_counter, node_counter + in_features))
        for i in range(in_features):
            G.add_node(node_counter, layer=0)
            node_counter += 1
            
        graph_layer_idx = 1
        for l in self.model.layers:
            if isinstance(l, nn.Linear):
                layer_map[graph_layer_idx] = list(range(node_counter, node_counter + l.out_features))
                for i in range(l.out_features):
                    G.add_node(node_counter, layer=graph_layer_idx)
                    node_counter += 1
                
                weights = torch.abs(l.weight.data.t())
                probs = torch.softmax(weights, dim=1)
                
                for u_local_idx, u_global_idx in enumerate(layer_map[graph_layer_idx - 1]):
                    for v_local_idx, v_global_idx in enumerate(layer_map[graph_layer_idx]):
                        prob = probs[u_local_idx, v_local_idx].item()
                        if prob > 1e-9:
                            G.add_edge(u_global_idx, v_global_idx, cost=1.0 - np.log(prob))
                graph_layer_idx += 1
                
        self.grounding_nodes = set(layer_map[graph_layer_idx - 1])
        return G

    def _get_hidden_nodes(self):
        max_layer_idx = max((data['layer'] for _, data in self.graph.nodes(data=True)), default=0)
        return [node for node, data in self.graph.nodes(data=True) if data['layer'] not in [0, max_layer_idx]]

    def find_all_paths_dfs(self, start, targets):
        memo_key = (start, tuple(sorted(list(targets))))
        if memo_key in self.memoized_paths:
            return self.memoized_paths[memo_key]
            
        paths = []
        stack = [(start, [start], 0)]
        
        while stack:
            curr, path, cost = stack.pop()
            
            if curr in targets:
                paths.append({'path': path, 'cost': cost})
                continue
            
            if len(path) > 10:  # Path depth limit
                continue
                
            for neighbor in self.graph.neighbors(curr):
                edge_cost = self.graph.get_edge_data(curr, neighbor, {}).get('cost', float('inf'))
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor], cost + edge_cost))
                    
        self.memoized_paths[memo_key] = paths
        return paths

    def calculate_metrics_for_node(self, node):
        paths = self.find_all_paths_dfs(node, self.grounding_nodes)
        if not paths:
            return float('inf'), 0.0
            
        costs = np.array([p['cost'] for p in paths])
        conductances = 1.0 / costs
        htse = 1.0 / np.sum(conductances) if np.sum(conductances) > 0 else float('inf')
        
        importances = np.exp(-1.0 * costs)
        probabilities = importances / np.sum(importances) if np.sum(importances) > 0 else np.zeros_like(importances)
        hsie = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        
        return htse, hsie

    def analyze_model_structure(self, analysis_sample_size):
        htse_vals, hsie_vals = [], []
        if not self.hidden_nodes:
            return 0, 0
            
        sample_size = min(analysis_sample_size, len(self.hidden_nodes))
        sampled_nodes = np.random.choice(self.hidden_nodes, size=sample_size, replace=False)
        
        for node in sampled_nodes:
            htse, hsie = self.calculate_metrics_for_node(node)
            if np.isfinite(htse) and np.isfinite(hsie):
                htse_vals.append(htse)
                hsie_vals.append(hsie)
                
        return np.mean(htse_vals) if htse_vals else 0, np.mean(hsie_vals) if hsie_vals else 0

# --- Parallelizable Task Function ---
def run_training_task(args):
    """
    This is the core function that each worker process will execute in parallel.
    It encapsulates a complete training and analysis cycle and accepts a fixed seed.
    """
    seed, d_size, config, device_id = args

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

    # --- CRITICAL CHANGE: Set a fixed seed at the "start" of each task ---
    # This ensures model initialization and the first data sampling are fixed.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Each process loads data independently
    transform = transforms.ToTensor()
    try:
        full_train_dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    except RuntimeError:
        print("Please download the FashionMNIST dataset first by running the notebook cell with 'download=True' once.")
        return None

    # The random sampling of the data subset is also controlled by the seed set above
    indices = torch.randperm(len(full_train_dataset))[:d_size]
    train_subset = Subset(full_train_dataset, indices)
    
    # The randomness of the DataLoader will introduce variations during training, which is the "real noise" we want to keep.
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # --- Implement the new scheme: Load pre-trained CNN, freeze parameters, and build MLP Head ---
    # 1. Load a simple CNN as a feature extractor
    cnn_feature_extractor = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten()
    ).to(device)

    # 2. Freeze the parameters of the feature extractor
    for param in cnn_feature_extractor.parameters():
        param.requires_grad = False

    # 3. Build the trainable MLP Head
    # The input features need to be calculated based on the CNN output shape
    mlp_head = MLP_Head(input_features=32*7*7, num_classes=10).to(device)

    # 4. Define the optimizer to target only the parameters of the MLP Head
    optimizer = optim.Adam(mlp_head.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        mlp_head.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass: data first goes through the frozen CNN, then the trainable MLP Head
            with torch.no_grad():
                features = cnn_feature_extractor(data)
            output = mlp_head(features)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    mlp_head.eval()
    final_test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features = cnn_feature_extractor(data)
            output = mlp_head(features)
            loss = criterion(output, target)
            final_test_loss += loss.item() * data.size(0)
    final_test_loss /= len(test_loader.dataset)

    # The TheoryAnalyzer only analyzes the MLP Head
    analyzer = TheoryAnalyzer(mlp_head)
    final_htse, final_hsie = analyzer.analyze_model_structure(config['analysis_sample_size'])

    return {
        'seed': seed,
        'data_size_d': d_size,
        'final_test_loss': final_test_loss,
        'final_htse': final_htse,
        'final_hsie': final_hsie
    }
