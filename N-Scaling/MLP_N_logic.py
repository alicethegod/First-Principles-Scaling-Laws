# -*- coding: utf-8 -*-
"""
Core Experiment Logic Module (MLP N-Scaling Experiment Version)

This file is designed for the Scaling Law experiment concerning model parameter count (N).
Core Changes:
1.  Added the `MLP_N` class, whose hidden layer widths can be scaled via the `hidden_sizes` parameter.
2.  The `run_training_task_N_scaling` function now accepts `hidden_sizes` as an argument and uses a fixed dataset.
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

class MLP_N(nn.Module):
    def __init__(self, hidden_sizes=(40, 20)):
        super(MLP_N, self).__init__()
        # Directly get hidden layer sizes from the tuple
        hidden1, hidden2 = hidden_sizes
        
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 10)
        )
    def forward(self, x):
        return self.layers(x.view(--1, 28 * 28))
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --- Theory Analyzer ---
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
        node_counter = 0; layer_map = {}
        in_features = self.model.layers[0].in_features
        layer_map[0] = list(range(node_counter, node_counter + in_features))
        for i in range(in_features):
            G.add_node(node_counter, layer=0); node_counter += 1
        graph_layer_idx = 1
        for l in self.model.layers:
            if isinstance(l, nn.Linear):
                layer_map[graph_layer_idx] = list(range(node_counter, node_counter + l.out_features))
                for i in range(l.out_features):
                    G.add_node(node_counter, layer=graph_layer_idx); node_counter += 1
                weights = torch.abs(l.weight.data.t()); probs = torch.softmax(weights, dim=1)
                for u_local_idx, u_global_idx in enumerate(layer_map[graph_layer_idx - 1]):
                    for v_local_idx, v_global_idx in enumerate(layer_map[graph_layer_idx]):
                        prob = probs[u_local_idx, v_local_idx].item()
                        if prob > 1e-9: G.add_edge(u_global_idx, v_global_idx, cost=1.0 - np.log(prob))
                graph_layer_idx += 1
        self.grounding_nodes = set(layer_map[graph_layer_idx - 1]); return G

    def _get_hidden_nodes(self):
        max_layer_idx = max((data['layer'] for _, data in self.graph.nodes(data=True)), default=0)
        return [node for node, data in self.graph.nodes(data=True) if data['layer'] not in [0, max_layer_idx]]

    def find_all_paths_dfs(self, start, targets):
        memo_key = (start, tuple(sorted(list(targets))))
        if memo_key in self.memoized_paths: return self.memoized_paths[memo_key]
        paths, stack = [], [(start, [start], 0)]
        while stack:
            curr, path, cost = stack.pop()
            if curr in targets: paths.append({'path': path, 'cost': cost}); continue
            if len(path) > 10: continue # Path depth limit
            for neighbor in self.graph.neighbors(curr):
                edge_cost = self.graph.get_edge_data(curr, neighbor, {}).get('cost', float('inf'))
                if neighbor not in path: stack.append((neighbor, path + [neighbor], cost + edge_cost))
        self.memoized_paths[memo_key] = paths; return paths

    def calculate_metrics_for_node(self, node):
        paths = self.find_all_paths_dfs(node, self.grounding_nodes)
        if not paths: return float('inf'), 0.0
        costs = np.array([p['cost'] for p in paths])
        conductances = 1.0 / costs
        htse = 1.0 / np.sum(conductances) if np.sum(conductances) > 0 else float('inf')
        importances = np.exp(-1.0 * costs)
        probabilities = importances / np.sum(importances) if np.sum(importances) > 0 else np.zeros_like(importances)
        hsie = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return htse, hsie

    def analyze_model_structure(self, analysis_sample_size):
        htse_vals, hsie_vals = [], []
        if not self.hidden_nodes: return 0, 0
        sample_size = min(analysis_sample_size, len(self.hidden_nodes))
        sampled_nodes = np.random.choice(self.hidden_nodes, size=sample_size, replace=False)
        for node in sampled_nodes:
            htse, hsie = self.calculate_metrics_for_node(node)
            if np.isfinite(htse) and np.isfinite(hsie): htse_vals.append(htse); hsie_vals.append(hsie)
        return np.mean(htse_vals) if htse_vals else 0, np.mean(hsie_vals) if hsie_vals else 0

def run_training_task_N_scaling(args):
    seed, hidden_sizes, config, device_id = args

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.ToTensor()
    full_train_dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
    full_test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)
    
    dataset_size = config.get("dataset_size", len(full_train_dataset))
    train_indices = torch.randperm(len(full_train_dataset))[:dataset_size]
    train_subset = Subset(full_train_dataset, train_indices)

    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(full_test_dataset, batch_size=config['batch_size'])

    # Directly use the hidden_sizes tuple to create the model
    model = MLP_N(hidden_sizes=hidden_sizes).to(device)
    num_params = model.get_num_params()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['epochs']):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    final_test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            final_test_loss += loss.item() * data.size(0)
    final_test_loss /= len(test_loader.dataset)

    analyzer = TheoryAnalyzer(model)
    final_htse, final_hsie = analyzer.analyze_model_structure(config['analysis_sample_size'])

    return {
        'seed': seed,
        'hidden_sizes': f"({hidden_sizes[0]}, {hidden_sizes[1]})",
        'num_params_N': num_params,
        'final_test_loss': final_test_loss,
        'final_htse': final_htse,
        'final_hsie': final_hsie
    }
