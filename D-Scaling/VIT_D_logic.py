# -*- coding: utf-8 -*-
"""
Core Experiment Logic Module (Vision Transformer Structure Analysis Version)

This file is designed for the D-Scaling experiment with a Vision Transformer,
and it introduces the H'tse/H'sie analysis.

Core Changes:
1.  The TheoryAnalyzer is applied to the MLP_Head part of the VisionTransformer.
2.  The run_training_task function now returns the complete analysis results.
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

# This is a standard MLP that the TheoryAnalyzer can understand.
class MLP_Head(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

class VisionTransformer(nn.Module):
    def __init__(self, *, image_size=28, patch_size=4, num_classes=10, dim=64, depth=2, heads=4, mlp_dim=128):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        # We use a simple Conv2d to create patches
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # The head is now a sequence of LayerNorm and our analyzable MLP_Head
        self.mlp_head_container = nn.Sequential(
            nn.LayerNorm(dim),
            MLP_Head(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = x.permute(0, 2, 1)

        b, n, _ = x.shape

        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer_encoder(x)

        x = x[:, 0]
        return self.mlp_head_container(x)

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

# --- Training Task Function (ViT Analysis Version) ---
def run_training_task(args):
    seed, d_size, config, device_id = args
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = transforms.ToTensor()
    full_train_dataset = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)

    indices = torch.randperm(len(full_train_dataset))[:d_size]
    train_subset = Subset(full_train_dataset, indices)

    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    model = VisionTransformer().to(device)
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
            final_test_loss += criterion(output, target).item() * data.size(0)
    final_test_loss /= len(test_loader.dataset)

    # --- [CRITICAL] Perform analysis ONLY on the MLP Head ---
    # The MLP_Head is the second element in the sequential container
    mlp_head_to_analyze = model.mlp_head_container[1]
    analyzer = TheoryAnalyzer(mlp_head_to_analyze)
    final_htse, final_hsie = analyzer.analyze_model_structure(config['analysis_sample_size'])
    # --------------------------------------------------------

    return {
        'seed': seed,
        'data_size_d': d_size,
        'final_test_loss': final_test_loss,
        'final_htse': final_htse,
        'final_hsie': final_hsie
    }
