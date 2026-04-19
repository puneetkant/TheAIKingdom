"""
Working Example: GNN Tools and Libraries
Covers PyTorch Geometric, DGL, OGB, NetworkX, and practical
patterns for building GNN pipelines.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_gnn_tools")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Library overview -------------------------------------------------------
def library_overview():
    print("=== GNN Libraries Overview ===")
    print()
    libs = [
        ("PyG (PyTorch Geometric)", "Fey & Lenssen 2019; most popular; 14k+ stars; rich dataset API"),
        ("DGL (Deep Graph Library)", "AWS + others; TF/PyTorch backends; fast; production-ready"),
        ("NetworkX",                 "Pure Python; analysis & visualisation; not for training"),
        ("OGB (Open Graph Benchmark)","Standardised datasets + evaluation; leaderboards"),
        ("GraphX (Spark)",           "Distributed graph processing; not DL-focused"),
        ("TensorFlow GNN",           "Google; TF 2.x; heterogeneous graphs"),
        ("Jraph (JAX)",              "Google DeepMind; functional style; research"),
    ]
    for l, d in libs:
        print(f"  {l:<30} {d}")


# -- 2. PyG patterns -----------------------------------------------------------
def pyg_patterns():
    print("\n=== PyG (PyTorch Geometric) Patterns ===")
    print()
    print("  Core data structure: torch_geometric.data.Data")
    print()
    print("  Code pattern:")
    code = '''
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# Build a graph
edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)  # COO format
x = torch.randn(3, 16)  # 3 nodes, 16 features
data = Data(x=x, edge_index=edge_index)

# GCN model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Mini-batch loader
from torch_geometric.loader import NeighborLoader
loader = NeighborLoader(dataset[0],
                        num_neighbors=[25, 10],   # fan-out per layer
                        batch_size=1024)
'''
    print(code)
    print("  Key modules:")
    modules = [
        ("GCNConv",          "Graph Convolutional Layer"),
        ("SAGEConv",         "GraphSAGE layer"),
        ("GATConv",          "Graph Attention layer"),
        ("GINConv",          "Graph Isomorphism layer"),
        ("MessagePassing",   "Base class for custom message passing"),
        ("DataLoader",       "Mini-batch loading for graph-level tasks"),
        ("NeighborLoader",   "Node-level mini-batch with neighbour sampling"),
        ("ClusterLoader",    "Cluster-GCN style subgraph loader"),
    ]
    for m, d in modules:
        print(f"  {m:<20} {d}")


# -- 3. OGB benchmark pattern --------------------------------------------------
def ogb_patterns():
    print("\n=== OGB (Open Graph Benchmark) ===")
    print()
    print("  Standardised datasets with official evaluators")
    print()
    datasets = [
        ("ogbn-arxiv",   "node", "arXiv citation; 169k nodes; 1.2M edges"),
        ("ogbn-products","node", "Amazon co-purchase; 2.4M nodes"),
        ("ogbl-collab",  "link", "Author collaboration; predict future links"),
        ("ogbl-citation","link", "Paper citations; large-scale"),
        ("ogbg-molhiv",  "graph","HIV activity; 41k molecular graphs"),
        ("ogbg-molpcba", "graph","128 bioactivity assays; 440k graphs"),
    ]
    print(f"  {'Dataset':<16} {'Type':<6} {'Description'}")
    for d, t, desc in datasets:
        print(f"  {d:<16} {t:<6} {desc}")
    print()
    print("  Usage pattern:")
    code = '''
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

dataset  = PygNodePropPredDataset(name='ogbn-arxiv')
data     = dataset[0]
split_idx = dataset.get_idx_split()  # train/valid/test

evaluator = Evaluator(name='ogbn-arxiv')
result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
print(result['acc'])
'''
    print(code)


# -- 4. NetworkX for graph analysis --------------------------------------------
def networkx_patterns():
    print("=== NetworkX Graph Analysis ===")
    try:
        import networkx as nx
        G = nx.karate_club_graph()
        print(f"  Karate club graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  Average clustering: {nx.average_clustering(G):.4f}")
        print(f"  Average shortest path: {nx.average_shortest_path_length(G):.4f}")
        print(f"  Is connected: {nx.is_connected(G)}")
        # Centrality
        deg_c = nx.degree_centrality(G)
        top3 = sorted(deg_c.items(), key=lambda x: -x[1])[:3]
        print(f"  Top-3 degree central nodes: {[(n, round(c,3)) for n,c in top3]}")
    except ImportError:
        print("  NetworkX not available. Code pattern:")
        code = '''
import networkx as nx
G = nx.karate_club_graph()
nx.betweenness_centrality(G)
nx.community.louvain_communities(G)
communities = nx.community.greedy_modularity_communities(G)
'''
        print(code)
    print()
    print("  Common operations:")
    ops = [
        ("nx.shortest_path(G, s, t)",         "BFS/Dijkstra path"),
        ("nx.betweenness_centrality(G)",       "Node betweenness"),
        ("nx.pagerank(G, alpha=0.85)",         "PageRank scores"),
        ("nx.is_isomorphic(G1, G2)",           "Graph isomorphism check"),
        ("nx.community.louvain_communities(G)","Louvain community detection"),
        ("nx.to_scipy_sparse_array(G)",        "Convert to sparse matrix"),
    ]
    for op, d in ops:
        print(f"  {op:<42} {d}")


if __name__ == "__main__":
    library_overview()
    pyg_patterns()
    ogb_patterns()
    networkx_patterns()
