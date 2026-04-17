"""
Working Example: GNN Applications
Covers drug discovery, traffic forecasting, social networks,
knowledge graph completion, and molecular property prediction.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_gnn_apps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# ── 1. Molecular property prediction ─────────────────────────────────────────
def molecular_property():
    print("=== Molecular Property Prediction ===")
    print("  Molecules as graphs: atoms = nodes, bonds = edges")
    print()
    print("  Node features (atom features):")
    print("    Atomic number, degree, formal charge, chirality,")
    print("    aromaticity, hydrogen count, ring membership")
    print()
    print("  Edge features (bond features):")
    print("    Bond type (single/double/triple/aromatic), conjugation,")
    print("    ring membership, stereochemistry")
    print()

    # Toy molecule: ethanol C2H5OH simplified
    # Atoms: [C, C, O]  Bonds: C-C, C-O
    atom_features = np.array([
        [6, 1, 0, 0],   # C: atomic_num, degree, formal_charge, aromatic
        [6, 2, 0, 0],   # C
        [8, 1, 0, 0],   # O
    ])
    A = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
    N, F_in = atom_features.shape
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.1, (F_in, 8))

    A_hat  = A + np.eye(N)
    D_inv  = np.diag(1.0 / (A_hat.sum(axis=1)**0.5 + 1e-9))
    H      = relu((D_inv @ A_hat @ D_inv) @ atom_features @ W)
    h_mol  = H.mean(axis=0)   # global mean pool
    print(f"  Ethanol (3 atoms): atom feats {atom_features.shape}")
    print(f"  GCN output per atom: {H.shape}")
    print(f"  Molecule embedding: {h_mol.round(4)}")
    print()
    print("  Tasks:")
    tasks = [
        ("Solubility",    "regression on logS"),
        ("Toxicity",      "binary; Tox21, MoleculeNet"),
        ("ADMET",         "pharmacokinetics; absorption, distribution, ..."),
        ("Drug-drug",     "interaction prediction (link prediction on mol graph)"),
        ("Protein binding","predict binding affinity; graph pair matching"),
    ]
    for t, d in tasks:
        print(f"  {t:<16} {d}")
    print()
    print("  Key models: MPNN (DeepMind), AttentiveFP, DimeNet, GemNet,")
    print("              Uni-Mol (3D pre-training), GPS, Graphormer")


# ── 2. Traffic forecasting ────────────────────────────────────────────────────
def traffic_forecasting():
    print("\n=== Traffic Speed Forecasting (Spatio-Temporal GNN) ===")
    print("  Road network as graph; nodes = sensors, edges = road connections")
    print()
    print("  Benchmark datasets:")
    datasets = [
        ("METR-LA",   "207 sensors, LA highways, 5-min intervals"),
        ("PEMS-BAY",  "325 sensors, Bay Area, 6-month traffic speed"),
        ("PEMS-D4/7/8","larger; 307-170 sensors; STSGCN/STGCN"),
    ]
    for d, desc in datasets:
        print(f"  {d:<12} {desc}")
    print()
    print("  Models:")
    models = [
        ("DCRNN",       2018, "Diffusion Conv RNN; encoder-decoder"),
        ("STGCN",       2018, "Spatio-temporal graph conv blocks"),
        ("Graph WaveNet",2019,"Adaptive adjacency + dilated conv"),
        ("ASTGCN",      2019,"Attention-based spatial-temporal"),
        ("STGODE",      2021,"Neural ODE + graph; continuous dynamics"),
        ("STAEformer",  2023,"Adaptive embedding + Transformer"),
    ]
    print(f"  {'Model':<16} {'Year'} {'Notes'}")
    for m, y, d in models:
        print(f"  {m:<16} {y}  {d}")

    # Simulate simple spatio-temporal prediction
    rng = np.random.default_rng(0)
    N_sensors = 5; T_in = 12; T_out = 3; F = 1
    speed_data = rng.normal(60, 10, (T_in, N_sensors, F))

    # Simple MEAN baseline: last observation
    baseline = speed_data[-1]   # (N, F)
    print()
    print(f"  Simulated: {N_sensors} sensors, input T={T_in}, forecast T={T_out}")
    print(f"  Baseline (last obs) predictions: {baseline.squeeze().round(1)}")


# ── 3. Knowledge graph completion ─────────────────────────────────────────────
def knowledge_graph_completion():
    print("\n=== Knowledge Graph Completion ===")
    print("  KG: (subject, relation, object) triples")
    print("  Task: predict missing triples — link prediction on KG")
    print()
    print("  Embedding models:")
    kge_models = [
        ("TransE",    "h + r ≈ t  (translation)"),
        ("DistMult",  "score = h^T diag(r) t  (bilinear)"),
        ("ComplEx",   "complex-valued embeddings; handles antisymmetry"),
        ("RotatE",    "relation as rotation in complex space"),
        ("ERNIE",     "entity + relation text init from BERT"),
        ("CompGCN",   "GCN over entity-relation composition"),
    ]
    for m, d in kge_models:
        print(f"  {m:<12} {d}")
    print()

    # TransE toy demo
    rng = np.random.default_rng(0)
    n_entities = 6; n_relations = 2; D = 4
    E = rng.normal(0, 1, (n_entities, D))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    R = rng.normal(0, 1, (n_relations, D))

    triples = [(0, 0, 1), (1, 0, 2), (0, 1, 3)]   # (h, r, t)

    def transe_score(h_idx, r_idx, t_idx):
        return -np.linalg.norm(E[h_idx] + R[r_idx] - E[t_idx])

    print("  TransE scores (higher/less-negative = better):")
    for h, r, t in triples:
        s = transe_score(h, r, t)
        print(f"    ({h}, r{r}, {t}): {s:.4f}")


# ── 4. Social networks and fraud detection ────────────────────────────────────
def social_fraud():
    print("\n=== Social Network Applications ===")
    print()
    print("  Community detection:")
    print("    Detect groups of densely connected nodes")
    print("    GNN as soft community assignment; DGI, GraphCL, GRACE")
    print()
    print("  Fraud detection:")
    print("    Fraudsters form dense suspicious subgraphs")
    print("    Node-level classification with neighbourhood features")
    print("    Challenges: class imbalance, camouflage, dynamic graph")
    print("    Models: CARE-GNN, PC-GNN, FRAUDRE")
    print()
    print("  Influence maximisation:")
    print("    Find k seed nodes to maximise information spread")
    print("    DeepIM: GNN approximation of influence spread")
    print()
    print("  Fake news / misinformation:")
    print("    News article + social context as heterogeneous graph")
    print("    SAFE, FANG, GCAN: propagation tree as graph")


if __name__ == "__main__":
    molecular_property()
    traffic_forecasting()
    knowledge_graph_completion()
    social_fraud()
