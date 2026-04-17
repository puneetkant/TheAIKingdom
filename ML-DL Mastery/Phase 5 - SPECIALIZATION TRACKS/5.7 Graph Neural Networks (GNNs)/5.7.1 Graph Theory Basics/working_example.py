"""
Working Example: Graph Theory Basics for GNNs
Covers graph representations, properties, common graph types,
and basic graph algorithms needed for GNN understanding.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_graph_theory")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Graph representations ──────────────────────────────────────────────────
def graph_representations():
    print("=== Graph Theory Basics ===")
    print()
    print("  Graph G = (V, E)")
    print("    V: set of nodes (vertices)  |V| = N")
    print("    E: set of edges             |E| = M")
    print()

    # Adjacency matrix
    N = 5
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]
    A = np.zeros((N, N), dtype=int)
    for u, v in edges:
        A[u, v] = A[v, u] = 1

    print("  Adjacency matrix (undirected):")
    for row in A:
        print("   ", row)

    # Degree
    D = np.diag(A.sum(axis=1))
    deg = A.sum(axis=1)
    print()
    print(f"  Node degrees: {deg}")
    print(f"  Max degree: {deg.max()}  Min: {deg.min()}  Avg: {deg.mean():.2f}")

    # Edge list (COO format for sparse graphs)
    src = [u for u,v in edges]
    dst = [v for u,v in edges]
    print(f"\n  Edge list (COO): src={src}, dst={dst}")

    # Adjacency list
    adj = {i: [] for i in range(N)}
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    print("  Adjacency list:")
    for node, nbrs in adj.items():
        print(f"    {node}: {sorted(nbrs)}")


# ── 2. Graph properties ───────────────────────────────────────────────────────
def graph_properties():
    print("\n=== Key Graph Properties ===")
    N = 5
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]
    A = np.zeros((N, N), dtype=int)
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    deg = A.sum(axis=1)

    # Diameter: BFS from each node
    def bfs_dist(A, src):
        dist = np.full(N, -1); dist[src] = 0
        queue = [src]
        while queue:
            node  = queue.pop(0)
            for nbr in np.where(A[node])[0]:
                if dist[nbr] == -1:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        return dist

    all_pairs = np.array([bfs_dist(A, s) for s in range(N)])
    diameter  = all_pairs.max()
    avg_path  = all_pairs[all_pairs > 0].mean()

    # Clustering coefficient
    cc = []
    for i in range(N):
        nbrs = np.where(A[i])[0]
        ki   = len(nbrs)
        if ki < 2: cc.append(0.0); continue
        triangles = sum(A[u, v] for idx, u in enumerate(nbrs) for v in nbrs[idx+1:])
        cc.append(2 * triangles / (ki * (ki-1)))

    # Global clustering coefficient (transitivity)
    triangles_global = sum(A[i,j]*A[j,k]*A[k,i] for i in range(N)
                           for j in range(N) for k in range(N)) // 6
    triples = sum(d*(d-1)//2 for d in deg)

    print(f"  N={N}  M={len(edges)}")
    print(f"  Density: {2*len(edges)/(N*(N-1)):.3f}")
    print(f"  Diameter: {diameter}")
    print(f"  Avg shortest path: {avg_path:.3f}")
    print(f"  Per-node clustering coefficients: {np.round(cc, 3)}")
    print(f"  Global clustering coefficient: {cc_g := 3*triangles_global/(triples+1e-9):.3f}")


# ── 3. Graph types ────────────────────────────────────────────────────────────
def graph_types():
    print("\n=== Graph Types ===")
    types = [
        ("Undirected",    "Edges have no direction; A = A^T"),
        ("Directed",      "Edges have direction (digraph); A ≠ A^T"),
        ("Weighted",      "Each edge has a weight w_{ij}"),
        ("Bipartite",     "V = U ∪ V; edges only between U and V (e.g. user-item)"),
        ("Heterogeneous", "Multiple node/edge types (e.g. user, item, category)"),
        ("Temporal",      "Edge timestamps; dynamic graph"),
        ("Hypergraph",    "Edges can connect >2 nodes"),
        ("Knowledge graph","(subject, relation, object) triples"),
    ]
    for t, d in types:
        print(f"  {t:<18} {d}")

    print()
    print("  Special structures:")
    specials = [
        ("Tree",       "N-1 edges; connected; acyclic"),
        ("DAG",        "Directed acyclic graph; used in computation graphs"),
        ("Complete",   "All N(N-1)/2 edges present"),
        ("Star",       "Central node connected to all others"),
        ("Grid",       "2D lattice; image as graph"),
    ]
    for s, d in specials:
        print(f"  {s:<12} {d}")


# ── 4. Spectral graph theory ──────────────────────────────────────────────────
def spectral_graph_theory():
    print("\n=== Spectral Graph Theory ===")
    N = 5
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]
    A = np.zeros((N, N))
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0
    D = np.diag(A.sum(axis=1))

    # Unnormalised Laplacian
    L = D - A
    print("  Graph Laplacian L = D - A:")
    for row in L:
        print("   ", row.astype(int))

    # Symmetric normalised Laplacian (used in GCN)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(A.sum(axis=1)) + 1e-9))
    L_sym = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    print()
    print("  Normalised Laplacian L_sym = I - D^{-1/2} A D^{-1/2}")

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(L)
    print()
    print(f"  Eigenvalues of L: {np.round(eigvals, 4)}")
    print(f"  λ_1=0 always (multiplicity = number of connected components)")
    print(f"  λ_2 (Fiedler value): {eigvals[1]:.4f}  (larger → better connected)")
    print()
    print("  Relationship to GCN:")
    print("    GCN propagation: Ã·H·W  where Ã = D̃^{-1/2}(A+I)D̃^{-1/2}")
    print("    Adds self-loops (A+I) for stability; renormalisation trick")


if __name__ == "__main__":
    graph_representations()
    graph_properties()
    graph_types()
    spectral_graph_theory()
