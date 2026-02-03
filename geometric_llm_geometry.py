# geometric_llm_geometry.py
"""
Geometric Structure in Language Model Representations

Illustrates three interconnected geometric concepts:
• Hyperbolic embeddings of semantic hierarchies
• Curvature-aware analysis of attention-like graphs
• Alignment of distributed representations via optimal transport

All operations are performed in the Poincaré ball model.

Requirements
------------
pip install torch geoopt POT networkx matplotlib numpy
"""

import torch
import geoopt
import numpy as np
import networkx as nx
import ot
import matplotlib.pyplot as plt
from matplotlib import cm

# ────────────────────────────────────────────────
# Configuration & Reproducibility
# ────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}\n")

# ────────────────────────────────────────────────
# Semantic hierarchy (WordNet-inspired subtree)
# ────────────────────────────────────────────────
hierarchy = [
    (0,  "entity",         []),
    (1,  "living_thing",   [0]),
    (2,  "non_living",     [0]),
    (3,  "animal",         [1]),
    (4,  "plant",          [1]),
    (5,  "mammal",         [3]),
    (6,  "bird",           [3]),
    (7,  "fish",           [3]),
    (8,  "tree",           [4]),
    (9,  "flower",         [4]),
    (10, "dog",            [5]),
    (11, "cat",            [5]),
    (12, "eagle",          [6]),
    (13, "sparrow",        [6]),
    (14, "salmon",         [7]),
    (15, "oak",            [8]),
    (16, "rose",           [9]),
]

tokens = [node[1] for node in hierarchy]
n_tokens = len(tokens)
parent_map = {child: parent for _, _, parents in hierarchy for child in parents}

# ────────────────────────────────────────────────
# 1. Hyperbolic embeddings (Poincaré ball)
# ────────────────────────────────────────────────
manifold = geoopt.PoincareBall(c=1.0)

def radial_tree_layout(root=0, max_radius=1.0):
    coords = np.zeros((n_tokens, 2))
    def recurse(node_id, depth=0, angle=0.0, angle_span=2 * np.pi):
        r = depth * max_radius / 5.0
        coords[node_id] = r * np.array([np.cos(angle), np.sin(angle)])
        children = [c for c, p in parent_map.items() if p == node_id]
        if children:
            sub_span = angle_span / len(children)
            for i, child in enumerate(children):
                sub_angle = angle - angle_span / 2 + (i + 0.5) * sub_span
                recurse(child, depth + 1, sub_angle, sub_span)
    recurse(root)
    return coords

euc_layout = radial_tree_layout() * 0.18
euc_tensor = torch.from_numpy(euc_layout).float().to(device)
hyp_embeddings = manifold.expmap0(euc_tensor)

# ────────────────────────────────────────────────
# 2. Attention-like graph & curvature proxy
# ────────────────────────────────────────────────
G = nx.Graph()
for i, token in enumerate(tokens):
    G.add_node(i, token=token, pos=hyp_embeddings[i].cpu().numpy())

# Edges weighted by inverse hyperbolic distance
distance_threshold = 0.85
for i in range(n_tokens):
    for j in range(i + 1, n_tokens):
        dist = manifold.dist(
            hyp_embeddings[i:i+1],
            hyp_embeddings[j:j+1]
        ).item()
        if dist < distance_threshold:
            weight = max(0.02, 1.0 / (dist + 0.005))
            G.add_edge(i, j, weight=weight)

# Curvature proxy (higher values indicate regions of higher positive curvature / mixing)
weighted_degree = nx.degree(G, weight="weight")
weighted_clustering = nx.clustering(G, weight="weight")
avg_neighbor_weight = {
    u: np.mean([G[u][v]["weight"] for v in G.neighbors(u)]) if G.degree(u) > 0 else 0.01
    for u in G
}

curvature_proxy = {
    u: weighted_degree[u] * (1 + weighted_clustering[u]) / (avg_neighbor_weight[u] + 0.005)
    for u in G.nodes()
}

# ────────────────────────────────────────────────
# Visualization: attention graph with curvature proxy
# ────────────────────────────────────────────────
pos = {i: G.nodes[i]["pos"] for i in G}
node_colors = [curvature_proxy[i] for i in G.nodes()]
labels = {i: tokens[i] for i in G.nodes()}

fig, ax = plt.subplots(figsize=(12, 9))
nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="lightgray", ax=ax)
sc = nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    cmap=cm.viridis_r,
    node_size=1000,
    edgecolors="black",
    ax=ax
)
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold", ax=ax)
plt.colorbar(sc, ax=ax, label="Curvature proxy")
ax.set_title("Graph induced by hyperbolic distances\n(node color = curvature proxy)")
ax.axis("equal")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
# 3. Distributed alignment via Wasserstein barycenter
# ────────────────────────────────────────────────
local_A = hyp_embeddings + torch.randn_like(hyp_embeddings) * 0.08
local_B = hyp_embeddings + torch.randn_like(hyp_embeddings) * 0.12

cloud_A = local_A.cpu().numpy()
cloud_B = local_B.cpu().numpy()
uniform_weights = np.ones(n_tokens) / n_tokens

# Hyperbolic cost matrix
cost_matrix = np.zeros((n_tokens, n_tokens))
for i in range(n_tokens):
    for j in range(n_tokens):
        cost_matrix[i, j] = manifold.dist(
            torch.from_numpy(cloud_A[i:i+1]).float().to(device),
            torch.from_numpy(cloud_B[j:j+1]).float().to(device)
        ).item()

# Compute barycenter
barycenter = ot.barycenter(
    [cloud_A, cloud_B],
    [0.5, 0.5],
    M=cost_matrix,
    weights=[uniform_weights, uniform_weights],
    method="sinkhorn",
    numItermax=12000
)

# ────────────────────────────────────────────────
# Visualization: original vs aligned embeddings
# ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(
    hyp_embeddings[:, 0].cpu(),
    hyp_embeddings[:, 1].cpu(),
    c="navy", s=140, label="Reference hierarchy"
)
axes[0].set_title("Original embeddings")
axes[0].axis("equal")
axes[0].grid(alpha=0.15)

axes[1].scatter(
    barycenter[:, 0],
    barycenter[:, 1],
    c="maroon", s=140, label="Aligned representation"
)
axes[1].set_title("Aligned representation (Wasserstein barycenter)")
axes[1].axis("equal")
axes[1].grid(alpha=0.15)

for ax in axes:
    ax.set_xlabel("Coordinate 1")
    ax.set_ylabel("Coordinate 2")

plt.suptitle("Alignment of distributed representations in hyperbolic space")
plt.tight_layout()
plt.show()

print("Execution complete.")
