"""
Geometric LLM Simulation - No external dependencies

- Hyperbolic embeddings (Poincare ball) via NumPy/Torch
- Attention-like graph with curvature proxy
- Wasserstein-like barycenter alignment (manual)
- Visualizations via matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

# -----------------------------
# Configuration & reproducibility
# -----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Toy "real-world" sentences / embeddings
# -----------------------------
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming modern industries.",
    "Climate change impacts are becoming increasingly severe.",
    "Quantum computing may revolutionize cryptography.",
    "The stock market fluctuates with global events."
]

n_tokens = len(sentences)
dim = 2  # low-dim for visualization

# Generate synthetic embeddings (replace with real embeddings if you want)
embeddings = np.random.randn(n_tokens, dim)

# -----------------------------
# Poincare Ball utilities
# -----------------------------
def expmap0(x, c=1.0):
    """Exponential map at the origin for Poincare ball"""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.tanh(np.sqrt(c) * norm) / (norm + 1e-8)
    return scale * x

def hyp_dist(u, v, c=1.0):
    """Hyperbolic distance in Poincare ball"""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    diff = np.linalg.norm(u - v)**2
    return np.arccosh(1 + 2*diff / ((1 - norm_u**2) * (1 - norm_v**2) + 1e-8))

# Map embeddings to Poincare ball
hyp_embeddings = expmap0(embeddings)

# -----------------------------
# Attention-like graph & curvature proxy
# -----------------------------
distance_threshold = 0.85
adj_matrix = np.zeros((n_tokens, n_tokens))
for i in range(n_tokens):
    for j in range(i+1, n_tokens):
        dist = hyp_dist(hyp_embeddings[i], hyp_embeddings[j])
        if dist < distance_threshold:
            weight = max(0.02, 1.0 / (dist + 1e-5))
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight

# Compute curvature proxy (simple approximation)
weighted_degree = np.sum(adj_matrix, axis=1)
clustering = np.zeros(n_tokens)
for i in range(n_tokens):
    neighbors = np.where(adj_matrix[i] > 0)[0]
    if len(neighbors) > 1:
        links = 0
        for u in neighbors:
            for v in neighbors:
                if u != v and adj_matrix[u, v] > 0:
                    links += 1
        clustering[i] = links / (len(neighbors) * (len(neighbors) - 1))
    else:
        clustering[i] = 0
avg_neighbor_weight = np.array([
    np.mean(adj_matrix[i][adj_matrix[i]>0]) if weighted_degree[i]>0 else 0.01
    for i in range(n_tokens)
])
curvature_proxy = weighted_degree * (1 + clustering) / (avg_neighbor_weight + 1e-5)

# -----------------------------
# Visualization: graph
# -----------------------------
plt.figure(figsize=(12,8))
for i in range(n_tokens):
    plt.scatter(hyp_embeddings[i,0], hyp_embeddings[i,1],
                s=1200, c=curvature_proxy[i], cmap='plasma', edgecolors='black')
    plt.text(hyp_embeddings[i,0]+0.01, hyp_embeddings[i,1]+0.01,
             sentences[i][:30]+"...", fontsize=9)
# draw lines
for i in range(n_tokens):
    for j in range(i+1, n_tokens):
        if adj_matrix[i,j] > 0:
            plt.plot([hyp_embeddings[i,0], hyp_embeddings[j,0]],
                     [hyp_embeddings[i,1], hyp_embeddings[j,1]],
                     'gray', alpha=0.4)
plt.colorbar(label="Curvature proxy")
plt.title("Attention-like graph with curvature proxy")
plt.axis("equal")
plt.show()

# -----------------------------
# Manual barycenter alignment (average)
# -----------------------------
local_A = hyp_embeddings + np.random.randn(n_tokens, dim) * 0.05
local_B = hyp_embeddings + np.random.randn(n_tokens, dim) * 0.07
barycenter = 0.5 * local_A + 0.5 * local_B

# -----------------------------
# Visualization: original vs aligned
# -----------------------------
fig, axes = plt.subplots(1,2, figsize=(14,6))
axes[0].scatter(hyp_embeddings[:,0], hyp_embeddings[:,1], c='navy', s=140)
axes[0].set_title("Original embeddings")
axes[0].grid(alpha=0.15)
axes[1].scatter(barycenter[:,0], barycenter[:,1], c='maroon', s=140)
axes[1].set_title("Aligned barycenter embeddings")
axes[1].grid(alpha=0.15)
for ax in axes:
    ax.set_xlabel("Coord 1")
    ax.set_ylabel("Coord 2")
plt.suptitle("Hyperbolic embeddings alignment (manual barycenter)")
plt.tight_layout()
plt.show()

# -----------------------------
# Summary table
# -----------------------------
print("\n=== Curvature Proxy Summary ===\n")
for i in range(n_tokens):
    print(f"{sentences[i][:40]:40} | Curvature: {curvature_proxy[i]:.3f}")
