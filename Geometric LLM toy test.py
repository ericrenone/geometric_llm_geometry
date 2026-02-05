import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# ────────────────────────────────────────────────
# 1. Configuration & Dataset
# ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming modern industries.",
    "Climate change impacts are becoming increasingly severe.",
    "Quantum computing may revolutionize cryptography.",
    "The stock market fluctuates with global events.",
    "Neuroscience explores the mysteries of the human brain.",
    "Space exploration expands our reach to the stars."
]

n_tokens = len(sentences)
# Initial high-dimensional latent space projected to 2D
latent_embeddings = np.random.randn(n_tokens, 2) * 0.45 

# ────────────────────────────────────────────────
# 2. Hyperbolic Geometry Engine (Poincaré Ball)
# ────────────────────────────────────────────────
def expmap0(x, c=1.0):
    """Maps Euclidean vectors to the Poincaré disk."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    # Ensure points stay strictly inside the boundary (< 1.0)
    scale = np.tanh(np.sqrt(c) * norm) / (norm + 1e-10)
    return scale * x

def hyp_dist(u, v, c=1.0):
    """Hyperbolic distance between two points in the Poincaré disk."""
    u_norm_sq = np.sum(u**2)
    v_norm_sq = np.sum(v**2)
    diff_norm_sq = np.sum((u - v)**2)
    eps = 1e-8
    # Poincaré distance formula
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq)
    res = 1 + 2 * c * diff_norm_sq / np.maximum(eps, denom)
    return np.arccosh(np.maximum(1.0, res))

# Project sentences into the hyperbolic manifold
hyp_embeddings = expmap0(latent_embeddings)

# ────────────────────────────────────────────────
# 3. Curvature & Graph Analysis
# ────────────────────────────────────────────────
# Construct an adjacency matrix based on hyperbolic proximity
dist_threshold = 1.2
adj = np.zeros((n_tokens, n_tokens))
for i in range(n_tokens):
    for j in range(i + 1, n_tokens):
        d = hyp_dist(hyp_embeddings[i], hyp_embeddings[j])
        if d < dist_threshold:
            adj[i, j] = adj[j, i] = 1.0 / (d + 0.1)

# Calculate Curvature Proxy (Richness of local connections)
weighted_degree = np.sum(adj, axis=1)
clustering = np.zeros(n_tokens)
for i in range(n_tokens):
    neighbors = np.where(adj[i] > 0)[0]
    if len(neighbors) > 1:
        # Check how many neighbors are connected to each other
        sub = adj[np.ix_(neighbors, neighbors)]
        clustering[i] = np.count_nonzero(sub) / (len(neighbors) * (len(neighbors)-1))

curvature_proxy = weighted_degree * (1 + clustering)

# ────────────────────────────────────────────────
# 4. Popout Real-Time Visualization
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 11), facecolor='#0d0d0d')
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)

# Setup Disk UI
disk_border = plt.Circle((0, 0), 1, color='#1a1a1a', fill=True, zorder=0)
ax.add_artist(disk_border)
ax.add_artist(plt.Circle((0, 0), 1, color='#33ccff', fill=False, lw=2, ls='--', alpha=0.5))
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal'); ax.axis('off')

# Elements for animation
sc = ax.scatter([], [], s=1200, c=[], cmap='plasma', edgecolors='white', lw=1.5, zorder=3)
edge_lines = [ax.plot([], [], color='#555555', alpha=0.3, lw=1, zorder=2)[0] for _ in range(int(np.sum(adj > 0)/2))]
labels = [ax.text(0, 0, "", color='white', fontsize=8, ha='center', fontweight='bold') for _ in range(n_tokens)]

def animate(frame):
    # Dynamic expansion effect
    alpha = min(1.0, frame / 40.0)
    current_pos = expmap0(latent_embeddings * alpha)
    
    # Update Nodes
    sc.set_offsets(current_pos)
    sc.set_array(curvature_proxy)
    
    # Update Edges
    e_idx = 0
    for i in range(n_tokens):
        for j in range(i+1, n_tokens):
            if adj[i, j] > 0:
                edge_lines[e_idx].set_data([current_pos[i,0], current_pos[j,0]], 
                                           [current_pos[i,1], current_pos[j,1]])
                e_idx += 1
    
    # Update Labels
    for i, txt in enumerate(labels):
        txt.set_position((current_pos[i, 0], current_pos[i, 1] - 0.12))
        txt.set_text(sentences[i][:25] + "...")
        txt.set_alpha(alpha)
        
    return [sc] + edge_lines + labels

ani = FuncAnimation(fig, animate, frames=80, interval=40, blit=True)
plt.title("LLM Geometric Manifold: Hyperbolic Expansion", color='white', fontsize=18, pad=20)
plt.show()

# ────────────────────────────────────────────────
# 5. Barycenter Alignment Simulation
# ────────────────────────────────────────────────
# Simulate two "views" of the same data (e.g., from two different LLM layers)
view_A = hyp_embeddings + np.random.normal(0, 0.03, hyp_embeddings.shape)
view_B = hyp_embeddings + np.random.normal(0, 0.06, hyp_embeddings.shape)
# Wasserstein-like midpoint (Euclidean proxy for visualization)
barycenter = 0.5 * (view_A + view_B)

print(f"{'Sentence Fragment':<35} | Curvature Proxy")
print("-" * 55)
for i, s in enumerate(sentences):
    print(f"{s[:32]:<35} | {curvature_proxy[i]:.4f}")
