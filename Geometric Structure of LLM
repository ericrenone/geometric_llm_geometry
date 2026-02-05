import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm

# ────────────────────────────────────────────────
# 1. Hyperbolic Geometry Engine (NumPy Only)
# ────────────────────────────────────────────────
def poincare_dist(x, y):
    """Calculates hyperbolic distance in the Poincaré disk (c=1)."""
    sq_dist = np.sum((x - y)**2)
    x_norm_sq = np.sum(x**2)
    y_norm_sq = np.sum(y**2)
    # Clamp values to avoid precision errors near the boundary (1.0)
    eps = 1e-7
    factor = 1 + 2 * sq_dist / (max(eps, 1 - x_norm_sq) * max(eps, 1 - y_norm_sq))
    return np.acosh(max(1.0, factor))

def expmap0(v):
    """Projects Euclidean vectors into the Poincaré ball via the Exponential Map."""
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.maximum(v_norm, 1e-15)
    return np.tanh(v_norm) * v / v_norm

# ────────────────────────────────────────────────
# 2. Data Structure & Hierarchy Logic
# ────────────────────────────────────────────────
hierarchy = [
    (0, "entity", []), (1, "living_thing", [0]), (2, "non_living", [0]),
    (3, "animal", [1]), (4, "plant", [1]), (5, "mammal", [3]),
    (6, "bird", [3]), (7, "fish", [3]), (8, "tree", [4]),
    (9, "flower", [4]), (10, "dog", [5]), (11, "cat", [5]),
    (12, "eagle", [6]), (13, "sparrow", [6]), (14, "salmon", [7]),
    (15, "oak", [8]), (16, "rose", [9])
]

tokens = [node[1] for node in hierarchy]
n_tokens = len(tokens)
# Fixed mapping: child -> parent
parent_map = {node_id: p_list[0] for node_id, _, p_list in hierarchy if p_list}

# ────────────────────────────────────────────────
# 3. Layout & Graph Generation
# ────────────────────────────────────────────────
def radial_tree_layout(max_radius=2.5):
    """Generates an initial Euclidean layout for the hierarchy."""
    coords = np.zeros((n_tokens, 2))
    def recurse(node_id, depth=0, angle=0.0, angle_span=2 * np.pi):
        r = depth * (max_radius / 5.0)
        coords[node_id] = [r * np.cos(angle), r * np.sin(angle)]
        children = [c for c, p in parent_map.items() if p == node_id]
        if children:
            sub_span = angle_span / len(children)
            for i, child in enumerate(children):
                sub_angle = angle - angle_span/2 + (i + 0.5) * sub_span
                recurse(child, depth + 1, sub_angle, sub_span)
    recurse(0)
    return coords

# Build the Graph for NetworkX analysis
G = nx.Graph()
for i, token in enumerate(tokens):
    G.add_node(i, label=token)
for child, parent in parent_map.items():
    G.add_edge(parent, child)

# Compute Curvature Proxy (Degree * Clustering)
# High values = 'dense' semantic regions; Low values = 'leaf' nodes
clustering = nx.clustering(G)
node_colors = [ (G.degree(i) * (1 + clustering[i])) for i in G.nodes()]

# ────────────────────────────────────────────────
# 4. Real-Time Animated Visualization
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10), facecolor='#111111')
plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)

euc_base = radial_tree_layout()
boundary_circle = plt.Circle((0, 0), 1, color='#333333', fill=True, zorder=0)
ax.add_artist(boundary_circle)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.axis('off')

# Plot elements
edges_plot = [ax.plot([], [], color='cyan', alpha=0.4, lw=1, zorder=1)[0] for _ in G.edges()]
scatter = ax.scatter([], [], c=[], cmap=cm.plasma, s=200, edgecolors='white', zorder=2)
texts = [ax.text(0, 0, tokens[i], color='white', fontsize=9, fontweight='bold', 
                 ha='center', va='center', alpha=0) for i in range(n_tokens)]

title = ax.set_title("Hyperbolic Semantic Projection", color='white', fontsize=16, pad=20)

def update(frame):
    # Transition from origin to full hyperbolic expansion
    scale = min(1.0, frame / 50.0)
    current_hyp = expmap0(euc_base * scale)
    
    # Update Nodes
    scatter.set_offsets(current_hyp)
    scatter.set_array(np.array(node_colors))
    
    # Update Edges
    for idx, (u, v) in enumerate(G.edges()):
        x_pts = [current_hyp[u, 0], current_hyp[v, 0]]
        y_pts = [current_hyp[u, 1], current_hyp[v, 1]]
        edges_plot[idx].set_data(x_pts, y_pts)
    
    # Update Labels (with fade-in)
    for i, t in enumerate(texts):
        t.set_position((current_hyp[i, 0], current_hyp[i, 1] + 0.06))
        t.set_alpha(scale)
        
    return scatter, *edges_plot, *texts

# Run Animation
ani = FuncAnimation(fig, update, frames=100, interval=30, blit=True, repeat=True)

print("Displaying real-time hyperbolic expansion...")
plt.show()
