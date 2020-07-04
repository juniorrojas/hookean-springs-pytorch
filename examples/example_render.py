import torch
from hookean_springs import HookeanSprings
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.animation import FuncAnimation

# vertex positions
x = torch.tensor([
    [0, 0],
    [1, 0],
    [2, 0.5],
    [1, 1],
    [1.5, 2],
    [2, -1],
    [0, -1],
    [1, -2]
], dtype=torch.float32)
x.requires_grad_()

# springs, specified as vertex indices
indices = [
    [0, 1],
    [0, 3],
    [1, 3],
    [1, 2],
    [2, 3],
    [3, 4],
    [2, 4],
    [1, 5],
    [5, 6],
    [6, 0],
    [6, 1],
    [6, 5],
    [5, 7],
    [6, 7]
]

# rest lengths
l0 = torch.tensor([
    1,
    1.3,
    1.4,
    1.5,
    2.0,
    1.4,
    1.4,
    1.3,
    1,
    0.8,
    1,
    1,
    1,
    0.9
], dtype=torch.float32)

num_vertices, d = x.shape
num_springs = len(indices)

# stiffness
k = torch.ones(num_springs, dtype=torch.float32)

springs = HookeanSprings(indices, l0, k, num_vertices)

optimizer = torch.optim.Adam([x], lr=1e-1)

# render
fig, ax = plt.subplots()

def make_segment_data():
    segments = []
    for ind in indices:
        a = x[ind[0]].tolist()
        b = x[ind[1]].tolist()
        segments.append([a, b])
    return segments

vertices = ax.scatter([], [])
segments = mc.LineCollection(make_segment_data())

def init():
    ax.add_collection(segments)
    ax.set_aspect(1)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-2, 2.5)
    return vertices, segments

def update(frame):
    optimizer.zero_grad()
    loss = springs.energy(x)
    loss.backward()
    optimizer.step()

    segments.set_segments(make_segment_data())
    vertices.set_offsets(x.detach())
    return vertices, segments

anim = FuncAnimation(fig, update, frames=None, init_func=init, interval=30, blit=True)

plt.show()