import torch
from hookean_springs import HookeanSprings

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

for i in range(100):
    optimizer.zero_grad()
    loss = springs.energy(x)
    loss.backward()
    optimizer.step()
    print("iteration: {}".format(i))
    print(x)