import torch
import torch.nn as nn

def make_incidence(indices, num_vertices):
    # this creates a dense matrix (incidence), but
    # sparse matrices or convolutions might be more appropriate
    # in certain cases
    num_springs = len(indices)
    incidence = torch.zeros(num_springs, num_vertices, dtype=torch.float32)
    for i, item in enumerate(indices):
        i1, i2 = item
        incidence[i, i1] = 1
        incidence[i, i2] = -1
    return incidence

class HookeanSprings(nn.Module):
    def __init__(self, indices, l0, k, num_vertices):
        super().__init__()
        self.indices = indices
        self.register_buffer("incidence", make_incidence(indices, num_vertices))
        self.register_buffer("l0", l0)
        self.register_buffer("k", k)

    def energy(self, x):
        d = self.incidence.mm(x)
        q = d.pow(2).sum(1)
        l = (q + 1e-6).sqrt()
        dl = l - self.l0
        e = 0.5 * (self.k * dl.pow(2)).sum()
        return e