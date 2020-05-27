## hookean-springs-pytorch

Hookean springs in PyTorch.

![](media/compgraph.gif)

The main implementation is in `hookean_spring_potential.py` which implements an `nn.Module` that can compute the potential energy of a collection of Hookean springs.

Read more about the ideas behind this implementation: https://medium.com/@juniorrojas/physics-based-simulation-via-backpropagation-on-energy-functions-6d3b0e93f5fb.

## examples

You might need to temporarily set your `PYTHONPATH` to run the examples:

```
export PYTHONPATH=$PYTHONPATH:.
```

A minimization loop that prints the state every iteration. The state is `x`, a `float(n, 2)` tensor where `n` is the number of vertices:

```
python examples/example_no_render.py
```

A basic visualization using matplotlib:

```
python examples/example_render.py
```

![](media/matplotlib.gif)
