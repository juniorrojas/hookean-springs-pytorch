## hookean-springs-pytorch

Hookean springs in PyTorch.

The code in this repository shows how to compute the potential energy of a mass-spring using differentiable tensor operations. Read more [here](https://medium.com/@juniorrojas/physics-based-simulation-via-backpropagation-on-energy-functions-6d3b0e93f5fb).

![](media/compgraph.gif)

## examples

You might need to temporarily set your `PYTHONPATH` to run the examples:

```
export PYTHONPATH=$PYTHONPATH:.
```

### minimization loop

The state containing vertex positions is `x: float(n, 2)`, where `n` is the number of vertices. The script prints `x` after every optimization step.

```
python examples/example_no_render.py
```

### matplotlib visualization

```
python examples/example_render.py
```

![](media/matplotlib.gif)
