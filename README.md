## hookean-springs-pytorch

Minimal implementation of Hookean springs in PyTorch.

This is demo code for https://medium.com/@juniorrojas/physics-based-simulation-via-backpropagation-on-energy-functions-6d3b0e93f5fb.

```
python example.py
```

The state of the simulation is stored in `x`, a `float(n, 2)` tensor where `n` is the number of vertices. The script `example.py` simply prints `x` (this repository does not contain any visualization code at the moment).

![](media/simulation.gif)
