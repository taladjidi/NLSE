# Callbacks

Callbacks allow you to monitor, record, or modify the simulation during propagation. They are executed at every step inside the main solver loop.

## Callback Signature

All callbacks have the signature:

```python
def my_callback(simu, A, z, i, *args):
    ...
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `simu` | solver instance | The solver object (access all attributes) |
| `A` | `np.ndarray` | The current field (on device for GPU backends) |
| `z` | `float` | Current propagation distance or time |
| `i` | `int` | Current step number |
| `*args` | any | Additional arguments passed via `callback_args` |

## Passing Callbacks to `out_field`

### Single callback

```python
E_out = simu.out_field(
    E_in, L,
    callback=my_callback,
    callback_args=(arg1, arg2),
)
```

### Multiple callbacks

Pass a list of callbacks and a list of argument tuples:

```python
E_out = simu.out_field(
    E_in, L,
    callback=[callback_a, callback_b],
    callback_args=[(args_a1, args_a2), (args_b1,)],
)
```

## Built-in Callbacks

Import built-in callbacks from the top-level package:

```python
from NLSE import sample, norm, evaluate_delta_n, adapt_delta_z
```

### `sample` -- Save field snapshots

Saves the full field every `save_every` steps into a pre-allocated array.

```python
from NLSE import sample

n_steps = int(L / simu.delta_z)
save_every = 100
n_samples = n_steps // save_every + 1
E_samples = np.zeros((n_samples, NY, NX), dtype=np.complex64)

E_out = simu.out_field(
    E_in, L,
    callback=sample,
    callback_args=(save_every, E_samples),
)
# E_samples now contains field snapshots at every 100th step
```

### `norm` -- Track field norm

Records the total intensity ($\sum |A|^2$) at regular intervals.

```python
from NLSE import norm

n_steps = int(L / simu.delta_z)
save_every = 50
norms = np.zeros(n_steps // save_every + 1)

E_out = simu.out_field(
    E_in, L,
    callback=norm,
    callback_args=(save_every, norms),
)
```

### `evaluate_delta_n` -- Monitor nonlinear index change

Evaluates the nonlinear refractive index change $\delta n = n_2 |E|^2 / (1 + |E|^2/I_\text{sat})$ at regular intervals.

```python
from NLSE import evaluate_delta_n

n_steps = int(L / simu.delta_z)
save_every = 100
delta_n = np.zeros((n_steps // save_every + 1,) + (NY, NX))

E_out = simu.out_field(
    E_in, L,
    callback=evaluate_delta_n,
    callback_args=(save_every, delta_n),
)
```

### `adapt_delta_z` -- Adaptive step sizing

Dynamically adjusts the step size based on the nonlinear phase accumulated per step. Updates every `update_every` steps.

```python
from NLSE import adapt_delta_z

delta_z_history = []

E_out = simu.out_field(
    E_in, L,
    callback=adapt_delta_z,
    callback_args=(100, delta_z_history),
)
# delta_z_history contains the step size at each step
```

## Writing Custom Callbacks

Since the solver instance is passed to the callback, you have access to all solver attributes:

```python
def print_progress(simu, A, z, i):
    if i % 1000 == 0:
        intensity = (A.real**2 + A.imag**2).max()
        print(f"Step {i}: z={z:.6f}, peak intensity={intensity:.4e}")

E_out = simu.out_field(E_in, L, callback=print_progress)
```

### Performance considerations

Callbacks run inside the main solver loop at every step, so they should be lightweight. Use a modulo check (`if i % N == 0`) to run expensive operations only periodically.

Avoid transferring data from GPU to CPU inside callbacks unless necessary, as this forces synchronization and reduces throughput.
