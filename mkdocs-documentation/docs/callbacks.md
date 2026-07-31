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

A callback normally returns nothing. Returning a number asks the solver to
**use that as the step** from here on; see `adapt_delta_z` below.

## Passing Callbacks to `out_field`

### Single callback

```python
E_out = simu.out_field(
    E_in, L,
    delta_z=L / 1000,
    callback=my_callback,
    callback_args=(arg1, arg2),
)
```

### Multiple callbacks

Pass a list of callbacks and a list of argument tuples:

```python
E_out = simu.out_field(
    E_in, L,
    delta_z=L / 1000,
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

delta_z = L / 1000          # divides L, so the run is exactly 1000 steps
n_steps = int(L / delta_z)
save_every = 100
n_samples = n_steps // save_every + 1
E_samples = np.zeros((n_samples, NY, NX), dtype=np.complex64)

E_out = simu.out_field(
    E_in, L,
    delta_z=delta_z,
    callback=sample,
    callback_args=(save_every, E_samples),
)
# E_samples now contains field snapshots at every 100th step
```

### `norm` -- Track field norm

Records the total intensity ($\sum |A|^2$) at regular intervals.

```python
from NLSE import norm

delta_z = L / 1000          # divides L, so the run is exactly 1000 steps
n_steps = int(L / delta_z)
save_every = 50
norms = np.zeros(n_steps // save_every + 1)

E_out = simu.out_field(
    E_in, L,
    delta_z=delta_z,
    callback=norm,
    callback_args=(save_every, norms),
)
```

### `evaluate_delta_n` -- Monitor nonlinear index change

Evaluates the nonlinear refractive index change $\delta n = n_2 |E|^2 / (1 + |E|^2/I_\text{sat})$ at regular intervals.

```python
from NLSE import evaluate_delta_n

delta_z = L / 1000          # divides L, so the run is exactly 1000 steps
n_steps = int(L / delta_z)
save_every = 100
delta_n = np.zeros((n_steps // save_every + 1,) + (NY, NX))

E_out = simu.out_field(
    E_in, L,
    delta_z=delta_z,
    callback=evaluate_delta_n,
    callback_args=(save_every, delta_n),
)
```

### `adapt_delta_z` -- Adaptive step sizing

Dynamically adjusts the step size based on the nonlinear phase accumulated per
step. Updates every `update_every` steps.

It changes the step by **returning** it. The solver rebuilds the linear
propagator to match before taking the next step, so the two halves of a split
step always advance by the same distance — assigning the step somewhere would
leave the propagator built from the previous one, and the linear part would
quietly advance by the wrong distance.

```python
delta_z = L / 1000     # divides L: exactly 1000 steps

from NLSE import adapt_delta_z

delta_z_history = []

E_out = simu.out_field(
    E_in, L,
    delta_z=delta_z,
    callback=adapt_delta_z,
    callback_args=(100, delta_z_history),
)
# delta_z_history contains the step size at each step
```

Writing your own adaptive callback, make sure it settles on a value rather than
adjusting on every step: one that shrinks the step unconditionally will never
reach the end of the propagation.

```python
def refine_past_halfway(simu, A, z, i, dz_fine):
    """Switch to a finer step once past the middle of the medium."""
    return dz_fine if z >= simu.L / 2 else None
```

`simu._current_delta_z` holds the step in force, for a callback that needs to
see it.

### `adapt_delta_z_to_error` -- step sizing from a measured error

`adapt_delta_z` reads the step off the peak nonlinear index and divides by
twelve. That is a *rate*: it says how fast the phase turns and nothing about
how much of the answer the splitting is losing. This one measures instead —
every `update_every` steps it takes the same distance once whole and once in
two halves and compares, then solves for the step that would have hit the
tolerance.

```python
from NLSE.callbacks import adapt_delta_z_to_error

steps = []
E_out = simu.out_field(
    E_in, L,
    callback=adapt_delta_z_to_error,
    callback_args=(1e-6, 20, (0.5, 2.0), 0.9, None, steps),
)
```

The arguments are `(tolerance, update_every, bounds, safety, min_step,
delta_z)`. It costs three extra steps each time it fires, so `update_every`
trades that overhead against how quickly the step follows the physics.

Two things worth knowing before trusting a tolerance:

- **The step that minimises the error is not the smallest one.** In `complex64`
  the round-off accumulating over steps eventually grows faster than the
  splitting error falls, so past some step size a finer one costs time and
  accuracy together.
- **The estimate has a floor of its own.** Below that same scale, one step and
  two halves differ by round-off rather than by splitting error, so a tolerance
  under the floor reads as "no error at all" and asks for a *bigger* step. The
  controller is capped by the physics ($\pi$ per step) so this cannot run away,
  but a tolerance it cannot meet will simply sit at the cap.

Pass `min_step` to stop it shrinking past a step you know is enough.

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
