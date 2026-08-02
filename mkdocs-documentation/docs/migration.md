# Migrating to 4.0.0

Four things changed that a 3.0.0 script will notice. Two of them raise, one is
silent, and one changes your numbers without changing your code.

## `precision` is now `splitting`

The old names counted how many non-linear applications a step made. Readers
took them for the float width — which the *field's* dtype decides, separately
and at the same time — so the argument is named for the scheme instead.

```python
# 3.0.0
E = simu.out_field(E_in, L, precision="single")
E = simu.out_field(E_in, L, precision="double")

# 4.0.0
E = simu.out_field(E_in, L, splitting="lie")     # was "single"
E = simu.out_field(E_in, L, splitting="strang")  # was "double"
```

There is no compatibility shim: `precision=` raises `TypeError`. There is also
a third scheme now, `splitting="yoshida"`, which is fourth order.

The float width is, and always was, the dtype of the array you pass in:

```python
E = simu.out_field(E_in.astype(np.complex128), L, splitting="strang")
```

## The step belongs to the call, not to the solver

```python
# 3.0.0
simu.delta_z = 1e-4
E = simu.out_field(E_in, L)

# 4.0.0
E = simu.out_field(E_in, L, delta_z=1e-4)
```

**This one is silent.** Assigning `simu.delta_z` no longer does anything, and
nothing will tell you: the run simply uses a step the solver chose. If your
results move after upgrading and you used to set `delta_z` this way, that is
where to look first.

## The default step is derived from the field

Given no `delta_z`, the solver now picks one from the field's own energy,
aiming at a fixed phase per step rather than running just under a stability
ceiling. It is a better default — it tracks the physics, so a stronger
non-linearity shortens the step on its own — but it is a *different* default,
so a run that relied on the old one takes a different number of steps.

Pass `delta_z` explicitly to reproduce an old run exactly.

## A lossy run gives a different answer

If `alpha > 0`, your results will change, because the old ones were wrong.

The real-space step applied `exp(-alpha*s*dz + i*g*|A|^2*s*dz)` with `|A|^2`
read once on entry, which is exact only while the step preserves `|A|^2` —
true of a pure rotation and false the moment there is loss. Every splitting
came out **first order** on a lossy problem, Strang and Yoshida included. The
step is solved rather than frozen now, which on the turbulence example is
worth about 69x in accuracy for Strang and 7000x for Yoshida at matched cost.

Nothing about how you call it changes, and a **lossless run is unchanged to
the bit**. See [Physical Validation](physical_validation.md) for the closed
form this is checked against.

## Also worth knowing

- **Absorption stops at the end of the medium.** Propagation past `L` used to
  switch off `n2` and leave `alpha` running, so a beam that had left the medium
  went on being absorbed by it. Both stop now. This only affects runs that
  propagate beyond `L` with loss.
- **`splitting="yoshida"` warns in two cases** rather than wasting your time
  quietly: on a `complex64` field, where round-off sets the error long before
  the splitting does, and in a lossy medium, where its backward middle sub-step
  amplifies instead of decaying.

## Older renames

If you are coming from further back than 3.0.0, three more names changed. They
are listed here because this is the page people find, not because they are new:

| Old | New |
|---|---|
| `puiss=` | `power=` |
| `backend="GPU"` | `backend="CUPY"` |
| `simu.delta_t = ...` | gone; the step is `out_field(..., delta_z=...)` |

## The full list

Everything else is in the [changelog](changelog.md).
