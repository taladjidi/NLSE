// CUDA C kernels for NLSE operations (optimized fused versions only)
// Precision-agnostic template using {{FP_TYPE}}, {{FP2_TYPE}}, {{FP_SUFFIX}}, {{SINCOS_FUNC}}
//
// A kernel that reads a potential is written once, between the VBLOCK markers
// and against the V_ARG/V_PHASE/V_LOSS macros, and compiled three times: as
// <name> with no potential, <name>_v with a real one and <name>_cv with a
// complex (absorbing) one. See _expand_v_blocks in kernels/cupy_kernels.py.

// The saturation the interaction should be evaluated at over a lossy step.
//
// A real-space step that freezes |A|^2 is exact only where |A|^2 is constant
// through it -- true of a pure rotation, false the moment there is loss, since
// the amplitude decays inside the step while the interaction goes on turning
// the phase at the rate the step began with. Frozen, the sub-step is O(dz^2)
// locally and every composition over it is first order: Lie, Strang and
// Yoshida alike, measured.
//
// With y = |A|^2, s(y) = 1/(1 + y/Isat) and u = 2*alpha*dz, two exact facts,
//
//     dy/dz = -2*alpha*s*y,  dphi/dz = g*y*s   =>   dphi/dy = -g/(2*alpha)
//
// give the phase over the step as (g/(2*alpha))*(y0 - y_end) whatever the
// saturation does in between, with y_end fixed by
// ln y + y/Isat = ln y0 + y0/Isat - u.
//
// Returned is P = (1 - y_end/y0)/u, that phase written as a saturation: the
// step applies g*y0*P*dz where it applied g*y0*sat*dz, and scales the
// amplitude by sqrt(1 - P*u) where it multiplied by exp(-alpha*sat*dz). Three
// passes put the composition back at fourth order, which is all Yoshida can
// use. The bracket is the series of (-log1p(-P*u) - P*u)/(P*u)^2, and it
// contracts while a step loses well under half the intensity -- which is what
// LOSS_PER_STEP_LIMIT in solvers/step_size.py keeps it to.
//
// u == 0 returns early, so a lossless run computes what it always did, to the
// bit, and pays one comparison for it.
__device__ inline {{FP_TYPE}} nlse_loss_factor(
    const {{FP_TYPE}} sat,
    const {{FP_TYPE}} u
) {
    if (u == ({{FP_TYPE}})0.0) return sat;
    {{FP_TYPE}} P = sat;
    for (int pass = 0; pass < 3; ++pass) {
        {{FP_TYPE}} Pu = P * u;
        P = sat * (({{FP_TYPE}})1.0 - Pu * P * (({{FP_TYPE}})0.5
            + Pu * ({{FP_TYPE}})(1.0 / 3.0) + Pu * Pu * ({{FP_TYPE}})0.25));
    }
    return P;
}

// Largest |u| = |2*alpha*dz| the iteration above is used at, and why there is a
// limit: it contracts only while the step takes out a small fraction of the
// intensity, and the fraction it sees is sat*u, so the bound has to hold at
// sat = 1. Measured against a stiff solve there it beats the frozen step by
// 2.6x at u = 0.1 and loses to it by u = 0.3; near 1 it walks away and returns
// a larger field than it was given. Above this the kernels compute what they
// computed before -- see LOSS_PER_STEP_LIMIT in solvers/step_size.py, which
// caps a propagation at half of it, so only a direct kernel call gets here.
#define NLSE_LOSS_SOLVED_LIMIT 0.1

// FUSED: square_mod + nl_prop
// {{VBLOCK}}
extern "C" __global__ void square_mod_nl_prop(
    {{FP2_TYPE}}* A,
    V_ARG(V)
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    // Compute square modulus inline
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;

    // Apply nonlinear propagation immediately. The loss has left arg_real for
    // exp_real_part below, where the step is solved rather than frozen; what
    // remains there is a complex potential's own absorption, which the
    // identity behind nlse_loss_factor does not cover and which is zero in the
    // other two twins.
    {{FP_TYPE}} u = ({{FP_TYPE}})2.0 * alpha * dz;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq_val / Isat);
    // Solved, or frozen as before where solving it is out of reach. Both
    // branches are uniform across the grid: u is a scalar.
    bool solved = u <= ({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT
        && u >= -({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT;
    {{FP_TYPE}} P = solved ? nlse_loss_factor(sat, u) : sat;
    {{FP_TYPE}} alpha_left = solved ? ({{FP_TYPE}})0.0 : alpha;
    {{FP_TYPE}} amp = solved ? sqrt{{FP_SUFFIX}}(({{FP_TYPE}})1.0 - P * u) : ({{FP_TYPE}})1.0;
    {{FP_TYPE}} arg_real = -(alpha_left * sat V_LOSS(V, idx)) * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq_val * P V_PHASE(V, idx)) * dz;
    {{FP_TYPE}} exp_real_part = amp * exp{{FP_SUFFIX}}(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    {{SINCOS_FUNC}}(arg_imag, &sin_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = make_{{FP2_TYPE}}(exp_real_part * cos_imag, exp_real_part * sin_imag);

    A[idx] = make_{{FP2_TYPE}}(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
// {{END_VBLOCK}}

// Propagator multiplication
extern "C" __global__ void apply_propagator(
    {{FP2_TYPE}}* A,
    const {{FP2_TYPE}}* propagator,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} A_val = A[idx];
    {{FP2_TYPE}} prop_val = propagator[idx];

    // Complex multiplication: A *= propagator
    A[idx] = make_{{FP2_TYPE}}(
        A_val.x * prop_val.x - A_val.y * prop_val.y,
        A_val.x * prop_val.y + A_val.y * prop_val.x
    );
}

// SEPARATE KERNELS (required when nl_length > 0 or for coupled solvers)

// Square modulus computation
extern "C" __global__ void square_mod(
    const {{FP2_TYPE}}* A,
    {{FP_TYPE}}* A_sq,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} A_val = A[idx];
    A_sq[idx] = A_val.x * A_val.x + A_val.y * A_val.y;
}

// Nonlinear propagation
// {{VBLOCK}}
extern "C" __global__ void nl_prop(
    {{FP2_TYPE}}* A,
    const {{FP_TYPE}}* A_sq,
    V_ARG(V)
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    // See square_mod_nl_prop above, and nlse_loss_factor for why P and not sat.
    {{FP_TYPE}} u = ({{FP_TYPE}})2.0 * alpha * dz;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq[idx] / Isat);
    // Solved, or frozen as before where solving it is out of reach. Both
    // branches are uniform across the grid: u is a scalar.
    bool solved = u <= ({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT
        && u >= -({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT;
    {{FP_TYPE}} P = solved ? nlse_loss_factor(sat, u) : sat;
    {{FP_TYPE}} alpha_left = solved ? ({{FP_TYPE}})0.0 : alpha;
    {{FP_TYPE}} amp = solved ? sqrt{{FP_SUFFIX}}(({{FP_TYPE}})1.0 - P * u) : ({{FP_TYPE}})1.0;
    {{FP_TYPE}} arg_real = -(alpha_left * sat V_LOSS(V, idx)) * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq[idx] * P V_PHASE(V, idx)) * dz;
    {{FP_TYPE}} exp_real_part = amp * exp{{FP_SUFFIX}}(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    {{SINCOS_FUNC}}(arg_imag, &sin_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = make_{{FP2_TYPE}}(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A_val = A[idx];
    A[idx] = make_{{FP2_TYPE}}(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
// {{END_VBLOCK}}

// Coupled nonlinear propagation (for CNLSE, DDGPE)
// {{VBLOCK}}
extern "C" __global__ void nl_prop_c(
    {{FP2_TYPE}}* A1,
    const {{FP_TYPE}}* A_sq_1,
    const {{FP_TYPE}}* A_sq_2,
    V_ARG(V)
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    {{FP_TYPE}} arg_real = -(alpha * sat V_LOSS(V, idx)) * dz;
    {{FP_TYPE}} arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat V_PHASE(V, idx)) * dz;
    {{FP_TYPE}} exp_real_part = exp{{FP_SUFFIX}}(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    {{SINCOS_FUNC}}(arg_imag, &sin_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = make_{{FP2_TYPE}}(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A1_val = A1[idx];
    A1[idx] = make_{{FP2_TYPE}}(
        A1_val.x * exp_arg.x - A1_val.y * exp_arg.y,
        A1_val.x * exp_arg.y + A1_val.y * exp_arg.x
    );
}
// {{END_VBLOCK}}

// RK4 utility kernels (stage building and accumulation)

// RK4 AXPY: out = A + c * k
extern "C" __global__ void rk4_axpy(
    {{FP2_TYPE}}* out,
    const {{FP2_TYPE}}* A,
    const {{FP_TYPE}} c,
    const {{FP2_TYPE}}* k,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} A_val = A[idx];
    {{FP2_TYPE}} k_val = k[idx];
    out[idx] = make_{{FP2_TYPE}}(A_val.x + c * k_val.x, A_val.y + c * k_val.y);
}

// RK4 Accumulate: acc += w * k
extern "C" __global__ void rk4_accumulate(
    {{FP2_TYPE}}* acc,
    const {{FP_TYPE}} w,
    const {{FP2_TYPE}}* k,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} acc_val = acc[idx];
    {{FP2_TYPE}} k_val = k[idx];
    acc[idx] = make_{{FP2_TYPE}}(acc_val.x + w * k_val.x, acc_val.y + w * k_val.y);
}

// FUSED RK4 stage update: acc = k, out = A + c * k  (used for stage 1)
// Combines the copy-to-acc and the axpy-to-A_tmp into one launch, which reads
// k once instead of twice.
extern "C" __global__ void rk4_set_and_axpy(
    {{FP2_TYPE}}* acc,
    {{FP2_TYPE}}* out,
    const {{FP2_TYPE}}* A,
    const {{FP2_TYPE}}* k,
    const {{FP_TYPE}} c,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} k_val = k[idx];
    acc[idx] = k_val;
    {{FP2_TYPE}} A_val = A[idx];
    out[idx] = make_{{FP2_TYPE}}(A_val.x + c * k_val.x, A_val.y + c * k_val.y);
}

// FUSED RK4 stage update: acc += w * k, out = A + c * k  (used for stages 2-3)
extern "C" __global__ void rk4_acc_and_axpy(
    {{FP2_TYPE}}* acc,
    {{FP2_TYPE}}* out,
    const {{FP2_TYPE}}* A,
    const {{FP2_TYPE}}* k,
    const {{FP_TYPE}} w,
    const {{FP_TYPE}} c,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} k_val = k[idx];
    {{FP2_TYPE}} acc_val = acc[idx];
    acc[idx] = make_{{FP2_TYPE}}(acc_val.x + w * k_val.x, acc_val.y + w * k_val.y);
    {{FP2_TYPE}} A_val = A[idx];
    out[idx] = make_{{FP2_TYPE}}(A_val.x + c * k_val.x, A_val.y + c * k_val.y);
}

// FUSED RK4 close-out: A += w * (acc + k)  (the last stage and the update)
// The step otherwise ends by accumulating the fourth slope into acc and then
// reading acc back to update A, which is six passes over memory where this is
// four. Exact, because acc + 1*k rounds the same as acc + k.
extern "C" __global__ void rk4_final_update(
    {{FP2_TYPE}}* A,
    const {{FP2_TYPE}}* acc,
    const {{FP2_TYPE}}* k,
    const {{FP_TYPE}} w,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} acc_val = acc[idx];
    {{FP2_TYPE}} k_val = k[idx];
    {{FP2_TYPE}} A_val = A[idx];
    A[idx] = make_{{FP2_TYPE}}(
        A_val.x + w * (acc_val.x + k_val.x),
        A_val.y + w * (acc_val.y + k_val.y)
    );
}

// RK4 nonlinear RHS kernels (additive, no exp)
// These accumulate onto A_prop: A_prop += (nonlinear terms) * A

// RK4 NL RHS
// {{VBLOCK}}
extern "C" __global__ void rk4_nl_rhs(
    {{FP2_TYPE}}* A_prop,
    const {{FP2_TYPE}}* A,
    const {{FP_TYPE}}* A_sq,
    V_ARG(V)
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq[idx] / Isat);
    // coeff = -alpha*sat + 1j*(g*A_sq*sat + V)
    {{FP_TYPE}} coeff_r = -(alpha * sat V_LOSS(V, idx));
    {{FP_TYPE}} coeff_i = g * A_sq[idx] * sat V_PHASE(V, idx);
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = make_{{FP2_TYPE}}(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}
// {{END_VBLOCK}}

// FUSED: |A|^2 + RK4 NL RHS
// {{VBLOCK}}
extern "C" __global__ void square_mod_rk4_nl_rhs(
    {{FP2_TYPE}}* A_prop,
    const {{FP2_TYPE}}* A,
    V_ARG(V)
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq_val / Isat);
    {{FP_TYPE}} coeff_r = -(alpha * sat V_LOSS(V, idx));
    {{FP_TYPE}} coeff_i = g * A_sq_val * sat V_PHASE(V, idx);
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = make_{{FP2_TYPE}}(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}
// {{END_VBLOCK}}

// Coupled RK4 NL RHS
// Interaction NOT multiplied by A_orig (matches CNLSE RK4 math)
// {{VBLOCK}}
extern "C" __global__ void rk4_nl_rhs_c(
    {{FP2_TYPE}}* A_prop,
    const {{FP2_TYPE}}* A_orig,
    const {{FP_TYPE}}* A_sq_1,
    const {{FP_TYPE}}* A_sq_2,
    V_ARG(V)
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    // NL coefficient: (1j*interact - alpha*sat + 1j*V) * A_orig
    {{FP_TYPE}} interact_i = (g11 * A_sq_1[idx] + g12 * A_sq_2[idx]) * sat;
    {{FP2_TYPE}} A_val = A_orig[idx];
    {{FP_TYPE}} coeff_r = -(alpha * sat V_LOSS(V, idx));
    {{FP_TYPE}} coeff_i = interact_i V_PHASE(V, idx);
    {{FP_TYPE}} nl_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} nl_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = make_{{FP2_TYPE}}(
        A_prop_val.x + nl_r,
        A_prop_val.y + nl_i
    );
}
// {{END_VBLOCK}}

// Whole-stage RK4 kernels
//
// A stage otherwise writes its slope to memory and the stage update reads it
// straight back: 8 accesses per element where 6 will do. These take the
// linear part as the transform left it, finish the slope in registers, and
// spend it on the accumulator and the next stage's argument without it ever
// reaching memory.
//
// One kernel serves all four stages. `mode` is uniform across the grid, so
// the branches cost nothing and each stage touches only what it needs -- in
// particular stage 1 sets the accumulator rather than reading it, and stage 4
// does not write it at all:
//
//   mode 0 (stage 1)    acc = rhs;         out = A + c*rhs
//   mode 1 (stages 2-3) acc = acc + w*rhs; out = A + c*rhs
//   mode 2 (stage 4)    acc untouched;     out = A + c*(acc + rhs)
//
// In stages 2-4 `A_orig` and `out` are the same buffer. Each thread reads and
// writes only its own index, so the read precedes the write for the element
// it concerns and no thread depends on another's.

// {{VBLOCK}}
extern "C" __global__ void square_mod_rk4_stage(
    const {{FP2_TYPE}}* lin,
    const {{FP2_TYPE}}* A_orig,
    V_ARG(V)
    {{FP2_TYPE}}* acc,
    {{FP2_TYPE}}* out,
    const {{FP2_TYPE}}* A,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat,
    const {{FP_TYPE}} w,
    const {{FP_TYPE}} c,
    const int mode,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    // The slope, exactly as square_mod_rk4_nl_rhs forms it.
    {{FP2_TYPE}} A_val = A_orig[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq_val / Isat);
    {{FP_TYPE}} coeff_r = -(alpha * sat V_LOSS(V, idx));
    {{FP_TYPE}} coeff_i = g * A_sq_val * sat V_PHASE(V, idx);
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} lin_val = lin[idx];
    {{FP2_TYPE}} rhs = make_{{FP2_TYPE}}(lin_val.x + contrib_r, lin_val.y + contrib_i);

    {{FP2_TYPE}} carried = rhs;
    if (mode != 0) {
        {{FP2_TYPE}} acc_val = acc[idx];
        if (mode == 1) {
            acc[idx] = make_{{FP2_TYPE}}(acc_val.x + w * rhs.x, acc_val.y + w * rhs.y);
        } else {
            carried = make_{{FP2_TYPE}}(acc_val.x + rhs.x, acc_val.y + rhs.y);
        }
    } else {
        acc[idx] = rhs;
    }

    {{FP2_TYPE}} base = A[idx];
    out[idx] = make_{{FP2_TYPE}}(
        base.x + c * carried.x,
        base.y + c * carried.y
    );
}
// {{END_VBLOCK}}

// The same, for an interleaved (2, N_sq) coupled field.
// {{VBLOCK}}
extern "C" __global__ void coupled_rk4_stage_c(
    const {{FP2_TYPE}}* lin,
    const {{FP2_TYPE}}* A_orig,
    V_ARG(V1)
    V_ARG(V2)
    {{FP2_TYPE}}* acc,
    {{FP2_TYPE}}* out,
    const {{FP2_TYPE}}* A,
    const {{FP_TYPE}} alpha1,
    const {{FP_TYPE}} alpha2,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} g22,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2,
    const {{FP_TYPE}} w,
    const {{FP_TYPE}} c,
    const int mode,
    const int N_sq
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_sq) return;
    {{FP2_TYPE}} a1 = A_orig[idx];
    {{FP2_TYPE}} a2 = A_orig[idx + N_sq];
    {{FP_TYPE}} sq1 = a1.x * a1.x + a1.y * a1.y;
    {{FP_TYPE}} sq2 = a2.x * a2.x + a2.y * a2.y;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + sq1 / Isat1 + sq2 / Isat2);

    {{FP_TYPE}} interact1_i = (g11 * sq1 + g12 * sq2) * sat;
    {{FP_TYPE}} coeff1_r = -(alpha1 * sat V_LOSS(V1, idx));
    {{FP_TYPE}} coeff1_i = interact1_i V_PHASE(V1, idx);
    {{FP_TYPE}} nl1_r = coeff1_r * a1.x - coeff1_i * a1.y;
    {{FP_TYPE}} nl1_i = coeff1_r * a1.y + coeff1_i * a1.x;

    {{FP_TYPE}} interact2_i = (g22 * sq2 + g12 * sq1) * sat;
    {{FP_TYPE}} coeff2_r = -(alpha2 * sat V_LOSS(V2, idx));
    {{FP_TYPE}} coeff2_i = interact2_i V_PHASE(V2, idx);
    {{FP_TYPE}} nl2_r = coeff2_r * a2.x - coeff2_i * a2.y;
    {{FP_TYPE}} nl2_i = coeff2_r * a2.y + coeff2_i * a2.x;

    {{FP2_TYPE}} lin1 = lin[idx];
    {{FP2_TYPE}} lin2 = lin[idx + N_sq];
    {{FP2_TYPE}} rhs1 = make_{{FP2_TYPE}}(lin1.x + nl1_r, lin1.y + nl1_i);
    {{FP2_TYPE}} rhs2 = make_{{FP2_TYPE}}(lin2.x + nl2_r, lin2.y + nl2_i);

    {{FP2_TYPE}} carried1 = rhs1;
    {{FP2_TYPE}} carried2 = rhs2;
    if (mode != 0) {
        {{FP2_TYPE}} acc1 = acc[idx];
        {{FP2_TYPE}} acc2 = acc[idx + N_sq];
        if (mode == 1) {
            acc[idx] = make_{{FP2_TYPE}}(acc1.x + w * rhs1.x, acc1.y + w * rhs1.y);
            acc[idx + N_sq] = make_{{FP2_TYPE}}(
                acc2.x + w * rhs2.x, acc2.y + w * rhs2.y);
        } else {
            carried1 = make_{{FP2_TYPE}}(acc1.x + rhs1.x, acc1.y + rhs1.y);
            carried2 = make_{{FP2_TYPE}}(acc2.x + rhs2.x, acc2.y + rhs2.y);
        }
    } else {
        acc[idx] = rhs1;
        acc[idx + N_sq] = rhs2;
    }

    {{FP2_TYPE}} base1 = A[idx];
    {{FP2_TYPE}} base2 = A[idx + N_sq];
    out[idx] = make_{{FP2_TYPE}}(
        base1.x + c * carried1.x, base1.y + c * carried1.y);
    out[idx + N_sq] = make_{{FP2_TYPE}}(
        base2.x + c * carried2.x, base2.y + c * carried2.y);
}
// {{END_VBLOCK}}

// Interleaved coupled kernels
//
// These read both components straight out of the (2, ...) array the coupled
// solvers already hold: the thread at idx takes A[idx] as component 1 and
// A[idx + N_sq] as component 2. The kernels above take one component at a
// time, so reaching them means copying each component out and the results
// back -- 36 complex and 10 real array copies per step across the coupled
// cases, moving 185 MB and computing nothing.
//
// The saturation factor is shared by the two components, so it is also
// computed once here rather than twice.

// Coupled nonlinear propagation on an interleaved (2, N_sq) field
// {{VBLOCK}}
extern "C" __global__ void coupled_nl_prop_c(
    {{FP2_TYPE}}* A,
    V_ARG(V1)
    V_ARG(V2)
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha1,
    const {{FP_TYPE}} alpha2,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} g22,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2,
    const int N_sq
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_sq) return;
    {{FP2_TYPE}} a1 = A[idx];
    {{FP2_TYPE}} a2 = A[idx + N_sq];
    {{FP_TYPE}} sq1 = a1.x * a1.x + a1.y * a1.y;
    {{FP_TYPE}} sq2 = a2.x * a2.x + a2.y * a2.y;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + sq1 / Isat1 + sq2 / Isat2);

    // Component 1
    {{FP_TYPE}} arg_real1 = -(alpha1 * sat V_LOSS(V1, idx)) * dz;
    {{FP_TYPE}} arg_imag1 = (g11 * sq1 * sat + g12 * sq2 * sat V_PHASE(V1, idx)) * dz;
    {{FP_TYPE}} exp_real1 = exp{{FP_SUFFIX}}(arg_real1);
    {{FP_TYPE}} cos_imag1, sin_imag1;
    {{SINCOS_FUNC}}(arg_imag1, &sin_imag1, &cos_imag1);
    {{FP2_TYPE}} exp_arg1 = make_{{FP2_TYPE}}(exp_real1 * cos_imag1, exp_real1 * sin_imag1);
    A[idx] = make_{{FP2_TYPE}}(
        a1.x * exp_arg1.x - a1.y * exp_arg1.y,
        a1.x * exp_arg1.y + a1.y * exp_arg1.x
    );

    // Component 2: the same, with the two components exchanging roles
    {{FP_TYPE}} arg_real2 = -(alpha2 * sat V_LOSS(V2, idx)) * dz;
    {{FP_TYPE}} arg_imag2 = (g22 * sq2 * sat + g12 * sq1 * sat V_PHASE(V2, idx)) * dz;
    {{FP_TYPE}} exp_real2 = exp{{FP_SUFFIX}}(arg_real2);
    {{FP_TYPE}} cos_imag2, sin_imag2;
    {{SINCOS_FUNC}}(arg_imag2, &sin_imag2, &cos_imag2);
    {{FP2_TYPE}} exp_arg2 = make_{{FP2_TYPE}}(exp_real2 * cos_imag2, exp_real2 * sin_imag2);
    A[idx + N_sq] = make_{{FP2_TYPE}}(
        a2.x * exp_arg2.x - a2.y * exp_arg2.y,
        a2.x * exp_arg2.y + a2.y * exp_arg2.x
    );
}
// {{END_VBLOCK}}

// Coupled RK4 nonlinear RHS on interleaved (2, N_sq) arrays: k += NL(A_orig)
// {{VBLOCK}}
extern "C" __global__ void coupled_rk4_nl_rhs_c(
    {{FP2_TYPE}}* k,
    const {{FP2_TYPE}}* A_orig,
    V_ARG(V1)
    V_ARG(V2)
    const {{FP_TYPE}} alpha1,
    const {{FP_TYPE}} alpha2,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} g22,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2,
    const int N_sq
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_sq) return;
    {{FP2_TYPE}} a1 = A_orig[idx];
    {{FP2_TYPE}} a2 = A_orig[idx + N_sq];
    {{FP_TYPE}} sq1 = a1.x * a1.x + a1.y * a1.y;
    {{FP_TYPE}} sq2 = a2.x * a2.x + a2.y * a2.y;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + sq1 / Isat1 + sq2 / Isat2);

    // Component 1: k += (1j*interact - alpha1*sat + 1j*V1) * a1
    // The contribution is rounded before it is accumulated, as the
    // one-component rk4_nl_rhs_c above does it: folding the two into one
    // expression lets --use_fast_math contract them into a different chain,
    // and the two paths stop agreeing bit for bit.
    {{FP_TYPE}} interact1_i = (g11 * sq1 + g12 * sq2) * sat;
    {{FP_TYPE}} coeff1_r = -(alpha1 * sat V_LOSS(V1, idx));
    {{FP_TYPE}} coeff1_i = interact1_i V_PHASE(V1, idx);
    {{FP_TYPE}} nl1_r = coeff1_r * a1.x - coeff1_i * a1.y;
    {{FP_TYPE}} nl1_i = coeff1_r * a1.y + coeff1_i * a1.x;
    {{FP2_TYPE}} k1_val = k[idx];
    k[idx] = make_{{FP2_TYPE}}(k1_val.x + nl1_r, k1_val.y + nl1_i);

    // Component 2: the same, with the two components exchanging roles
    {{FP_TYPE}} interact2_i = (g22 * sq2 + g12 * sq1) * sat;
    {{FP_TYPE}} coeff2_r = -(alpha2 * sat V_LOSS(V2, idx));
    {{FP_TYPE}} coeff2_i = interact2_i V_PHASE(V2, idx);
    {{FP_TYPE}} nl2_r = coeff2_r * a2.x - coeff2_i * a2.y;
    {{FP_TYPE}} nl2_i = coeff2_r * a2.y + coeff2_i * a2.x;
    {{FP2_TYPE}} k2_val = k[idx + N_sq];
    k[idx + N_sq] = make_{{FP2_TYPE}}(k2_val.x + nl2_r, k2_val.y + nl2_i);
}
// {{END_VBLOCK}}

// Rabi coupling on an interleaved (2, N_sq) field
extern "C" __global__ void rabi_coupling_interleaved(
    {{FP2_TYPE}}* A,
    const {{FP_TYPE}} cos_val,
    const {{FP_TYPE}} sin_val,
    const int N_sq
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N_sq) return;
    {{FP2_TYPE}} a1 = A[idx];
    {{FP2_TYPE}} a2 = A[idx + N_sq];
    // -1j * (x + iy) = (y, -x)
    A[idx] = make_{{FP2_TYPE}}(cos_val * a1.x + sin_val * a2.y,
                                cos_val * a1.y - sin_val * a2.x);
    A[idx + N_sq] = make_{{FP2_TYPE}}(cos_val * a2.x + sin_val * a1.y,
                                       cos_val * a2.y - sin_val * a1.x);
}

// Rabi coupling: 2x2 rotation of (A1, A2) pair
extern "C" __global__ void rabi_coupling(
    {{FP2_TYPE}}* A1,
    {{FP2_TYPE}}* A2,
    const {{FP_TYPE}} cos_val,
    const {{FP_TYPE}} sin_val,
    const int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    {{FP2_TYPE}} a1 = A1[idx];
    {{FP2_TYPE}} a2 = A2[idx];
    // -1j * (x + iy) = (y, -x)
    A1[idx] = make_{{FP2_TYPE}}(cos_val * a1.x + sin_val * a2.y,
                                 cos_val * a1.y - sin_val * a2.x);
    A2[idx] = make_{{FP2_TYPE}}(cos_val * a2.x + sin_val * a1.y,
                                 cos_val * a2.y - sin_val * a1.x);
}
