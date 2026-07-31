// OpenCL C kernels for NLSE operations (optimized fused versions only)
// Precision-agnostic template using {{FP_TYPE}}, {{FP2_TYPE}}, {{FP_SUFFIX}}
//
// A kernel that reads a potential is written once, between the VBLOCK markers
// and against the V_ARG/V_PHASE/V_LOSS macros, and compiled three times: as
// <name> with no potential, <name>_v with a real one and <name>_cv with a
// complex (absorbing) one. See _expand_v_blocks in kernels/cl.py.

// The saturation the interaction should be evaluated at over a lossy step.
// The CUDA twin of this, in cuda_source/kernels.cu, carries the derivation;
// briefly: freezing |A|^2 across a step is exact only while the step preserves
// it, which loss does not, and the frozen step costs every composition built
// on it its order. With u = 2*alpha*dz this returns P = (1 - y_end/y0)/u, so
// the step applies g*y0*P*dz for the phase and sqrt(1 - P*u) for the
// amplitude. u == 0 returns early and a lossless run is unchanged to the bit.
inline {{FP_TYPE}} nlse_loss_factor(const {{FP_TYPE}} sat, const {{FP_TYPE}} u) {
    if (u == 0.0{{FP_SUFFIX}}) return sat;
    {{FP_TYPE}} P = sat;
    for (int pass = 0; pass < 3; ++pass) {
        {{FP_TYPE}} Pu = P * u;
        P = sat * (1.0{{FP_SUFFIX}} - Pu * P * (0.5{{FP_SUFFIX}}
            + Pu / 3.0{{FP_SUFFIX}} + Pu * Pu * 0.25{{FP_SUFFIX}}));
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
// Combines square modulus calculation and nonlinear propagation in single kernel
// Eliminates kernel launch overhead and one memory pass
// {{VBLOCK}}
__kernel void square_mod_nl_prop(
    __global {{FP2_TYPE}}* A,
    V_ARG(V)
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);

    // Compute square modulus inline
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;

    // Apply nonlinear propagation immediately. The loss has left arg_real for
    // exp_real_part, where the step is solved rather than frozen; what remains
    // there is a complex potential's own absorption, zero in the other twins.
    {{FP_TYPE}} u = 2.0{{FP_SUFFIX}} * alpha * dz;
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_val / Isat);
    // Solved, or frozen as before where solving it is out of reach. Both
    // branches are uniform across the grid: u is a scalar.
    bool solved = u <= ({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT
        && u >= -({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT;
    {{FP_TYPE}} P = solved ? nlse_loss_factor(sat, u) : sat;
    {{FP_TYPE}} alpha_left = solved ? 0.0{{FP_SUFFIX}} : alpha;
    {{FP_TYPE}} amp = solved ? sqrt(1.0{{FP_SUFFIX}} - P * u) : 1.0{{FP_SUFFIX}};
    {{FP_TYPE}} arg_real = -(alpha_left * sat V_LOSS(V, idx - (int)get_global_offset(0))) * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq_val * P V_PHASE(V, idx - (int)get_global_offset(0))) * dz;
    {{FP_TYPE}} exp_real_part = amp * exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);

    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
// {{END_VBLOCK}}

// Propagator multiplication (replaces slow PyOpenCL array expression)
__kernel void apply_propagator(
    __global {{FP2_TYPE}}* A,
    __global const {{FP2_TYPE}}* propagator
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    {{FP2_TYPE}} prop_val = propagator[idx - (int)get_global_offset(0)];

    // Complex multiplication: A *= propagator
    A[idx] = ({{FP2_TYPE}})(
        A_val.x * prop_val.x - A_val.y * prop_val.y,
        A_val.x * prop_val.y + A_val.y * prop_val.x
    );
}

// SEPARATE KERNELS (required when nl_length > 0 or for coupled solvers)
// Convolution between square_mod and nl_prop requires separate kernel calls

// Square modulus computation
__kernel void square_mod(
    __global const {{FP2_TYPE}}* A,
    __global {{FP_TYPE}}* A_sq
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    A_sq[idx] = A_val.x * A_val.x + A_val.y * A_val.y;
}

// Nonlinear propagation
// {{VBLOCK}}
__kernel void nl_prop(
    __global {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    V_ARG(V)
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    // See square_mod_nl_prop above, and nlse_loss_factor for why P and not sat.
    {{FP_TYPE}} u = 2.0{{FP_SUFFIX}} * alpha * dz;
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    // Solved, or frozen as before where solving it is out of reach. Both
    // branches are uniform across the grid: u is a scalar.
    bool solved = u <= ({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT
        && u >= -({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT;
    {{FP_TYPE}} P = solved ? nlse_loss_factor(sat, u) : sat;
    {{FP_TYPE}} alpha_left = solved ? 0.0{{FP_SUFFIX}} : alpha;
    {{FP_TYPE}} amp = solved ? sqrt(1.0{{FP_SUFFIX}} - P * u) : 1.0{{FP_SUFFIX}};
    {{FP_TYPE}} arg_real = -(alpha_left * sat V_LOSS(V, idx - (int)get_global_offset(0))) * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq[idx] * P V_PHASE(V, idx - (int)get_global_offset(0))) * dz;
    {{FP_TYPE}} exp_real_part = amp * exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A_val = A[idx];
    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
// {{END_VBLOCK}}

// Coupled nonlinear propagation (for CNLSE, DDGPE)
// {{VBLOCK}}
__kernel void nl_prop_c(
    __global {{FP2_TYPE}}* A1,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    V_ARG(V)
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    {{FP_TYPE}} arg_real = -(alpha * sat V_LOSS(V, idx)) * dz;
    {{FP_TYPE}} arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat V_PHASE(V, idx)) * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A1_val = A1[idx];
    A1[idx] = ({{FP2_TYPE}})(
        A1_val.x * exp_arg.x - A1_val.y * exp_arg.y,
        A1_val.x * exp_arg.y + A1_val.y * exp_arg.x
    );
}
// {{END_VBLOCK}}

// RK4 utility kernels (stage building and accumulation)

// RK4 AXPY: out = A + c * k
__kernel void rk4_axpy(
    __global {{FP2_TYPE}}* out,
    __global const {{FP2_TYPE}}* A,
    const {{FP_TYPE}} c,
    __global const {{FP2_TYPE}}* k
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    {{FP2_TYPE}} k_val = k[idx];
    out[idx] = ({{FP2_TYPE}})(A_val.x + c * k_val.x, A_val.y + c * k_val.y);
}

// RK4 Accumulate: acc += w * k
__kernel void rk4_accumulate(
    __global {{FP2_TYPE}}* acc,
    const {{FP_TYPE}} w,
    __global const {{FP2_TYPE}}* k
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} acc_val = acc[idx];
    {{FP2_TYPE}} k_val = k[idx];
    acc[idx] = ({{FP2_TYPE}})(acc_val.x + w * k_val.x, acc_val.y + w * k_val.y);
}

// RK4 nonlinear RHS kernels (additive, no exp)
// These accumulate onto A_prop: A_prop += (nonlinear terms) * A

// RK4 NL RHS
// {{VBLOCK}}
__kernel void rk4_nl_rhs(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    V_ARG(V)
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    // coeff = -alpha*sat + 1j*(g*A_sq*sat + V)
    {{FP_TYPE}} coeff_r = -(alpha * sat V_LOSS(V, idx - (int)get_global_offset(0)));
    {{FP_TYPE}} coeff_i = g * A_sq[idx] * sat V_PHASE(V, idx - (int)get_global_offset(0));
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}
// {{END_VBLOCK}}

// FUSED: |A|^2 + RK4 NL RHS
// {{VBLOCK}}
__kernel void square_mod_rk4_nl_rhs(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A,
    V_ARG(V)
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_val / Isat);
    {{FP_TYPE}} coeff_r = -(alpha * sat V_LOSS(V, idx - (int)get_global_offset(0)));
    {{FP_TYPE}} coeff_i = g * A_sq_val * sat V_PHASE(V, idx - (int)get_global_offset(0));
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}
// {{END_VBLOCK}}

// Coupled RK4 NL RHS
// Interaction NOT multiplied by A_orig (matches CNLSE RK4 math)
// {{VBLOCK}}
__kernel void rk4_nl_rhs_c(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A_orig,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    V_ARG(V)
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    // NL coefficient: (1j*interact - alpha*sat + 1j*V) * A_orig
    {{FP_TYPE}} interact_i = (g11 * A_sq_1[idx] + g12 * A_sq_2[idx]) * sat;
    {{FP2_TYPE}} A_val = A_orig[idx];
    {{FP_TYPE}} coeff_r = -(alpha * sat V_LOSS(V, idx));
    {{FP_TYPE}} coeff_i = interact_i V_PHASE(V, idx);
    {{FP_TYPE}} nl_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} nl_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(
        A_prop_val.x + nl_r,
        A_prop_val.y + nl_i
    );
}
// {{END_VBLOCK}}

// FUSED RK4 stage update: acc = k, out = A + c * k  (used for stage 1)
// Combines the copy-to-acc and axpy-to-A_tmp into a single kernel launch.
__kernel void rk4_set_and_axpy(
    __global {{FP2_TYPE}}* acc,
    __global {{FP2_TYPE}}* out,
    __global const {{FP2_TYPE}}* A,
    __global const {{FP2_TYPE}}* k,
    const {{FP_TYPE}} c
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} k_val = k[idx];
    acc[idx] = k_val;
    {{FP2_TYPE}} A_val = A[idx];
    out[idx] = ({{FP2_TYPE}})(A_val.x + c * k_val.x, A_val.y + c * k_val.y);
}

// FUSED RK4 stage update: acc += w * k, out = A + c * k  (used for stages 2-3)
// Combines accumulate-to-acc and axpy-to-A_tmp into a single kernel launch.
__kernel void rk4_acc_and_axpy(
    __global {{FP2_TYPE}}* acc,
    __global {{FP2_TYPE}}* out,
    __global const {{FP2_TYPE}}* A,
    __global const {{FP2_TYPE}}* k,
    const {{FP_TYPE}} w,
    const {{FP_TYPE}} c
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} k_val = k[idx];
    {{FP2_TYPE}} acc_val = acc[idx];
    acc[idx] = ({{FP2_TYPE}})(acc_val.x + w * k_val.x, acc_val.y + w * k_val.y);
    {{FP2_TYPE}} A_val = A[idx];
    out[idx] = ({{FP2_TYPE}})(A_val.x + c * k_val.x, A_val.y + c * k_val.y);
}

// Rabi coupling: 2x2 rotation of (A1, A2) pair
__kernel void rabi_coupling(
    __global {{FP2_TYPE}}* A1,
    __global {{FP2_TYPE}}* A2,
    const {{FP_TYPE}} cos_val,
    const {{FP_TYPE}} sin_val
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} a1 = A1[idx];
    {{FP2_TYPE}} a2 = A2[idx];
    // -1j * (x + iy) = (y, -x)
    A1[idx] = ({{FP2_TYPE}})(cos_val * a1.x + sin_val * a2.y,
                              cos_val * a1.y - sin_val * a2.x);
    A2[idx] = ({{FP2_TYPE}})(cos_val * a2.x + sin_val * a1.y,
                              cos_val * a2.y - sin_val * a1.x);
}

// ── Interleaved coupled kernels ─────────────────────────────────────────────
// These process both components from a (2, N_sq) layout in a single kernel.
// Each thread at idx processes A[idx] (comp 1) and A[idx + N_sq] (comp 2).
// The saturation factor 1/(1 + |A1|^2/Isat1 + |A2|^2/Isat2) is computed once.

// Coupled NL propagation on interleaved (2, N_sq) array
// {{VBLOCK}}
__kernel void coupled_nl_prop_c(
    __global {{FP2_TYPE}}* A,
    V_ARG(V1)
    V_ARG(V2)
    const int N_sq,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha1,
    const {{FP_TYPE}} alpha2,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} g22,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} a1 = A[idx];
    {{FP2_TYPE}} a2 = A[idx + N_sq];
    {{FP_TYPE}} sq1 = a1.x * a1.x + a1.y * a1.y;
    {{FP_TYPE}} sq2 = a2.x * a2.x + a2.y * a2.y;
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + sq1 / Isat1 + sq2 / Isat2);

    // Component 1
    {{FP_TYPE}} r1 = -(alpha1 * sat V_LOSS(V1, idx)) * dz;
    {{FP_TYPE}} i1 = (g11 * sq1 * sat + g12 * sq2 * sat V_PHASE(V1, idx)) * dz;
    {{FP_TYPE}} e1r = exp(r1);
    {{FP_TYPE}} c1, s1;
    s1 = sincos(i1, &c1);
    A[idx] = ({{FP2_TYPE}})(
        a1.x * e1r * c1 - a1.y * e1r * s1,
        a1.x * e1r * s1 + a1.y * e1r * c1
    );

    // Component 2
    {{FP_TYPE}} r2 = -(alpha2 * sat V_LOSS(V2, idx)) * dz;
    {{FP_TYPE}} i2 = (g22 * sq2 * sat + g12 * sq1 * sat V_PHASE(V2, idx)) * dz;
    {{FP_TYPE}} e2r = exp(r2);
    {{FP_TYPE}} c2, s2;
    s2 = sincos(i2, &c2);
    A[idx + N_sq] = ({{FP2_TYPE}})(
        a2.x * e2r * c2 - a2.y * e2r * s2,
        a2.x * e2r * s2 + a2.y * e2r * c2
    );
}
// {{END_VBLOCK}}

// Coupled RK4 NL RHS on interleaved (2, N_sq) arrays
// k += NL(A_orig) for both components
// {{VBLOCK}}
__kernel void coupled_rk4_nl_rhs_c(
    __global {{FP2_TYPE}}* k,
    __global const {{FP2_TYPE}}* A_orig,
    V_ARG(V1)
    V_ARG(V2)
    const int N_sq,
    const {{FP_TYPE}} alpha1,
    const {{FP_TYPE}} alpha2,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} g22,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} a1 = A_orig[idx];
    {{FP2_TYPE}} a2 = A_orig[idx + N_sq];
    {{FP_TYPE}} sq1 = a1.x * a1.x + a1.y * a1.y;
    {{FP_TYPE}} sq2 = a2.x * a2.x + a2.y * a2.y;
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + sq1 / Isat1 + sq2 / Isat2);

    // Component 1: k += (1j*interact - alpha1*sat + 1j*V1) * a1
    {{FP_TYPE}} interact1_i = (g11 * sq1 + g12 * sq2) * sat;
    {{FP_TYPE}} coeff1_r = -(alpha1 * sat V_LOSS(V1, idx));
    {{FP_TYPE}} coeff1_i = interact1_i V_PHASE(V1, idx);
    {{FP_TYPE}} nl1_r = coeff1_r * a1.x - coeff1_i * a1.y;
    {{FP_TYPE}} nl1_i = coeff1_r * a1.y + coeff1_i * a1.x;
    {{FP2_TYPE}} k1_val = k[idx];
    k[idx] = ({{FP2_TYPE}})(k1_val.x + nl1_r, k1_val.y + nl1_i);

    // Component 2: same structure with swapped params
    {{FP_TYPE}} interact2_i = (g22 * sq2 + g12 * sq1) * sat;
    {{FP_TYPE}} coeff2_r = -(alpha2 * sat V_LOSS(V2, idx));
    {{FP_TYPE}} coeff2_i = interact2_i V_PHASE(V2, idx);
    {{FP_TYPE}} nl2_r = coeff2_r * a2.x - coeff2_i * a2.y;
    {{FP_TYPE}} nl2_i = coeff2_r * a2.y + coeff2_i * a2.x;
    {{FP2_TYPE}} k2_val = k[idx + N_sq];
    k[idx + N_sq] = ({{FP2_TYPE}})(k2_val.x + nl2_r, k2_val.y + nl2_i);
}
// {{END_VBLOCK}}

// Rabi coupling on interleaved (2, N_sq) array
__kernel void rabi_coupling_interleaved(
    __global {{FP2_TYPE}}* A,
    const int N_sq,
    const {{FP_TYPE}} cos_val,
    const {{FP_TYPE}} sin_val
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} a1 = A[idx];
    {{FP2_TYPE}} a2 = A[idx + N_sq];
    // -1j * (x + iy) = (y, -x)
    A[idx] = ({{FP2_TYPE}})(cos_val * a1.x + sin_val * a2.y,
                              cos_val * a1.y - sin_val * a2.x);
    A[idx + N_sq] = ({{FP2_TYPE}})(cos_val * a2.x + sin_val * a1.y,
                                     cos_val * a2.y - sin_val * a1.x);
}
