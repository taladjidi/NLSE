// CUDA C kernels for NLSE operations (optimized fused versions only)
// Precision-agnostic template using {{FP_TYPE}}, {{FP2_TYPE}}, {{FP_SUFFIX}}, {{SINCOS_FUNC}}
//
// A kernel that reads a potential is written once, between the VBLOCK markers
// and against the V_ARG/V_PHASE/V_LOSS macros, and compiled three times: as
// <name> with no potential, <name>_v with a real one and <name>_cv with a
// complex (absorbing) one. See _expand_v_blocks in kernels/cupy_kernels.py.

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

    // Apply nonlinear propagation immediately
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq_val / Isat);
    {{FP_TYPE}} arg_real = -(alpha * sat V_LOSS(V, idx)) * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq_val * sat V_PHASE(V, idx)) * dz;
    {{FP_TYPE}} exp_real_part = exp{{FP_SUFFIX}}(arg_real);
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
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq[idx] / Isat);
    {{FP_TYPE}} arg_real = -(alpha * sat V_LOSS(V, idx)) * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq[idx] * sat V_PHASE(V, idx)) * dz;
    {{FP_TYPE}} exp_real_part = exp{{FP_SUFFIX}}(arg_real);
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
