// OpenCL C kernels for NLSE operations (optimized fused versions only)
// Precision-agnostic template using {{FP_TYPE}}, {{FP2_TYPE}}, {{FP_SUFFIX}}

// FUSED: square_mod + nl_prop_without_V
// Combines square modulus calculation and nonlinear propagation in single kernel
// Eliminates kernel launch overhead and one memory pass
__kernel void square_mod_nl_prop_fused(
    __global {{FP2_TYPE}}* A,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);

    // Compute square modulus inline
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;

    // Apply nonlinear propagation immediately (no temporary arrays!)
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_val / Isat);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = g * A_sq_val * sat * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);

    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}

// FUSED: square_mod + nl_prop (with potential)
__kernel void square_mod_nl_prop_v_fused(
    __global {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);

    // Compute square modulus inline
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;

    // Apply nonlinear propagation immediately
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_val / Isat);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq_val * sat + V[idx]) * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);

    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}

// Propagator multiplication (replaces slow PyOpenCL array expression)
__kernel void apply_propagator(
    __global {{FP2_TYPE}}* A,
    __global const {{FP2_TYPE}}* propagator
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    {{FP2_TYPE}} prop_val = propagator[idx];

    // Complex multiplication: A *= propagator
    A[idx] = ({{FP2_TYPE}})(
        A_val.x * prop_val.x - A_val.y * prop_val.y,
        A_val.x * prop_val.y + A_val.y * prop_val.x
    );
}

// SEPARATE KERNELS (required when nl_length > 0 or for coupled solvers)
// Convolution between square_mod and nl_prop requires separate kernel calls

// Square modulus computation
__kernel void square_mod_fused(
    __global const {{FP2_TYPE}}* A,
    __global {{FP_TYPE}}* A_sq
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    A_sq[idx] = A_val.x * A_val.x + A_val.y * A_val.y;
}

// Nonlinear propagation without potential
__kernel void nl_prop_without_v_fused(
    __global {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = g * A_sq[idx] * sat * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A_val = A[idx];
    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}

// Nonlinear propagation with potential
__kernel void nl_prop_fused(
    __global {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq[idx] * sat + V[idx]) * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A_val = A[idx];
    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}

// Coupled nonlinear propagation with potential (for CNLSE, DDGPE)
__kernel void nl_prop_c_fused(
    __global {{FP2_TYPE}}* A1,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat + V[idx]) * dz;
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

// Coupled nonlinear propagation without potential (for CNLSE, DDGPE)
__kernel void nl_prop_c_without_v_fused(
    __global {{FP2_TYPE}}* A1,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat) * dz;
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

// RK4 nonlinear RHS kernels (additive, no exp)
// These accumulate onto A_prop: A_prop += (nonlinear terms) * A

// RK4 NL RHS without potential
__kernel void rk4_nl_rhs_fused(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    // coeff = -alpha*sat + 1j*g*A_sq*sat
    {{FP_TYPE}} coeff_r = -alpha * sat;
    {{FP_TYPE}} coeff_i = g * A_sq[idx] * sat;
    // contrib = coeff * A
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}

// RK4 NL RHS with potential
__kernel void rk4_nl_rhs_v_fused(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    // coeff = -alpha*sat + 1j*(g*A_sq*sat + V)
    {{FP_TYPE}} coeff_r = -alpha * sat;
    {{FP_TYPE}} coeff_i = g * A_sq[idx] * sat + V[idx];
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}

// FUSED: |A|^2 + RK4 NL RHS without potential
__kernel void square_mod_rk4_nl_rhs_fused(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_val / Isat);
    {{FP_TYPE}} coeff_r = -alpha * sat;
    {{FP_TYPE}} coeff_i = g * A_sq_val * sat;
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}

// FUSED: |A|^2 + RK4 NL RHS with potential
__kernel void square_mod_rk4_nl_rhs_v_fused(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    {{FP_TYPE}} A_sq_val = A_val.x * A_val.x + A_val.y * A_val.y;
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_val / Isat);
    {{FP_TYPE}} coeff_r = -alpha * sat;
    {{FP_TYPE}} coeff_i = g * A_sq_val * sat + V[idx];
    {{FP_TYPE}} contrib_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} contrib_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(A_prop_val.x + contrib_r, A_prop_val.y + contrib_i);
}

// Coupled RK4 NL RHS without potential
// Interaction NOT multiplied by A_orig (matches CNLSE RK4 math)
__kernel void rk4_nl_rhs_c_fused(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A_orig,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    // Interaction: pure imaginary, NOT multiplied by A_orig
    {{FP_TYPE}} interact_i = (g11 * A_sq_1[idx] + g12 * A_sq_2[idx]) * sat;
    // Loss: -alpha*sat * A_orig (complex)
    {{FP2_TYPE}} A_val = A_orig[idx];
    {{FP_TYPE}} loss_coeff = alpha * sat;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(
        A_prop_val.x - loss_coeff * A_val.x,
        A_prop_val.y + interact_i - loss_coeff * A_val.y
    );
}

// Coupled RK4 NL RHS with potential
__kernel void rk4_nl_rhs_c_v_fused(
    __global {{FP2_TYPE}}* A_prop,
    __global const {{FP2_TYPE}}* A_orig,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    // Interaction: pure imaginary
    {{FP_TYPE}} interact_i = (g11 * A_sq_1[idx] + g12 * A_sq_2[idx]) * sat;
    // Loss + potential: (-alpha*sat + 1j*V) * A_orig
    {{FP2_TYPE}} A_val = A_orig[idx];
    {{FP_TYPE}} coeff_r = -alpha * sat;
    {{FP_TYPE}} coeff_i = V[idx];
    {{FP_TYPE}} lv_r = coeff_r * A_val.x - coeff_i * A_val.y;
    {{FP_TYPE}} lv_i = coeff_r * A_val.y + coeff_i * A_val.x;
    {{FP2_TYPE}} A_prop_val = A_prop[idx];
    A_prop[idx] = ({{FP2_TYPE}})(
        A_prop_val.x + lv_r,
        A_prop_val.y + interact_i + lv_i
    );
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
