import numpy as np
from scipy.constants import c, epsilon_0

from NLSE import NLSE

if NLSE.__CUPY_AVAILABLE__:
    import cupy as cp
if NLSE.__PYOPENCL_AVAILABLE__:
    import pyopencl.array as cla
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
AVAILABLE_BACKENDS = ["CPU"]
if NLSE.__CUPY_AVAILABLE__:
    AVAILABLE_BACKENDS.append("CUPY")
if NLSE.__PYOPENCL_AVAILABLE__:
    AVAILABLE_BACKENDS.append("CL")

N = 2048
n2 = -1.6e-9
waist = 2.23e-3
waist2 = 70e-6
window = 4 * waist
power = 1.05
Isat = 10e4  # saturation intensity in W/m^2
L = 10e-3
alpha = 20


def test_build_propagator() -> None:
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        prop = simu._build_propagator()
        assert np.allclose(
            prop,
            np.exp(-1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z),
        ), f"Propagator is wrong. (Backend {backend})"


def test_build_fft_plan() -> None:
    from NLSE.backends import FFTPlan

    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        # Allocate on the right device
        A, _ = simu._prepare_output_array(
            np.random.random((N, N)).astype(PRECISION_COMPLEX)
            + 1j * np.random.random((N, N)).astype(PRECISION_COMPLEX),
            normalize=False,
        )
        plan = simu._build_fft_plan(A)
        assert isinstance(plan, FFTPlan), (
            f"Plan should be a FFTPlan instance. (Backend {backend})"
        )
        # Verify the plan can do a roundtrip FFT (CPU only, others have device arrays)
        if backend == "CPU":
            A_copy = A.copy()
            plan.fft(A_copy)
            plan.ifft(A_copy)
            assert np.allclose(A_copy, A, atol=1e-5), (
                f"FFT roundtrip failed. (Backend {backend})"
            )


def test_prepare_output_array() -> None:
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        if backend == "CPU" or backend == "CL":
            A = np.random.random((N, N)) + 1j * np.random.random((N, N))
        elif backend == "CUPY" and NLSE.__CUPY_AVAILABLE__:
            A = cp.random.random((N, N)) + 1j * cp.random.random((N, N))
        A = A.astype(PRECISION_COMPLEX)
        out, out_sq = simu._prepare_output_array(A, normalize=True)
        assert out.flags.c_contiguous, (
            f"Output array is not C-contiguous. (Backend {backend})"
        )
        assert out_sq.flags.c_contiguous, (
            f"Output array is not C-contiguous. (Backend {backend})"
        )
        if backend == "CPU":
            assert out.flags.aligned, (
                f"Output array is not aligned. (Backend {backend})"
            )
            assert out_sq.flags.aligned, (
                f"Output array is not aligned. (Backend {backend})"
            )
        if simu.backend == "CUPY" and NLSE.__CUPY_AVAILABLE__ or simu.backend == "CPU":
            integral = (
                (out.real * out.real + out.imag * out.imag)
                * simu.delta_X
                * simu.delta_Y
            ).sum(axis=simu._last_axes)
        if backend == "CL" and NLSE.__PYOPENCL_AVAILABLE__:
            arr = out.real * out.real + out.imag * out.imag
            arr = arr * simu.delta_X * simu.delta_Y
            integral = cla.sum(
                arr,
                dtype=arr.dtype,
                queue=simu._cl_queue,
            )
            integral = integral.get()
        integral = integral * c * epsilon_0 / 2
        error_string = f"Normalization failed. (Backend {backend})"
        error_string += f" : {integral} != {simu.power}"
        assert np.allclose(integral, simu.power), error_string
        assert out.shape == (
            N,
            N,
        ), f"Output array has wrong shape. (Backend {backend})"
        if backend == "CPU":
            assert isinstance(out, np.ndarray), (
                f"Output array type does not match backend. (Backend {backend})"
            )
            out /= np.max(np.abs(out))
            A /= np.max(np.abs(A))
            assert np.allclose(out, A), (
                f"Output array does not match input array. (Backend {backend})"
            )
        elif backend == "CUPY" and NLSE.__CUPY_AVAILABLE__:
            assert isinstance(out, cp.ndarray), (
                f"Output array type does not match backend. (Backend {backend})"
            )
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(out, A), (
                f"Output array does not match input array. (Backend {backend})"
            )


def test_send_arrays_to_gpu() -> None:
    if NLSE.__CUPY_AVAILABLE__:
        alpha_val = 20
        Isat_val = 10e4
        n2_val = -1.6e-9
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        alpha_arr = np.repeat(alpha_val, 2)  # type: ignore[arg-type]
        alpha_arr = alpha_arr[..., cp.newaxis, cp.newaxis]  # type: ignore[arg-type]
        n2_arr = np.repeat(n2_val, 2)  # type: ignore[arg-type]
        n2_arr = n2_arr[..., cp.newaxis, cp.newaxis]  # type: ignore[arg-type]
        Isat_arr = np.repeat(Isat_val, 2)  # type: ignore[arg-type]
        Isat_arr = Isat_arr[..., cp.newaxis, cp.newaxis]  # type: ignore[arg-type]
        simu = NLSE(
            alpha_arr, power, window, n2_arr, V, L, NX=N, NY=N, Isat=Isat_arr, backend="CUPY"  # type: ignore[arg-type]
        )
        simu.propagator = simu._build_propagator()
        simu._send_arrays_to_gpu()
        assert isinstance(simu.propagator, cp.ndarray), (
            "propagator is not a cp.ndarray. (Backend CUPY)"
        )
        assert isinstance(simu.V, cp.ndarray), "V is not a cp.ndarray. (Backend CUPY)"
        assert isinstance(simu.alpha, cp.ndarray), (
            "alpha is not a cp.ndarray. (Backend CUPY)"
        )
        assert isinstance(simu.n2, cp.ndarray), "n2 is not a cp.ndarray. (Backend CUPY)"
        assert isinstance(simu.I_sat, cp.ndarray), (
            "I_sat is not a cp.ndarray. (Backend CUPY)"
        )
    else:
        pass


def test_retrieve_arrays_from_gpu() -> None:
    if NLSE.__CUPY_AVAILABLE__:
        alpha_val = 20
        Isat_val = 10e4
        n2_val = -1.6e-9
        V = np.random.random((N, N)) + 1j * np.random.random((N, N))
        alpha_arr = np.repeat(alpha_val, 2)  # type: ignore[arg-type]
        alpha_arr = alpha_arr[..., cp.newaxis, cp.newaxis]  # type: ignore[arg-type]
        n2_arr = np.repeat(n2_val, 2)  # type: ignore[arg-type]
        n2_arr = n2_arr[..., cp.newaxis, cp.newaxis]  # type: ignore[arg-type]
        Isat_arr = np.repeat(Isat_val, 2)  # type: ignore[arg-type]
        Isat_arr = Isat_arr[..., cp.newaxis, cp.newaxis]  # type: ignore[arg-type]
        simu = NLSE(
            alpha_arr, power, window, n2_arr, V, L, NX=N, NY=N, Isat=Isat_arr, backend="CUPY"  # type: ignore[arg-type]
        )
        simu.propagator = simu._build_propagator()
        simu._send_arrays_to_gpu()
        simu._retrieve_arrays_from_gpu()
        assert isinstance(simu.propagator, np.ndarray), (
            "propagator is not a np.ndarray. (Backend CUPY)"
        )
        assert isinstance(simu.V, np.ndarray), "V is not a np.ndarray. (Backend CUPY)"
        assert isinstance(simu.alpha, np.ndarray), (
            "alpha is not a np.ndarray. (Backend CUPY)"
        )
        assert isinstance(simu.n2, np.ndarray), "n2 is not a np.ndarray. (Backend CUPY)"
        assert isinstance(simu.I_sat, np.ndarray), (
            "I_sat is not a np.ndarray. (Backend CUPY)"
        )
    else:
        pass


def test_split_step() -> None:
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            alpha,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        simu.delta_z = 0
        simu.propagator = simu._build_propagator()
        E = np.ones((N, N), dtype=PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E, normalize=False)
        simu.plans = simu._build_fft_plan(A)
        simu.propagator = simu._build_propagator()
        if backend == "CUPY" and NLSE.__CUPY_AVAILABLE__:
            E = cp.asarray(E)
        if (
            backend == "CUPY"
            and NLSE.__CUPY_AVAILABLE__
            or backend == "CL"
            and NLSE.__PYOPENCL_AVAILABLE__
        ):
            simu._send_arrays_to_gpu()
        simu.split_step(
            A, A_sq, simu.V, simu.propagator, simu.plans, precision="double"
        )
        if backend == "CPU":
            assert np.allclose(A, np.ones((N, N), dtype=PRECISION_COMPLEX)), (
                f"Split step is not unitary. (Backend {backend})"
            )
        elif backend == "CUPY" and NLSE.__CUPY_AVAILABLE__:
            assert cp.allclose(A, cp.ones((N, N), dtype=PRECISION_COMPLEX)), (
                f"Split step is not unitary. (Backend {backend})"
            )


# tests for convergence of the solver : the norm of the field should be
#  conserved
def test_out_field() -> None:
    E = np.ones((N, N), dtype=PRECISION_COMPLEX)
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE(
            0,
            power,
            window,
            n2,
            None,
            L,
            NX=N,
            NY=N,
            Isat=Isat,
            backend=backend,
        )
        E = simu.out_field(E, L, verbose=False, plot=False, precision="single")
        norm = np.sum(np.abs(E) ** 2 * simu.delta_X * simu.delta_Y)
        norm *= c * epsilon_0 / 2
        assert E.shape == (
            N,
            N,
        ), f"Output array has wrong shape. (Backend {backend})"
        assert np.allclose(norm, simu.power, rtol=1e-4), (
            f"Norm not conserved. (Backend {backend})"
        )
