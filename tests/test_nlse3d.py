import numpy as np
from scipy.constants import c, epsilon_0

from NLSE import NLSE_3d

if NLSE_3d.__CUPY_AVAILABLE__:
    import cupy as cp
PRECISION_COMPLEX = np.complex64
PRECISION_REAL = np.float32
AVAILABLE_BACKENDS = ["CPU"]
if NLSE_3d.__CUPY_AVAILABLE__:
    AVAILABLE_BACKENDS.append("CUPY")
# TODO
# if NLSE_3d.__PYOPENCL_AVAILABLE__:
#     AVAILABLE_BACKENDS.append("CL")

N = 256
NZ = 128
n2 = -1.6e-9
D0 = 1e-16
vg = 1e-1 * c
waist = 2.23e-3
duration = 2e-6
waist2 = 70e-6
window = np.array([4 * waist, 8 * duration])  # 4*waist transverse, 10e-6 s temporal
energy = 1.05 * duration
Isat = 10e4  # saturation intensity in W/m^2
L = 1e-2
alpha = 20


def test_build_propagator() -> None:
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE_3d(
            alpha=alpha,
            energy=energy,
            window=window,
            n2=n2,
            D0=D0,
            vg=vg,
            V=None,
            L=L,
            NX=N,
            NY=N,
            NZ=NZ,
            Isat=Isat,
            backend=backend,
        )
        prop = simu._build_propagator()
        prop_th = np.exp(
            -1j * 0.5 * (simu.Kxx**2 + simu.Kyy**2) / simu.k * simu.delta_z
        )
        prop_th *= np.exp(-1j * simu.D0 / 2 * simu.Omega**2)
        assert np.allclose(
            prop,
            prop_th,
        ), f"Propagator is wrong. (Backend {backend})"


def test_build_fft_plan() -> None:
    from NLSE.backends import FFTPlan

    for backend in AVAILABLE_BACKENDS:
        simu = NLSE_3d(
            alpha=alpha,
            energy=energy,
            window=window,
            n2=n2,
            D0=D0,
            vg=vg,
            V=None,
            L=L,
            NX=N,
            NY=N,
            NZ=NZ,
            Isat=Isat,
            backend=backend,
        )
        # Allocate on the right device
        A, _ = simu._prepare_output_array(
            np.random.random((N, N, NZ)).astype(PRECISION_COMPLEX)
            + 1j * np.random.random((N, N, NZ)).astype(PRECISION_COMPLEX),
            normalize=False,
        )
        plan = simu._build_fft_plan(A)
        assert isinstance(plan, FFTPlan), (
            f"Plan should be a FFTPlan instance. (Backend {backend})"
        )
        if backend == "CPU":
            A_copy = A.copy()
            plan.fft(A_copy)
            plan.ifft(A_copy)
            assert np.allclose(A_copy, A, atol=1e-5), (
                f"FFT roundtrip failed. (Backend {backend})"
            )


def test_prepare_output_array() -> None:
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE_3d(
            alpha=alpha,
            energy=energy,
            window=window,
            n2=n2,
            D0=D0,
            vg=vg,
            V=None,
            L=L,
            NX=N,
            NY=N,
            NZ=NZ,
            Isat=Isat,
            backend=backend,
        )
        if backend == "CPU":
            A = np.random.random((N, N, NZ)) + 1j * np.random.random((N, N, NZ))
        elif backend == "CUPY" and NLSE_3d.__CUPY_AVAILABLE__:
            A = cp.random.random((N, N, NZ)) + 1j * cp.random.random((N, N, NZ))
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
        integral = (
            (out.real * out.real + out.imag * out.imag)
            * simu.delta_X
            * simu.delta_Y
            * simu.delta_T
        ).sum(axis=simu._last_axes)
        integral *= c * epsilon_0 / 2
        assert np.allclose(integral, simu.energy), (
            f"Normalization failed. (Backend {backend})"
        )
        assert out.shape == (
            N,
            N,
            NZ,
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
        elif backend == "CUPY" and NLSE_3d.__CUPY_AVAILABLE__:
            assert isinstance(out, cp.ndarray), (
                f"Output array type does not match backend. (Backend {backend})"
            )
            out /= cp.max(cp.abs(out))
            A /= cp.max(cp.abs(A))
            assert cp.allclose(out, A), (
                f"Output array does not match input array. (Backend {backend})"
            )


def test_send_arrays_to_gpu() -> None:
    if NLSE_3d.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = np.random.random((N, N, NZ)) + 1j * np.random.random((N, N, NZ))
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis]
        simu = NLSE_3d(
            alpha=alpha,
            energy=energy,
            window=window,
            n2=n2,
            D0=D0,
            vg=vg,
            V=V,
            L=L,
            NX=N,
            NY=N,
            NZ=NZ,
            Isat=Isat,
            backend="CUPY",
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
    if NLSE_3d.__CUPY_AVAILABLE__:
        alpha = 20
        Isat = 10e4
        n2 = -1.6e-9
        V = np.random.random((N, N, NZ)) + 1j * np.random.random((N, N, NZ))
        alpha = np.repeat(alpha, 2)
        alpha = alpha[..., np.newaxis, np.newaxis]
        n2 = np.repeat(n2, 2)
        n2 = n2[..., np.newaxis, np.newaxis]
        Isat = np.repeat(Isat, 2)
        Isat = Isat[..., np.newaxis, np.newaxis]
        simu = NLSE_3d(
            alpha=alpha,
            energy=energy,
            window=window,
            n2=n2,
            D0=D0,
            vg=vg,
            V=V,
            L=L,
            NX=N,
            NY=N,
            NZ=NZ,
            Isat=Isat,
            backend="CUPY",
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
        simu = NLSE_3d(
            alpha=alpha,
            energy=energy,
            window=window,
            n2=n2,
            D0=D0,
            vg=vg,
            V=None,
            L=L,
            NX=N,
            NY=N,
            NZ=NZ,
            Isat=Isat,
            backend=backend,
        )
        simu.delta_z = 0
        simu.propagator = simu._build_propagator()
        E = np.ones((N, N, NZ), dtype=PRECISION_COMPLEX)
        A, A_sq = simu._prepare_output_array(E, normalize=False)
        simu.plans = simu._build_fft_plan(A)
        simu.propagator = simu._build_propagator()
        if backend == "CUPY" and NLSE_3d.__CUPY_AVAILABLE__:
            E = cp.asarray(E)
            simu._send_arrays_to_gpu()
        simu.split_step(
            E, A_sq, simu.V, simu.propagator, simu.plans, precision="double"
        )
        if backend == "CPU":
            assert np.allclose(E, np.ones((N, N, NZ), dtype=PRECISION_COMPLEX)), (
                f"Split step is not unitary. (Backend {backend})"
            )
        elif backend == "CUPY" and NLSE_3d.__CUPY_AVAILABLE__:
            assert cp.allclose(E, cp.ones((N, N, NZ), dtype=PRECISION_COMPLEX)), (
                f"Split step is not unitary. (Backend {backend})"
            )


# tests for convergence of the solver : the norm of the field should be
# conserved
def test_out_field() -> None:
    for backend in AVAILABLE_BACKENDS:
        simu = NLSE_3d(
            alpha=alpha,
            energy=energy,
            window=window,
            n2=n2,
            D0=D0,
            vg=vg,
            V=None,
            L=L,
            NX=N,
            NY=N,
            NZ=NZ,
            Isat=Isat,
            backend=backend,
        )
        E0 = np.ones((N, N, NZ), dtype=PRECISION_COMPLEX)
        E = simu.out_field(
            E0, simu.delta_z, verbose=False, plot=False, precision="single"
        )
        norm = np.sum(np.abs(E) ** 2 * simu.delta_X * simu.delta_Y * simu.delta_T)
        norm *= c * epsilon_0 / 2
        assert E.shape == (
            N,
            N,
            NZ,
        ), f"Output array has wrong shape. (Backend {backend})"
        assert np.allclose(norm, simu.energy, rtol=1e-4), (
            f"Norm not conserved. (Backend {backend})"
        )
