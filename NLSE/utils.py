import warnings

__BACKEND__ = "CUPY"
__PYOPENCL_DOUBLE_SUPPORT__ = False


try:
    import cupy

    __CUPY_AVAILABLE__ = True

except ImportError:
    warnings.warn(
        "CuPy not available, falling back to CPU backend. "
        "Install cupy for GPU acceleration.",
        ImportWarning,
        stacklevel=2,
    )
    __CUPY_AVAILABLE__ = False
    __BACKEND__ = "CPU"


try:
    # for OpenCL backend you need to install OpenCL first
    # sudo apt install intel-opencl-icd opencl-headers ocl-icd-opencl-dev
    # or for AMD
    # sudo apt install opencl-headers ocl-icd-opencl-dev
    import pyopencl

    __PYOPENCL_AVAILABLE__ = True

    # Check for double precision support
    try:
        import pyopencl as cl

        ctx = cl.create_some_context(interactive=False)
        device = ctx.devices[0]
        __PYOPENCL_DOUBLE_SUPPORT__ = bool(device.double_fp_config)
    except Exception:
        __PYOPENCL_DOUBLE_SUPPORT__ = False

except ImportError:
    warnings.warn(
        "PyOpenCL not available, OpenCL backend unavailable. "
        "Install pyopencl for OpenCL support.",
        ImportWarning,
        stacklevel=2,
    )
    __PYOPENCL_AVAILABLE__ = False
