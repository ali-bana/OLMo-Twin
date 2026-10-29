# CPU-host shim: raising ImportError here makes deepspeed set
# HAS_TRITON=False and skip its triton inference kernels, whose module-level
# autotuner crashes with "0 active drivers" on any machine without a GPU
# driver. Prepend this directory to PYTHONPATH ONLY for CPU utility runs
# (dataset caching, checkpoint conversion) -- never for training jobs.
raise ImportError("triton disabled by cpu_shims (no GPU driver on this node)")
