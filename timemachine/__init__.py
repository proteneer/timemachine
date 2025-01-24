def _suppress_jax_no_gpu_warning():
    """Suppresses the following warning:

       WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)

    See https://github.com/google/jax/issues/6805
    """
    import jax

    jax.config.update("jax_platform_name", "cpu")


_suppress_jax_no_gpu_warning()
