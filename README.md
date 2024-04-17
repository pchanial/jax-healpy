# Healpy with JAX

This project intends to assess the interest of implementing healpy functions using JAX.

*WARNING: BETA STAGE!!!*
<div align="center">
<img src="https://raw.githubusercontent.com/pchanial/jax-healpy/main/docs/benchmarks/chart-darkbackground-n10000000.png" alt="Benchmark"></img>
</div>

# Installation

    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install jax-healpy
    pip install "s2fft @ git+https://github.com/astro-informatics/s2fft@0.0.1"
