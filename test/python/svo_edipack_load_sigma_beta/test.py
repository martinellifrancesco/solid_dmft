"""
Test that load_sigma with a different beta correctly transforms the self-energy
via DLR interpolation when starting from a T=0 (zerotemp) edipack calculation.

Run 1: edipack at T=0 (zerotemp=True), beta=600 controls only the Matsubara spacing.
Run 2: edipack at finite T, beta=400, loading and transforming sigma from run 1.

NOTE: requires PKG_CONFIG_PATH to include the edipack library directory, e.g.:
  export PKG_CONFIG_PATH=/home/fmartinelli/miniconda3/envs/triqs_test/lib/pkgconfig
"""
import shutil
import numpy as np
from h5 import HDFArchive
import triqs.utility.mpi as mpi

import solid_dmft.main as solid

if mpi.is_master_node():
    shutil.rmtree('out', ignore_errors=True)

mpi.barrier()

# Run 1: T=0 edipack, beta=600 sets the Matsubara frequency spacing — writes to out/inp.h5
solid.main([None, 'dmft_config_run1.toml'])

mpi.barrier()

# Run 2: finite-T edipack, beta=400, load_sigma=true — reads sigma from same out/inp.h5, transforms via DLR
solid.main([None, 'dmft_config_run2.toml'])

mpi.barrier()

if mpi.is_master_node():
    with HDFArchive('out/inp.h5', 'r') as ar:
        last_iter = ar['DMFT_results']['last_iter']
        sigma = last_iter['Sigma_freq_0']
        for block, gf in sigma:
            data = gf.data
            assert np.all(np.isfinite(data)), \
                f"Sigma_freq_0[{block}] contains non-finite values after beta change"
            assert np.any(np.abs(data) > 1e-10), \
                f"Sigma_freq_0[{block}] is identically zero after beta change"

    with HDFArchive('out/inp.h5', 'r') as ar:
        beta_out = ar['DMFT_input']['general_params']['beta']
        assert abs(beta_out - 400) < 1e-10, \
            f"Expected beta=400 in run2 output, got {beta_out}"

    print("Test passed: load_sigma from T=0 edipack (beta=600) to finite-T (beta=400) via DLR works correctly.")
