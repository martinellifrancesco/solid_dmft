"""
Test that load_sigma with a different beta correctly transforms the self-energy
via DLR interpolation. Runs two calculations: beta=600 followed by beta=400
loading sigma from the first run.
"""
import shutil
import numpy as np
from h5 import HDFArchive
import triqs.utility.mpi as mpi

import solid_dmft.main as solid

if mpi.is_master_node():
    shutil.rmtree('out', ignore_errors=True)

mpi.barrier()

# Run 1: beta=600, normal DMFT calculation — writes to out/inp.h5
solid.main([None, 'dmft_config_run1.toml'])

mpi.barrier()

# Run 2: beta=400, load_sigma=true — reads sigma from the same out/inp.h5, transforms via DLR
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

    # Verify run2 stored the new beta
    with HDFArchive('out/inp.h5', 'r') as ar:
        beta_out = ar['DMFT_input']['general_params']['beta']
        assert abs(beta_out - 400) < 1e-10, \
            f"Expected beta=400 in run2 output, got {beta_out}"

    print("Test passed: load_sigma with beta change (600 -> 400) via DLR works correctly.")
