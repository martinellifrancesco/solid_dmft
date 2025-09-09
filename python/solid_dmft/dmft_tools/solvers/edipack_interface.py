# from triqs.gf import MeshReFreq, Gf, make_hermitian
from triqs.gf import *
import numpy as np
from itertools import product
from triqs.gf.descriptors import Fourier
from triqs.operators import c, c_dag, dagger
from triqs.gf.tools import inverse

# import of the abstract class
from solid_dmft.dmft_tools.solvers.abstractdmftsolver import AbstractDMFTSolver

# import triqs solver
from edipack2triqs.solver import EDIpackSolver
from edipack2triqs.fit import BathFittingParams
from triqs_hubbardI.version import triqs_hubbardI_hash, version
import triqs.utility.mpi as mpi

import os
import shutil
import pickle

class EDIpackInterface(AbstractDMFTSolver):
    def __init__(
        self, general_params, solver_params, sum_k, icrsh, h_int, iteration_offset, deg_orbs_ftps, gw_params=None, advanced_params=None
    ):
        # Call the base class constructor
        super().__init__(general_params, solver_params, sum_k, icrsh, h_int, iteration_offset, deg_orbs_ftps, gw_params, advanced_params)

        # Solver parameters for the EDIpack solver
        self.triqs_solver_params_solve, self.fitting_kwargs, self.triqs_solver_params = {}, {}, {}
        keys_h_params = ['bath_sites','bath_type','bath_mixing']
        allowed_params = ['scheme', 'method', 'grad', 'tol', 'stop', 'niter', 'n_iw','weight', 'norm', 'pow', 'minimize_ver', 'minimize_hh']
        solve_params = ['beta', 'n_iw']

        for key in self.solver_params.keys():
            if key in keys_h_params:
                continue
            elif key.startswith("fit_") and key[len("fit_"):] in allowed_params:
                self.fitting_kwargs[key[len("fit_"):]] = self.solver_params[key]
            else:
                self.triqs_solver_params[key] = self.solver_params[key]
        for key in solve_params:
            self.triqs_solver_params_solve[key] = self.general_params[key] 
        
        # sets up necessary GF objects on ImFreq
        self._init_ImFreq_objects()

        # Define the bath Hamiltonian
        spins, orbs = ['up', 'down'], [i for i in range(sum_k.corr_shells[0]['dim'])]
        Nbath, Nspin, Norb = int(self.solver_params['bath_sites']), len(spins), sum_k.corr_shells[0]['dim']

        # Impurity Hamiltonian
        eps = np.zeros((Norb, Norb, Nspin, Nspin), dtype='complex')
        sumk_eal = self.sum_k.eff_atomic_levels()[icrsh]
        solver_eal = self.sum_k.block_structure.convert_matrix(sumk_eal, space_from='sumk', ish_from=self.sum_k.inequiv_to_corr[icrsh])
        for s, spin_block in solver_eal.items():
            for o1 in range(spin_block.shape[0]):
                for o2 in range(spin_block.shape[1]):
                    if 'ud' in s:
                        eps[o1,o2,0,1] = spin_block[o1,o2]
                        eps[o1,o2,1,0] = spin_block[o1,o2]
                    elif 'up' in s:
                        eps[o1,o2,0,0] = spin_block[o1,o2]
                    else:
                        eps[o1,o2,1,1] = spin_block[o1,o2]
        degeneracy = np.diag(eps[:,:,0,0] - eps[0,0,0,0]).real
        H = sum(eps[o, op, s, sp] 
            * c_dag(spins[s] + f'_{icrsh}', o) * c(spins[sp] + f'_{icrsh}', op)
            for s, sp, o, op in product(range(Nspin), range(Nspin), range(Norb), range(Norb))) 
        H += h_int

        fops_imp_up, fops_imp_dn = [('up' + f'_{icrsh}', o) for o in orbs], [('down' + f'_{icrsh}', o) for o in orbs]

        if self.solver_params['bath_type'] == 'normal':
            # Bath Hamiltonian
            eps_b = np.linspace(0.1, 1, Nbath)
            H += sum(eps_b[nu]
            * c_dag("B_" + spins[s] + f'_{icrsh}', nu + o * Nbath) * c("B_" + spins[s] + f'_{icrsh}', nu + o * Nbath)
            for s, o, nu in product(range(Nspin),range(Norb), range(Nbath)))
            # Impurity-bath hybridization
            hyb_b = np.linspace(0.1, 0.5, Nbath)
            V = np.zeros((Nbath, Norb), dtype='float')
            if np.allclose(degeneracy, 0.0):
                for o in range(Norb):
                    V[:,o] = hyb_b
            else:
                for o in range(Norb):
                    V[:,o] = hyb_b - degeneracy[o]
            H += sum(V[nu, o]
                * (c_dag(spins[s] + f'_{icrsh}', o) * c("B_" + spins[s] + f'_{icrsh}', nu + o * Nbath) + c_dag("B_" + spins[s] + f'_{icrsh}', nu + o * Nbath) * c(spins[s] + f'_{icrsh}', o))
                for s, o, nu in product(range(Nspin), range(Norb), range(Nbath)))
            fops_bath_up, fops_bath_dn = [('B_up' + f'_{icrsh}', i) for i in range(Norb * Nbath)], [('B_down' + f'_{icrsh}', i) for i in range(Norb * Nbath)]
            
        elif self.solver_params['bath_type'] == 'hybrid':
            # Bath Hamiltonian
            eps_b = np.linspace(0.1, 1, Nbath)
            H += sum(eps_b[nu]
            * c_dag("B_" + spins[s] + f'_{icrsh}', nu) * c("B_" + spins[s] + f'_{icrsh}', nu)
            for s, nu in product(range(Nspin), range(Nbath)))
            # Impurity-bath hybridization
            hyb_b = np.linspace(0.1, 0.5, Nbath)
            V = np.zeros((Nbath, Norb), dtype='float')
            if np.allclose(degeneracy, 0.0):
                for o in range(Norb):
                    V[:,o] = hyb_b
            else:
                for o in range(Norb):
                    V[:,o] = hyb_b - degeneracy[o]
            H += sum(V[nu, o]
                * (c_dag(spins[s] + f'_{icrsh}', o) * c("B_" + spins[s] + f'_{icrsh}', nu) + c_dag("B_" + spins[s] + f'_{icrsh}', nu) * c(spins[s] + f'_{icrsh}', o))
                for s, o, nu in product(range(Nspin), range(Norb), range(Nbath)))
            fops_bath_up, fops_bath_dn = [('B_up' + f'_{icrsh}', i) for i in range(Nbath)], [('B_down' + f'_{icrsh}', i) for i in range(Nbath)]
            
        else:
            # Bath Hamiltonian
            H += sum(eps[o, op, s, sp] 
                * c_dag("B_" + spins[s] + f'_{icrsh}', o + nu * Norb) * c("B_" + spins[sp] + f'_{icrsh}', op + nu * Norb)
                for s, sp, o, op, nu in product(range(Nspin), range(Nspin), range(Norb), range(Norb), range(Nbath)))
            # Impurity-bath hybridization
            V = np.zeros((Nbath, Norb), dtype='float')
            if self.solver_params['bath_type'] == 'replica':
                hyb_b = 0.1*np.ones(Norb)
                for nu in range(Nbath):
                    V[nu,:] = hyb_b + nu/10
            else: 
                hyb_b = np.linspace(0.1, 0.5, Norb)
                for nu in range(Nbath):
                    V[nu,:] = hyb_b + nu/10
            H += sum(V[nu, o]
                * (c_dag(spins[s] + f'_{icrsh}', o) * c("B_" + spins[s] + f'_{icrsh}', o + nu * Norb) + c_dag("B_" + spins[s] + f'_{icrsh}', o + nu * Norb) * c(spins[s] + f'_{icrsh}', o))
                for s, o, nu in product(range(Nspin), range(Norb), range(Nbath)))
            fops_bath_up, fops_bath_dn = [('B_up' + f'_{icrsh}', i) for i in range(Norb * Nbath)], [('B_down' + f'_{icrsh}', i) for i in range(Norb * Nbath)]
        
        if mpi.is_master_node():
            check = H - dagger(H)
            coeffs = [coeff for _, coeff in check]
            if np.any(np.abs(coeffs) > 1e-10):
                raise ValueError('Hamiltonian is not hermitian, please check the input Hamiltonian.')
            else:
                print('Hamiltonian is hermitian.')

        H = (H + dagger(H)) / 2

        fit_params = BathFittingParams(**self.fitting_kwargs)
        self.triqs_solver = EDIpackSolver(H, fops_imp_up, fops_imp_dn, fops_bath_up, fops_bath_dn, bath_fitting_params=fit_params, keep_dir=True, 
                                         **self.triqs_solver_params)
        
        self.git_hash = triqs_hubbardI_hash  # edipack_hash
        self.version = version # version

        self.it = 0

        if mpi.is_master_node():    
            print(self.triqs_solver.h_params.Hloc)
            print('Exact diagonalization mode chosen for EDIpack is: ', self.triqs_solver.h_params.ed_mode)
            print('Bath topology chosen for EDIpack is: ', self.triqs_solver.bath.name)

    def solve(self, **kwargs):
        # Solve the impurity problem for icrsh shell
        # *************************************
        # this is done on every node due to very slow bcast
        bath_old = self.triqs_solver.bath
        if 'bath_mixing' in self.solver_params.keys() and self.it > 0:
            alpha = self.solver_params['bath_mixing']
        else:
            alpha = 1.0
        self.triqs_solver.bath = alpha * self.triqs_solver.chi2_fit_bath(self.G0_freq)[0] + (1 - alpha) * bath_old
        self.triqs_solver.solve(**self.triqs_solver_params_solve)
        self.it += 1 
        if mpi.is_master_node(): 

            ed_tmp_dir = [entry for entry in os.listdir('./') if entry.endswith('.tmp')][0]
            os.makedirs(self.general_params['jobname']+f'/solver_it{self.it}', exist_ok=True)
            for item in os.listdir(ed_tmp_dir):
                src_path = os.path.join(ed_tmp_dir, item)
                dst_path = os.path.join(self.general_params['jobname']+f'/solver_it{self.it}', item)
                shutil.copy2(src_path, dst_path)
                
            with open(self.general_params['jobname']+f'/solver_it{self.it}/solver_it{self.it}.pkl', 'wb') as file:
                pickle.dump(self, file)
            
        self.postprocess(bath_old)

        return

    def postprocess(self, bath_old):
        r"""
        Organize G_freq, Sigma_freq and bath from hartree solver
        """

        self.G_freq_unsym << self.triqs_solver.g_iw
        self.G_freq << self.triqs_solver.g_iw
        self.sum_k.symm_deg_gf(self.G_freq, ish=self.icrsh)
        self.Sigma_freq << self.triqs_solver.Sigma_iw
        self.G_time << Fourier(self.G_freq)

        return