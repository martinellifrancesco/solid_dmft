# from triqs.gf import MeshReFreq, Gf, make_hermitian
from triqs.gf import *
import numpy as np
from itertools import product
from triqs.gf.descriptors import Fourier
from triqs.operators import c, c_dag, dagger, Operator
from triqs.gf.tools import inverse

# import of the abstract class
from solid_dmft.dmft_tools.solvers.abstractdmftsolver import AbstractDMFTSolver

# import triqs solver
from edipack2triqs.solver import EDIpackSolver, LanczosParams
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
        self.triqs_solver_params_h, self.triqs_solver_params_fit, self.triqs_solver_params_lanczos, self.triqs_solver_params_solve, self.triqs_solver_params_general = {}, {}, {}, {}, {}
        
        h_params = ['bath_sites','bath_type','bath_hybridization_max', 'bath_energy_max','bath_mixing']
        fit_params = ["cg_scheme", "cg_method", "cg_grad", "cg_ftol", "cg_stop", "cg_niter", "cg_weight", "cg_norm", "cg_pow", "cg_minimize_ver", "cg_minimize_hh", "lfit"]
        lanczos_params = ["lanc_method", "lanc_nstates_sector", "lanc_nstates_total", "lanc_nstates_step", "lanc_ncv_factor", "lanc_ncv_add", "lanc_niter", "lanc_ngfiter", "lanc_tolerance", "lanc_dim_threshold"]
        solve_params = ['beta', 'n_iw', 'n_w', 'zerotemp']
        general_solver_params = ['ed_verbose', 'print_input_vars', 'cutoff', 'gs_threshold', 'ed_sparse_h']

        for key in self.solver_params.keys():
            if key in h_params:
                self.triqs_solver_params_h[key] = self.solver_params[key]
            elif key in fit_params:
                if key.startswith('cg_'):
                    if key == 'cg_ftol':
                        self.triqs_solver_params_fit[key[len('cg_f'):]] = self.solver_params[key]
                    else:
                        self.triqs_solver_params_fit[key[len('cg_'):]] = self.solver_params[key]
                else:
                    self.triqs_solver_params_fit['n_iw'] = self.solver_params[key]
            elif key in lanczos_params:
                self.triqs_solver_params_lanczos[key[len('lanc_'):]] = self.solver_params[key]
            elif key in general_solver_params:
                self.triqs_solver_params_general[key] = self.solver_params[key]
            else:
                self.triqs_solver_params_solve[key] = self.solver_params[key]

        for key in solve_params:
            if key in self.general_params.keys() or key in self.solver_params.keys():
                self.triqs_solver_params_solve[key] = self.general_params[key] 
        
        # sets up necessary GF objects on ImFreq
        self._init_ImFreq_objects()
        if 'zerotemp' in self.triqs_solver_params_solve.keys() and 'n_w' in self.triqs_solver_params_solve.keys():
            self._init_ReFreq_ED()
        
        sumk_eal = self.sum_k.eff_atomic_levels()[icrsh] 
        solver_eal = self.sum_k.block_structure.convert_matrix(sumk_eal, space_from='sumk', ish_from=self.sum_k.inequiv_to_corr[icrsh])
        # Debug
        self.sumk_eal = sumk_eal
        self.solver_eal = solver_eal
        
        spins = ['up', 'down']
        if sum_k.SO == 1:
            orbs = [i for i in range(sum_k.corr_shells[0]['dim']//2)]
            Nbath, Nspin, Norb = int(self.solver_params['bath_sites']), len(spins), len(orbs)
            for s, spin_block in solver_eal.items():
                eps = spin_block.reshape(Norb, Nspin, Norb, Nspin)
                eps = eps.transpose(0,2,1,3)
        else:
            orbs = [i for i in range(sum_k.corr_shells[0]['dim'])]
            Nbath, Nspin, Norb = int(self.solver_params['bath_sites']), len(spins), len(orbs)
            eps = np.zeros((Norb, Norb, Nspin, Nspin), dtype='complex')
            for s, spin_block in solver_eal.items():
                for o1 in range(spin_block.shape[0]):
                    for o2 in range(spin_block.shape[1]):
                        if 'up' in s:
                            eps[o1,o2,0,0] = spin_block[o1,o2]
                        else:
                            eps[o1,o2,1,1] = spin_block[o1,o2]
                            
        # if mpi.is_master_node():                  
        #     print('Effective atomic levels for the impurity problem (in solver basis):')
        #     for s in range(Nspin):
        #         for sp in range(Nspin):
        #             print(f'Spin {spins[s]}{spins[sp]}:')
        #             print(f'{eps[:,:,s,sp]}')

        # Impurity Hamiltonian 
        degeneracy = np.diag(eps[:,:,0,0] - eps[0,0,0,0]).real
        H = sum(eps[o, op, s, sp] 
            * c_dag(spins[s] + f'_{icrsh}', o) * c(spins[sp] + f'_{icrsh}', op)
            for s, sp, o, op in product(range(Nspin), range(Nspin), range(Norb), range(Norb))) 
        
        # Add interaction term to the Hamiltonian
        if sum_k.SO == 1:
            H += self._h_int_transform(h_int, icrsh)
        else:
            H += h_int
        
        # Bath Hamiltonian and hybridization
        if self.solver_params['bath_type'] == 'normal':
            eps_b = np.linspace(-self.triqs_solver_params_h['bath_energy_max'], self.triqs_solver_params_h['bath_energy_max'], Nbath)
            H += sum(eps_b[nu]
            * c_dag("B_" + spins[s] + f'_{icrsh}', nu + o * Nbath) * c("B_" + spins[s] + f'_{icrsh}', nu + o * Nbath)
            for s, o, nu in product(range(Nspin),range(Norb), range(Nbath)))
            # Impurity-bath hybridization
            hyb_b = np.linspace(0, self.triqs_solver_params_h['bath_hybridization_max'], Nbath)
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
            eps_b = np.linspace(-self.triqs_solver_params_h['bath_energy_max'], self.triqs_solver_params_h['bath_energy_max'], Nbath)
            H += sum(eps_b[nu]
            * c_dag("B_" + spins[s] + f'_{icrsh}', nu) * c("B_" + spins[s] + f'_{icrsh}', nu)
            for s, nu in product(range(Nspin), range(Nbath)))
            # Impurity-bath hybridization
            hyb_b = np.linspace(0, self.triqs_solver_params_h['bath_hybridization_max'], Nbath)
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
            H += sum(eps[o, op, s, sp] 
                * c_dag("B_" + spins[s] + f'_{icrsh}', o + nu * Norb) * c("B_" + spins[sp] + f'_{icrsh}', op + nu * Norb)
                for s, sp, o, op, nu in product(range(Nspin), range(Nspin), range(Norb), range(Norb), range(Nbath)))
            # Impurity-bath hybridization
            V = np.zeros((Nbath, Norb), dtype='float')
            if self.solver_params['bath_type'] == 'replica':
                hyb_b = self.triqs_solver_params_h['bath_hybridization_max'] * np.ones(Norb)
                for nu in range(Nbath):
                    V[nu,:] = hyb_b + nu/10
            else: 
                hyb_b = np.linspace(0, self.triqs_solver_params_h['bath_hybridization_max'], Norb)
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
        fops_imp_up, fops_imp_dn = [('up' + f'_{icrsh}', o) for o in orbs], [('down' + f'_{icrsh}', o) for o in orbs]
        fit_params = BathFittingParams(**self.triqs_solver_params_fit)
        self.triqs_solver = EDIpackSolver(H, fops_imp_up, fops_imp_dn, fops_bath_up, fops_bath_dn, bath_fitting_params=fit_params, keep_dir=True, 
                                         lanczos_params=LanczosParams(**self.triqs_solver_params_lanczos), **self.triqs_solver_params_general)
        
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
            
        self.postprocess(bath_old)
        
        if mpi.is_master_node():
            with open(self.general_params['jobname']+f'/solver_it{self.it}/solver_it{self.it}.pkl', 'wb') as file:
                pickle.dump(self, file)

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
        # Add G_omega if T=0 flag
        if 'zerotemp' in self.triqs_solver_params_solve.keys() and 'n_w' in self.triqs_solver_params_solve.keys():
            self.G_Refreq << self.triqs_solver.g_w
            self.Sigma_Refreq << self.triqs_solver.Sigma_w
        return


    def _init_ReFreq_ED(self):
        r"""
        Initialize all ReFreq objects
        """

        # create all ReFreq instances
        self.n_w = self.general_params['n_w']
        self.G_Refreq = self.sum_k.block_structure.create_gf(
            ish=self.icrsh, gf_function=Gf, space='solver', mesh=MeshReFreq(n_w=self.n_w, window=self.general_params['w_range'])
        )
        self.Sigma_Refreq = self.sum_k.block_structure.create_gf(
            ish=self.icrsh, gf_function=Gf, space='solver', mesh=MeshReFreq(n_w=self.n_w, window=self.general_params['w_range'])
        )
        
    def _h_int_transform(self, h_int_so, icrsh):
        """
        Transform operators:
            c_dag('ud_0', n)
            c('ud_0', n)
        into
            c_dag('up_0'/'down_0', n//2)
            c('up_0'/'down_0', n//2)
        """

        h_int = Operator()

        for term, coeff in h_int_so:
            new_term = Operator(1.0)

            for single_op in term:
                idx  = single_op[-1][-1]
                if idx % 2 == 0:
                    new_name = 'up'+ f'_{icrsh}'
                else:
                    new_name = 'down'+ f'_{icrsh}'
                new_idx = idx // 2
                if single_op[0]:
                    new_term *= c_dag(new_name, new_idx)
                else:
                    new_term *= c(new_name, new_idx)

            h_int += coeff * new_term

        return h_int