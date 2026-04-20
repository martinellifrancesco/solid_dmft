from triqs.gf import *
import numpy as np
from itertools import count, product
from triqs.gf.descriptors import Fourier
from triqs.operators import c, c_dag, dagger, Operator
from triqs.gf.tools import inverse
from triqs_hubbardI.version import triqs_hubbardI_hash, version
import triqs.utility.mpi as mpi

# import of the abstract class
from solid_dmft.dmft_tools.solvers.abstractdmftsolver import AbstractDMFTSolver

# import triqs solver
from edipack2triqs.solver import EDIpackSolver, LanczosParams
from edipack2triqs.fit import BathFittingParams

import os
import shutil
import pickle

class EDIpackInterface(AbstractDMFTSolver):
    def __init__(self, general_params, solver_params, sum_k, icrsh, h_int, iteration_offset, deg_orbs_ftps, gw_params=None, advanced_params=None):
        # Call the base class constructor
        super().__init__(general_params, solver_params, sum_k, icrsh, h_int, iteration_offset, deg_orbs_ftps, gw_params, advanced_params)

        # Solver parameters for the EDIpack solver
        self.triqs_solver_params_h, self.triqs_solver_params_fit, self.triqs_solver_params_lanczos, self.triqs_solver_params_solve, self.triqs_solver_params_general = {}, {}, {}, {}, {}
        
        h_params = ['bath_sites','bath_type','bath_hybridization_max', 'bath_energy_max','bath_mixing', 'bath_fit_it0', 'symmetrize_bath']
        fit_params = ["cg_scheme", "cg_method", "cg_grad", "cg_ftol", "cg_stop", "cg_niter", "cg_weight", "cg_norm", "cg_pow", "cg_minimize_ver", "cg_minimize_hh", "lfit"]
        lanczos_params = ["lanc_method", "lanc_nstates_sector", "lanc_nstates_total", "lanc_nstates_step", "lanc_ncv_factor", "lanc_ncv_add", "lanc_niter", "lanc_ngfiter", "lanc_tolerance", "lanc_dim_threshold"]
        solve_params = ['beta', 'n_iw', 'n_w']
        general_solver_params = ['ed_verbose', 'print_input_vars', 'cutoff', 'gs_threshold', 'ed_sparse_h', 'zerotemp']

        # Parse the solver parameters and assign them to the appropriate dictionaries
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

        # Some parameters needed in the solve step can be set in general_params or solver_params
        for key in solve_params:
            
            if key in self.general_params.keys():
                self.triqs_solver_params_solve[key] = self.general_params[key]
                 
            elif key in self.solver_params.keys():
                self.triqs_solver_params_solve[key] = self.solver_params[key]
        
        # sets up necessary GF objects on ImFreq
        self._init_ImFreq_objects()
        if 'zerotemp' in self.triqs_solver_params_solve.keys() and 'n_w' in self.triqs_solver_params_solve.keys():
            self._init_ReFreq_ED()
        
        # Prepare the impurity operator lists for the solver 
        if self.sum_k.SO == 1:
            if mpi.is_master_node():
                print('Spin-orbit coupling is enabled. Bath topology is set to "general"')
            self.triqs_solver_params_h['bath_type'] = 'general'
            self.spins = ['ud']
            self.orbs = [i for i in range(self.sum_k.corr_shells[0]['dim'])]
            self.Nbath, self.Nspin, self.Norb = int(self.solver_params['bath_sites']), len(self.spins), len(self.orbs)
            self.fops_imp_up, self.fops_imp_dn = [('ud' + f'_{icrsh}', o) for o in self.orbs if o % 2 == 0], [('ud' + f'_{icrsh}', o) for o in self.orbs if o % 2 == 1]
        else:
            self.spins = ['up', 'down']
            self.orbs = [i for i in range(self.sum_k.corr_shells[0]['dim'])]
            self.Nbath, self.Nspin, self.Norb = int(self.solver_params['bath_sites']), len(self.spins), len(self.orbs)
            self.fops_imp_up, self.fops_imp_dn = [('up' + f'_{icrsh}', o) for o in self.orbs], [('down' + f'_{icrsh}', o) for o in self.orbs]
        
        # Store the effective atomic levels for the current shell
        self.sumk_eal = self.sum_k.eff_atomic_levels()[icrsh] 
        self.solver_eal = self.sum_k.block_structure.convert_matrix(self.sumk_eal, space_from='sumk', ish_from=self.sum_k.inequiv_to_corr[icrsh])

        # Impurity Hamiltonian 
        H = self.prepare_H_loc()
        
        # Add interaction term to the Hamiltonian
        H += h_int
        self.H_loc_int = h_int
        
        # Bath Hamiltonian and hybridization
        if self.solver_params['bath_type'] == 'normal':
            # Initialize bath energies with degeneracy-based modulation
            eps_b = np.linspace(-self.triqs_solver_params_h['bath_energy_max'], self.triqs_solver_params_h['bath_energy_max'], self.Nbath)
            H += sum(eps_b[nu] * c_dag("B_" + self.spins[s] + f'_{icrsh}', nu + o * self.Nbath) * c("B_" + self.spins[s] + f'_{icrsh}', nu + o * self.Nbath)
                    for s, o, nu in product(range(self.Nspin),range(self.Norb), range(self.Nbath)))
            # Add hybridization term with degeneracy-based modulation
            V = np.linspace(self.triqs_solver_params_h['bath_hybridization_max']/2, self.triqs_solver_params_h['bath_hybridization_max'], self.Nbath)[:, None] * np.ones(self.Norb * self.Nspin) + self.degeneracy[np.newaxis, :]
            H += sum(V[nu, 2*o + s] * (c_dag(self.spins[s] + f'_{icrsh}', o) * c("B_" + self.spins[s] + f'_{icrsh}', nu + o * self.Nbath) + c_dag("B_" + self.spins[s] + f'_{icrsh}', nu + o * self.Nbath) * c(self.spins[s] + f'_{icrsh}', o))
                    for s, o, nu in product(range(self.Nspin), range(self.Norb), range(self.Nbath)))
            # Prepare the bath operator lists for the solver
            self.fops_bath_up, self.fops_bath_dn = [('B_up' + f'_{icrsh}', i) for i in range(self.Norb * self.Nbath)], [('B_down' + f'_{icrsh}', i) for i in range(self.Norb * self.Nbath)]
            
        elif self.solver_params['bath_type'] == 'hybrid':
            # Initialize bath energies
            eps_b = np.linspace(-self.triqs_solver_params_h['bath_energy_max'], self.triqs_solver_params_h['bath_energy_max'], self.Nbath)
            H += sum(eps_b[nu] * c_dag("B_" + self.spins[s] + f'_{icrsh}', nu) * c("B_" + self.spins[s] + f'_{icrsh}', nu)
                    for s, nu in product(range(self.Nspin), range(self.Nbath)))
            # Add hybridization term with degeneracy-based modulation
            V = np.linspace(self.triqs_solver_params_h['bath_hybridization_max']/2, self.triqs_solver_params_h['bath_hybridization_max'], self.Nbath)[:, None] * np.ones(self.Norb * self.Nspin) + self.degeneracy[np.newaxis, :]
            H += sum(V[nu, 2*o + s] * (c_dag(self.spins[s] + f'_{icrsh}', o) * c("B_" + self.spins[s] + f'_{icrsh}', nu) + c_dag("B_" + self.spins[s] + f'_{icrsh}', nu) * c(self.spins[s] + f'_{icrsh}', o))
                    for s, o, nu in product(range(self.Nspin), range(self.Norb), range(self.Nbath)))
            # Prepare the bath operator lists for the solver
            self.fops_bath_up, self.fops_bath_dn = [('B_up' + f'_{icrsh}', i) for i in range(self.Nbath)], [('B_down' + f'_{icrsh}', i) for i in range(self.Nbath)]
            
        else:
            # Initialize bath energies with impurity replicas
            H += sum(self.eps[i, j] * c_dag("B_" + self.spins[i % self.Nspin] + f'_{icrsh}', i // self.Nspin + nu * self.Norb) * c("B_" + self.spins[j % self.Nspin] + f'_{icrsh}', j // self.Nspin + nu * self.Norb)
                for i, j, nu in product(range(self.Nspin*self.Norb), range(self.Nspin*self.Norb), range(self.Nbath)))
            # Add hybridization term with degeneracy-based modulation   
            if self.solver_params['bath_type'] == 'replica':
                V = (self.triqs_solver_params_h['bath_hybridization_max']*np.ones(self.Nbath))[:, None] * np.ones(self.Norb * self.Nspin) + self.degeneracy[np.newaxis, :]
            else: 
                V = np.linspace(self.triqs_solver_params_h['bath_hybridization_max']/2, self.triqs_solver_params_h['bath_hybridization_max'], self.Nbath)[:, None] * np.ones(self.Norb * self.Nspin) - self.degeneracy[np.newaxis, :]
            H += sum(V[nu, i] * (c_dag(self.spins[i % self.Nspin] + f'_{icrsh}', i // self.Nspin) * c("B_" + self.spins[i % self.Nspin] + f'_{icrsh}', i // self.Nspin + nu * self.Norb) + c_dag("B_" + self.spins[i % self.Nspin] + f'_{icrsh}', i // self.Nspin + nu * self.Norb) * c(self.spins[i % self.Nspin] + f'_{icrsh}', i // self.Nspin))
                for i, nu in product(range(self.Nspin*self.Norb), range(self.Nbath)))
            # Prepare the bath operator lists for the solver
            if sum_k.SO == 1:
                self.fops_bath_up, self.fops_bath_dn = [('B_ud' + f'_{icrsh}', o) for o in range(self.Norb * self.Nbath) if o % 2 == 0], [('B_ud' + f'_{icrsh}', o) for o in range(self.Norb * self.Nbath) if o % 2 == 1]
            else:
                self.fops_bath_up, self.fops_bath_dn = [('B_up' + f'_{icrsh}', i) for i in range(self.Norb * self.Nbath)], [('B_down' + f'_{icrsh}', i) for i in range(self.Norb * self.Nbath)]
        
        self.H_bath = H - self.H_loc - self.H_loc_int
        
        # Check hermiticity of the total Hamiltonian
        if mpi.is_master_node():
            check = H - dagger(H)
            coeffs = [coeff for _, coeff in check]
            if np.any(np.abs(coeffs) > 1e-10):
                raise ValueError('Hamiltonian is not hermitian, please check the input Hamiltonian.')
            else:
                print('Hamiltonian is hermitian.')

        # Make sure the Hamiltonian is explicitly hermitian to avoid numerical issues in the solver
        H = (H + dagger(H)) / 2
        fit_params = BathFittingParams(**self.triqs_solver_params_fit)
        self.triqs_solver = EDIpackSolver(H, self.fops_imp_up, self.fops_imp_dn, self.fops_bath_up, self.fops_bath_dn, bath_fitting_params=fit_params, keep_dir=False, 
                                         lanczos_params=LanczosParams(**self.triqs_solver_params_lanczos), **self.triqs_solver_params_general)
        
        # Store git hash and version. TO-DO
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
        
        # Update the local Hamiltonian if not the first iteration to account for changes in the chemical potential
        if self.it > 0:
            self.triqs_solver.hloc = self.prepare_H_loc()

        if self.it == 0 and 'bath_fit_it0' in self.triqs_solver_params_h.keys():
            skip_fit = not self.triqs_solver_params_h['bath_fit_it0']
        else:            
            skip_fit = False
        # Update the bath parameters by fitting the Weiss field, with optional mixing with the previous bath
        bath_old = self.triqs_solver.bath
        if 'bath_mixing' in self.solver_params.keys() and self.it > 0:
            alpha = self.solver_params['bath_mixing']
        else:
            alpha = 1.0
        if not skip_fit:
            # Perform the bath fitting, with optional symmetrization of the Weiss field
            # if 'symmetrize_bath' in self.triqs_solver_params_h.keys():
            #     if self.triqs_solver_params_h['symmetrize_bath']:
            #         G0_fit = self.G0_freq.copy()
            #         for block, G0_b in G0_fit:
            #             n_orb = G0_b.data.shape[1]
            #             diag_avg = np.mean(np.diagonal(G0_b.data, axis1=1, axis2=2), axis=1)
            #             for i in range(n_orb):
            #                 G0_b.data[:, i, i] = diag_avg
            #                 for j in range(i+1, n_orb):
            #                     avg = 0.5 * (G0_b.data[:, i, j] + np.conj(G0_b.data[:, j, i]))
            #                     G0_fit[block].data[:, i, j] = avg
            #                     G0_fit[block].data[:, j, i] = np.conj(avg)
            #     else:
            #         G0_fit = self.G0_freq.copy()
            G0_fit = self.G0_freq.copy()
            self.triqs_solver.bath = alpha * self.triqs_solver.chi2_fit_bath(G0_fit)[0] + (1 - alpha) * bath_old
            
            if 'symmetrize_bath' in self.triqs_solver_params_h.keys():
                if self.triqs_solver_params_h['symmetrize_bath']:
                    for no in range(self.Norb):
                        self.triqs_solver.bath.eps[0,no,:] = np.mean(self.triqs_solver.bath.eps[0,:,:], axis=0)
                        self.triqs_solver.bath.V[0,no,:] = np.mean(self.triqs_solver.bath.V[0,:,:], axis=0)
            
        
        # Solve the impurity problem with the current bath parameters
        self.triqs_solver.solve(**self.triqs_solver_params_solve)
        self.it += 1 
        
        # Copy the solver output files from the temporary directory to a permanent location. 
        if mpi.is_master_node(): 
            ed_tmp_dir = [entry for entry in os.listdir('./') if entry.endswith('.tmp')][0]
            os.makedirs(self.general_params['jobname']+f'/solver_it{self.it}', exist_ok=True)
            for item in os.listdir(ed_tmp_dir):
                src_path = os.path.join(ed_tmp_dir, item)
                dst_path = os.path.join(self.general_params['jobname']+f'/solver_it{self.it}', item)
                shutil.copy2(src_path, dst_path)  
            with open(self.general_params['jobname']+f'/solver_it{self.it}/solver_it{self.it}.pkl', 'wb') as file:
                pickle.dump(self, file)
            
        # Organize the solver output into the appropriate GF objects and store them in the class attributes for later use in the DMFT loop
        self.postprocess()
        
        return


    def postprocess(self):
        r"""
        Organize G_freq, Sigma_freq and bath from hartree solver
        """
        
        # Store solver output in the appropriate GF objects
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
        self.G_Refreq = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver', mesh=MeshReFreq(n_w=self.n_w, window=self.general_params['w_range']))
        self.Sigma_Refreq = self.sum_k.block_structure.create_gf(ish=self.icrsh, gf_function=Gf, space='solver', mesh=MeshReFreq(n_w=self.n_w, window=self.general_params['w_range']))

    def prepare_H_loc(self):
        
        # Store the effective atomic levels for the current shell 
        self.sumk_eal = self.sum_k.eff_atomic_levels()[self.icrsh] 
        self.solver_eal = self.sum_k.block_structure.convert_matrix(self.sumk_eal, space_from='sumk', ish_from=self.sum_k.inequiv_to_corr[self.icrsh])
        
        if self.sum_k.SO == 1:
            self.eps = self.solver_eal['ud'+f'_{self.icrsh}']
        else:
            self.eps = np.zeros((self.Norb*self.Nspin, self.Norb*self.Nspin), dtype='complex')
            for s, spin_block in self.solver_eal.items():
                spin_offset = 0 if 'up' in s else 1
                self.eps[spin_offset::2, spin_offset::2] = spin_block

        # Build impurity Hamiltonian 
        self.degeneracy = np.diag(self.eps - np.min(self.eps)).real
        self.H_loc = sum(self.eps[i, j] * c_dag(self.spins[i % self.Nspin] + f'_{self.icrsh}', i // self.Nspin) * c(self.spins[j % self.Nspin] + f'_{self.icrsh}', j // self.Nspin) for i, j in product(range(self.Nspin*self.Norb), range(self.Nspin*self.Norb)))
        
        return self.H_loc