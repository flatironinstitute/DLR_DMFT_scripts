#DMFT calculation for a hubbard model on a 2D square lattice
#uses Triqs version 3.0.x
import numpy as np
from triqs.sumk import *
from triqs.gf import *
import triqs.utility.mpi as mpi
from triqs.utility.dichotomy import dichotomy
from triqs_cthyb import *
from h5 import HDFArchive
from triqs.operators import util
from triqs.lattice.utils import k_space_path, TB_from_wannier90
from triqs.operators import c_dag, c, Operator

from timeit import default_timer as timer
from wannier90_tools import *

np.set_printoptions(precision=6,suppress=True)

beta = 40.
U = 2.3                   # hubbard U parameter
J = 0.40                  # hubbard J parameter
nloops = 100              # number of DMFT loops needs 5-10 loops to converge
nk = 30                   # number of k points in each dimension
density_required = 4.     # target density for setting the chemical potential
n_orb = 3                 # number of orbitals
mu = 11.3715              # chemical potential
add_spin = False
w90_seedname = 'sr2ruo4'
w90_pathname = './'

eps = 10**-6              # DLR accuracy

outfile = f'U{U:.1F}_{eps:.1E}'

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 200
p["n_warmup_cycles"] = int(1e4)
p["n_cycles"] = int(4e7/mpi.size)
p["imag_threshold"] = 1e-8
p["off_diag_threshold"] = 1e-6

# tail fit
# turn this off to get the raw QMC results without fitting
tailfit = 'DLR'
assert tailfit in ['TRIQS','DLR','off']
if tailfit == 'TRIQS':
    p["perform_tail_fit"] = True
    p["fit_max_moment"] = 4
    p["fit_min_w"] = 5
    p["fit_max_w"] = 12
elif tailfit == 'DLR':
    p["perform_tail_fit"] = False
    from scipy.linalg import lstsq
elif tailfit == 'off':
    p["perform_tail_fit"] = False  
outfile += f'_tailfit_{tailfit}'

grid = 'DLR'
assert grid in ['full','DLR']
outfile += f'_grid_{grid}'

l_read = False
if l_read:
    readfile = outfile
    outfile += '_test'

l_previous_runs = False
if l_previous_runs:
    previous_runs = 7

S = Solver(beta=beta, gf_struct = [('up', 3), ('down', 3)], n_iw = 1025, n_tau=10001, delta_interface=True)

block_lst = list(['up','down'])

# set up interaction Hamiltonian
Umat, Upmat = util.U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)
h_int = util.h_int_kanamori(['up', 'down'], range(n_orb), off_diag=True, U=Umat, Uprime=Upmat, J_hund=J)

Gloc = S.G_iw.copy() #local Green's function

def kernel_it(τ,ω): 
    assert τ >= 0.
    if ω > 0.:
        return np.exp(-ω*τ)/(1+np.exp(-ω))
    else:
        return np.exp(ω*(1-τ))/(1+np.exp(ω))
    
def kernel_mf(n,ω):    
    return 1/(ω-(2*n+1)*np.pi*1j)

if tailfit == 'DLR' or grid == 'DLR':
    rf = np.loadtxt(f'dlrrf_{round(beta)}_{eps:.1E}.dat')
    rf.sort()
    
    mf = np.loadtxt(f'dlrmf_{round(beta)}_{eps:.1E}.dat',dtype=int)
    mf.sort()
    index = mf - (-S.n_iw)
    assert len(rf) == len(mf)
    κ_mf = np.zeros((len(mf),len(rf)),dtype=complex)
    for i,n in enumerate(mf):
        for j,ω in enumerate(rf):
            κ_mf[i,j] = kernel_mf(n,ω)
    
    mf_all = np.array([round((iw.value.imag*beta/np.pi-1)/2) for iw in Gloc.mesh])
    assert np.abs(mf_all[0]) == S.n_iw
    κ_mf_all = np.zeros((len(mf_all),len(rf)),dtype=complex)
    for i,n in enumerate(mf_all):
        for j,ω in enumerate(rf):
            κ_mf_all[i,j] = kernel_mf(n,ω)
            
    it_all = np.arange(0,S.n_tau) / ( S.n_tau - 1 )
    κ_it_all = np.zeros((len(it_all),len(rf)))
    for i,τ in enumerate(it_all):
        for j,ω in enumerate(rf):
            κ_it_all[i,j] = kernel_it(τ,ω)

# set up Wannier Hamiltonian
H_add_loc = np.zeros((n_orb, n_orb), dtype=complex)
H_add_loc += np.diag([-mu]*n_orb)

L = get_TBL(path=w90_pathname, name=w90_seedname, extend_to_spin=add_spin, add_local=H_add_loc)

SK = SumkDiscreteFromLattice(lattice=L, n_points=nk)

# extract epsilon0 from hoppings and add
e0 = L.hoppings[(0, 0, 0)]
mpi.report('epsilon0 (impurity energies):\n',e0.real)

def sumk(mu, Sigma, bz_weights, hopping, iw_index = None):
    '''
    calc Gloc with mpi parallelism
    '''
    Gloc = Sigma.copy()
    Gloc << 0.0+0.0j

    n_orb = Gloc['up'].target_shape[0]

    if isinstance(iw_index,(np.ndarray,list)):
        iw_values =  np.array(list(Gloc.mesh.values()))[iw_index]
        iw_mat = np.array([iw * np.eye(n_orb) for iw in iw_values])
        
        mu_mat = mu * np.eye(n_orb)

        #Loop on k points...
        for wk, eps_k in zip(*[mpi.slice_array(A) for A in [bz_weights, hopping]]):
            for block, gf in Gloc:
                # these are now all python numpy arrays, here you can replace by any array like object shape [nw, norb,norb]
                # numpy vectorizes the inversion automatically of shape [nw,orb,orb] over nw
                # speed up factor 30, comparable to C++!
                gf.data[iw_index,:,:] += wk*np.linalg.inv(iw_mat[:] + mu_mat - eps_k - Sigma[block].data[iw_index,:,:])
                
    else:
        iw_mat = np.array([iw.value * np.eye(n_orb) for iw in Gloc.mesh])

        mu_mat = mu * np.eye(n_orb)

        #Loop on k points...
        for wk, eps_k in zip(*[mpi.slice_array(A) for A in [bz_weights, hopping]]):
            for block, gf in Gloc:
                # these are now all python numpy arrays, here you can replace by any array like object shape [nw, norb,norb]
                # numpy vectorizes the inversion automatically of shape [nw,orb,orb] over nw
                # speed up factor 30, comparable to C++!
                gf.data[:,:,:] += wk*np.linalg.inv(iw_mat[:] + mu_mat - eps_k - Sigma[block].data[:,:,:])

    Gloc << mpi.all_reduce(mpi.world,Gloc,lambda x,y: x+y)
    mpi.barrier()
    
    if isinstance(iw_index,(np.ndarray,list)):
        DLR_gloc2gloc(Gloc)

    return Gloc

# old function to extract density for a given mu, to be used by dichotomy function to determine mu
def Dens(mu):
    # calling the k sum here
    # github.com/TRIQS/triqs/blob/3.0.x/python/triqs/sumk/sumk_discrete.py#L73
    dens =  SK(mu = mu, Sigma = S.Sigma_iw).total_density()
    if abs(dens.imag) > 1e-20:
            mpi.report("Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
    return dens.real

# function to determine density python only
def dens_dlr(mu):
    if grid == 'DLR':
        dens =  sumk(mu = mu, Sigma = S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping, iw_index=index).total_density()
    else:
        dens =  sumk(mu = mu, Sigma = S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping).total_density()
    if abs(dens.imag) > 1e-20:
            mpi.report("Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
    return dens.real

def DLR_gtau2giw(S):

    α = {}

    for i, block in enumerate(block_lst,1):
        for i_orb in range(n_orb):
            for j_orb in range(n_orb):
                α[block] = np.array(lstsq(κ_it_all,S.G_tau[block].data[:,i_orb,j_orb].real)[0])
                if i_orb == j_orb:
                    tail_shift = ( np.sum(α[block]) + 1.0 ) / 2
                    α[block][0] -= tail_shift
                    α[block][-1] -= tail_shift

                S.G_tau[block].data[:,i_orb,j_orb] = (κ_it_all@α[block]).astype(complex)
                S.G_iw[block].data[:,i_orb,j_orb] = (κ_mf_all@α[block]*beta).astype(complex)

    return

def DLR_gloc2gloc(Gloc):

    α = {}

    for i, block in enumerate(block_lst,1):

#         min ||A @ x - b||
#         subject to  C @ x = d
#         can be obtained using the Python code

#         # Define the matrices as usual, then
#         x = lapack.dgglse(A, C, b, d)[3]

        for i_orb in range(n_orb):
            for j_orb in range(n_orb):
                
                α[block] = np.linalg.solve(κ_mf,Gloc[block].data[index,i_orb,j_orb])
                if i_orb == j_orb:
                    tail_shift = ( np.sum(α[block]) + beta ) / 2
                    α[block][0] -= tail_shift
                    α[block][-1] -= tail_shift
                    
                Gloc[block].data[:,i_orb,j_orb] = (κ_mf_all@α[block]).astype(complex)
                    
#                 if type_DLR_gloc2gloc == 'constrained':
#                     if i_orb == j_orb:
#                         α[block] = lapack.zgglse(κ_mf, np.ones([1,len(rf)]), Gloc[block].data[index,i_orb,j_orb], np.array([[1]]))[3]
#                     else:
#                         α[block] = np.linalg.solve(κ_mf,Gloc[block].data[index,i_orb,j_orb])
#                 symmetrize(Gloc[block].data[:, i_orb, j_orb])

    return

def DLR_Diw2Dtau(S,Delta_iw):
    
    α = {}
    
    for i, block in enumerate(block_lst,1):
        for i_orb in range(n_orb):
            for j_orb in range(n_orb):
                α[block] = np.linalg.solve(κ_mf,Delta_iw[block].data[index,i_orb,j_orb])

                S.Delta_tau[block].data[:,i_orb,j_orb] = κ_it_all@α[block] / beta
            
    return

#check if there are previous runs in the outfile and if so restart from there
if not l_previous_runs:
    previous_runs = 0
previous_present = False
mu = 0.
if mpi.is_master_node():
    if l_read:
        ar = HDFArchive(readfile+'.h5','a')
        if 'iterations' in ar:
            previous_present = True
            if l_previous_runs:
                pass
            else:
                previous_runs = ar['iterations']
            S.Sigma_iw = ar[f'Sigma_iw-{previous_runs}']
            mu = ar[f'mu-{previous_runs}']
            del ar
    ar = HDFArchive(outfile+'.h5','w')
    del ar
    
previous_runs    = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)
S.Sigma_iw = mpi.bcast(S.Sigma_iw)
mu = mpi.bcast(mu)

for iteration_number in range(1,nloops+1):
    it = iteration_number + previous_runs
    if mpi.is_master_node():
        print('-----------------------------------------------')
        print("Iteration = %s"%it)
        print('-----------------------------------------------')

    if it > 1:
        S.Sigma_iw['up'] << .5*(S.Sigma_iw['up'] + S.Sigma_iw['down'])
        S.Sigma_iw['down'] << S.Sigma_iw['up']

    # determination of the next chemical potential via function Dens. Involves k summation
    start_time = timer()
    #mu, density = dichotomy(Dens, mu, density_required, 1e-4, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
    mu, density = dichotomy(dens_dlr, mu, density_required, 1e-4, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
    mpi.barrier()
    mpi.report('\nnumber of kpoints:'+str(SK.n_kpts()))
    mpi.report('time for calculating mu: {:.2f} s'.format(timer() - start_time))
    # calling the k sum here which you need to manually implement or change
    # but it is all python, even the invert is done on a numpy array
    # github.com/TRIQS/triqs/blob/3.0.x/python/triqs/sumk/sumk_discrete.py#L73

    # TODO step 3 write a new python function which replaces the SK() call which accepts a iw_vector and does the k sum only on these frequencies
    start_time = timer()
    # Gloc << SK(mu = mu, Sigma = S.Sigma_iw)
    Gloc << sumk(mu = mu, Sigma= S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping)
    mpi.barrier()
    mpi.report('time for k sum: {:.2f} s'.format(timer() - start_time))
    # TODO step 1
    # extract special Gloc(iw) points and do the DLR and back
    # iw_freq = [1, 10, 55, ..]
    # Gloc_spec = Gloc['up].data[iw_freq,:,:] #Gloc_spec is now a np array
    # DLR fit
    # evaluate DLR fit at all iw frequencies
    # see if the result is the same

    nlat = Gloc.total_density().real # lattice density
    if mpi.is_master_node():
        print('Gloc density matrix:')
        for block, gf in Gloc:
            print(block)
            print(gf.density().real)
            print('--------------')
        print('total occupation {:.4f}'.format(nlat))

    # calculate effective atomic levels (eal)
    solver_eal = e0 - np.diag([mu]*n_orb)
    Hloc_0 = Operator()
    for spin in ['up','down']:
        for o1 in range(n_orb):
            for o2 in range(n_orb):
                # check if off-diag element is larger than threshold
                if o1 != o2 and abs(solver_eal[o1,o2]) < p['off_diag_threshold']:
                    continue
                else:
                    Hloc_0 += (solver_eal[o1,o2].real)/2 * (c_dag(spin,o1) * c(spin,o2) + c_dag(spin,o2) * c(spin,o1))
    p['h_loc0'] = Hloc_0
    
    # note with DLR it is good do replace this with the Delta(tau) interface
    G0_iw = Gloc.copy()
    G0_iw << 0.0+0.0j
    
    G0_iw << inverse(S.Sigma_iw + inverse(Gloc))
    
    if grid == 'DLR':
        G0_iw['up'] << .5*(G0_iw['up'] + G0_iw['down'])
        G0_iw['down'] << G0_iw['up']
        
        for name, g0 in G0_iw:
            G0_iw[name] << make_hermitian(g0)
    
    Delta_iw = G0_iw.copy()
    Delta_iw << 0.0+0.0j
    for name, g0 in G0_iw:
        Delta_iw[name] << iOmega_n - inverse(g0) - solver_eal
        if grid == 'DLR':
            DLR_Diw2Dtau(S,Delta_iw)
        else:
            known_moments = make_zero_tail(Delta_iw[name], 1)
            tail, err = fit_hermitian_tail(Delta_iw[name], known_moments)
            mpi.report('tail fit error Delta_iw for block {}: {}'.format(name,err))
            S.Delta_tau[name] << make_gf_from_fourier(Delta_iw[name], S.Delta_tau.mesh, tail).real
        
    if grid == 'DLR':
        S.Delta_tau['up'] << .5*(S.Delta_tau['up'] + S.Delta_tau['down'])
        S.Delta_tau['down'] << S.Delta_tau['up']
        
        for name, D_tau in S.Delta_tau:
            S.Delta_tau[name] << make_hermitian(D_tau)
    
    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
        ar['iterations'] = it
        ar['G_0'] = G0_iw
        ar['G_0-%s'%it] = G0_iw
        ar['Delta_tau'] = S.Delta_tau
        ar['Delta_tau-%s'%it] = S.Delta_tau
#         ar['G_tau'] = S.G_tau
#         ar['G_tau-%s'%it] = S.G_tau
#         ar['G_iw'] = S.G_iw
#         ar['G_iw-%s'%it] = S.G_iw
#         ar['Sigma_iw'] = S.Sigma_iw
#         ar['Sigma_iw-%s'%it] = S.Sigma_iw
        ar['Gloc'] = Gloc
        ar['Gloc-%s'%it] = Gloc
#         ar['nimp-%s'%it] = nimp
#         ar['nlat-%s'%it] = nlat
#         ar['mu-%s'%it] = mu
        del ar

    # solve the impurity problem. The solver is performing the dyson equation as postprocessing
    S.solve(h_int=h_int, **p)
    
    if tailfit == 'DLR':
        DLR_gtau2giw(S)    
        S.Sigma_iw << inverse(G0_iw) - inverse(S.G_iw)

    if mpi.is_master_node():
        print('impurity density matrix:')
        for block, gf in S.G_iw:
            print(block)
            print(gf.density().real)
            print('--------------')
        print('total occupation {:.4f}'.format(S.G_iw.total_density().real))

    # a manual dyson equation would look like this
    # S.Sigma_iw << inverse(G0_iw) - inverse(S.G_iw)
    # TODO step 2:
    # replace this dyson eq by a fitting S.G_tau with a DLR fit and then extract Sigma
    # on DLR frequency grid to get rid of tail-fitting
    # write Fourier() replacement method with DLR

    #force self energy obtained from solver to be hermitian
    for name, s_iw in S.Sigma_iw:
        S.Sigma_iw[name] = make_hermitian(s_iw)
    
    S.Sigma_iw['up'] << .5*(S.Sigma_iw['up'] + S.Sigma_iw['down'])
    S.Sigma_iw['down'] << S.Sigma_iw['up']

    nimp = S.G_iw.total_density().real  #impurity density
    mpi.report('Impurity density is {:.4f}'.format(nimp))

    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
#         ar['iterations'] = it
#         ar['G_0'] = G0_iw
        ar['G_tau'] = S.G_tau
        ar['G_tau-%s'%it] = S.G_tau
        ar['G_iw'] = S.G_iw
        ar['G_iw-%s'%it] = S.G_iw
        ar['Sigma_iw'] = S.Sigma_iw
        ar['Sigma_iw-%s'%it] = S.Sigma_iw
        ar['nimp-%s'%it] = nimp
        ar['nlat-%s'%it] = nlat
        ar['mu-%s'%it] = mu
        del ar

