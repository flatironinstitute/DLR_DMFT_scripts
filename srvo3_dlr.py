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

from timeit import default_timer as timer
from wannier90_tools import *

np.set_printoptions(precision=6,suppress=True)

beta = 40.
U = 8.0                   # hubbard U parameter
J = 0.65                  # hubbard J parameter
nloops = 100              # number of DMFT loops needs 5-10 loops to converge
nk = 21                   # number of k points in each dimension
density_required = 1.     # target density for setting the chemical potential
n_orb = 3                 # number of orbitals
mu = 12.3958              # chemical potential
add_spin = False
w90_seedname = 'srvo3'
w90_pathname = './'

eps = 10**-6              # DLR accuracy

outfile = f'U{U:.1F}_{eps:.1E}'

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 120
p["n_warmup_cycles"] = int(1e4)
p["n_cycles"] = int(4e7/mpi.size)
# p["imag_threshold"] = 1e-07
p["imag_threshold"] = 1e-12
# tail fit
# turn this off to get the raw QMC results without fitting
l_tailfit = False
if l_tailfit: 
    p["perform_tail_fit"] = True
    p["fit_max_moment"] = 4
    p["fit_min_w"] = 5
    p["fit_max_w"] = 12
else:
    p["perform_tail_fit"] = False
    
l_DLR_gloc2gloc = True
type_DLR_gloc2gloc = 'crude'
if l_DLR_gloc2gloc:
    outfile += '_gloc2gloc'
    outfile += f'_{type_DLR_gloc2gloc}'
    if type_DLR_gloc2gloc == 'crude':
        from scipy.linalg import lstsq
    elif type_DLR_gloc2gloc == 'constrained':
        from scipy.linalg import lapack
        
l_DLR_gtau2giw = True
type_DLR_gtau2giw = 'crude'
if l_DLR_gtau2giw:
    outfile += '_gtau2giw'
    outfile += f'_{type_DLR_gtau2giw}'
    if type_DLR_gtau2giw == 'crude':
        from scipy.linalg import lstsq
    elif type_DLR_gtau2giw == 'constrained':
        from scipy.linalg import lapack
        
l_DLR_g02g0 = False
type_DLR_g02g0 = 'crude'
if l_DLR_g02g0:
    outfile += '_g02g0'
    outfile += f'_{type_DLR_g02g0}'
    if type_DLR_gtau2giw == 'crude':
        from scipy.linalg import lstsq

l_symmetrize = True
if l_symmetrize:
    outfile += '_symmetrize'

if p["perform_tail_fit"]:
    outfile += '_tailfit'
    
l_read = False
if l_read:
    readfile = outfile
    outfile += '_test'
    
l_previous_runs = False
if l_previous_runs:
    previous_runs = 7 
    
S = Solver(beta=beta, gf_struct = [('up', 3), ('down', 3)], n_iw = 1025, n_tau=10001)

block_lst = list(['up','down'])

Gloc = S.G_iw.copy() #local Green's function

def kernel_it(τ,ω): 
    assert τ >= 0.
    if ω > 0.:
        return np.exp(-ω*τ)/(1+np.exp(-ω))
    else:
        return np.exp(ω*(1-τ))/(1+np.exp(ω))
    
def kernel_mf(n,ω):    
    return 1/(ω-(2*n+1)*np.pi*1j)

if l_DLR_gloc2gloc or l_DLR_gtau2giw or l_DLR_g02g0:
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
            
    it_all = np.arange(0,S.n_tau) / S.n_tau
    κ_it_all = np.zeros((len(it_all),len(rf)))
    for i,τ in enumerate(it_all):
        for j,ω in enumerate(rf):
            κ_it_all[i,j] = kernel_it(τ,ω)
    
# set up interaction Hamiltonian
Umat, Upmat = util.U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)
h_int = util.h_int_kanamori(['up', 'down'], range(n_orb), off_diag=True, U=Umat, Uprime=Upmat, J_hund=J, H_dump='H_int.txt')

# set up Wannier Hamiltonian
H_add_loc = np.zeros((n_orb, n_orb), dtype=complex)
H_add_loc += np.diag([-mu]*n_orb)

L = get_TBL(path=w90_pathname, name=w90_seedname, extend_to_spin=add_spin, add_local=H_add_loc)

SK = SumkDiscreteFromLattice(lattice=L, n_points=nk)

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
    if l_DLR_gloc2gloc:
        dens =  sumk(mu = mu, Sigma = S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping, iw_index=index).total_density()
    else:
        dens =  sumk(mu = mu, Sigma = S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping).total_density()
    if abs(dens.imag) > 1e-20:
            mpi.report("Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
    return dens.real

def symmetrize(vec):
    
    assert len(vec) % 2 == 0
    
    n_iw = round(len(vec)/2)
    
    for n in range(n_iw):
        real = np.real((vec[n_iw + n] + vec[n_iw - 1 - n])/2)
        imag = np.imag((vec[n_iw + n] - vec[n_iw - 1 - n])/2)
        vec[n_iw + n] = real + imag * 1j
        vec[n_iw - 1 - n] = real - imag * 1j
    
    return

def DLR_gtau2giw(S):
    
    α = {}
    
    for i, block in enumerate(block_lst,1):
        for i_orb in range(n_orb):
            for j_orb in range(n_orb):
                if type_DLR_gtau2giw == 'constrained':
                    if i_orb == j_orb:
                        α[block] = lapack.zgglse(κ_it_all, np.ones([1,len(rf)]), S.G_tau[block].data[:,i_orb,j_orb], np.array([[1]]))[3]
                    else:
                        α[block] = np.array(lstsq(κ_it_all,S.G_tau[block].data[:,i_orb,j_orb].real)[0])
                elif type_DLR_gtau2giw == 'crude':
                    α[block] = np.array(lstsq(κ_it_all,S.G_tau[block].data[:,i_orb,j_orb].real)[0])
                    if i_orb == j_orb:
                        tail_shift = ( np.sum(α[block]) + 1.0 ) / 2 
                        α[block][0] -= tail_shift
                        α[block][-1] -= tail_shift
                else:
                    print('Error: unknown type of DLR_gtau2giw!')

                S.G_tau[block].data[:,i_orb,j_orb] = (κ_it_all@α[block]).astype(complex)
                S.G_iw[block].data[:,i_orb,j_orb] = (κ_mf_all@α[block]*beta).astype(complex)

    return

def DLR_g02g0(S):
    
    α = {}
        
    for i, block in enumerate(block_lst,1):    
        for i_orb in range(n_orb):
            for j_orb in range(n_orb):
                if type_DLR_g02g0 == 'crude':
                    α[block] = np.array(lstsq(κ_mf_all,S.G0_iw[block].data[:,i_orb,j_orb])[0])
                    if i_orb == j_orb:
                        tail_shift = ( np.sum(α[block]) + beta ) / 2 
                        α[block][0] -= tail_shift
                        α[block][-1] -= tail_shift
                else:
                    print('Error: unknown type of DLR_g02g0!')

                S.G0_iw[block].data[:,i_orb,j_orb] = (κ_mf_all@α[block]).astype(complex)
    
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
            
                if type_DLR_gloc2gloc == 'constrained':
                    if i_orb == j_orb:
                        α[block] = lapack.zgglse(κ_mf, np.ones([1,len(rf)]), Gloc[block].data[index,i_orb,j_orb], np.array([[1]]))[3]
                    else:
                        α[block] = np.linalg.solve(κ_mf,Gloc[block].data[index,i_orb,j_orb])
                elif type_DLR_gloc2gloc == 'crude':
                    α[block] = np.linalg.solve(κ_mf,Gloc[block].data[index,i_orb,j_orb])
                    if i_orb == j_orb:
                        tail_shift = ( np.sum(α[block]) + beta ) / 2 
                        α[block][0] -= tail_shift
                        α[block][-1] -= tail_shift
                else:
                    print('Error: unknown type of DLR_gloc2gloc!')
                    
                Gloc[block].data[:,i_orb,j_orb] = (κ_mf_all@α[block]).astype(complex)
        
#                 symmetrize(Gloc[block].data[:, i_orb, j_orb])
    
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

    # symmetrize Sigma
    if it > 1:
        S.Sigma_iw['up'] << .5*(S.Sigma_iw['up'] + S.Sigma_iw['down'])
        S.Sigma_iw['down'] << S.Sigma_iw['up']

        # all three orb are degenerate
        S.Sigma_iw['up'] << 0.0+0.0j
        for i_orb in range(n_orb):
            S.Sigma_iw['up'][0,0] << S.Sigma_iw['up'][0,0] + (S.Sigma_iw['down'][i_orb,i_orb]/n_orb)

        # write to all orbitals
        for block, gf in S.Sigma_iw:
            for i_orb in range(n_orb):
                S.Sigma_iw[block][i_orb,i_orb] << S.Sigma_iw['up'][0,0]

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
    if l_DLR_gloc2gloc:
        Gloc << sumk(mu = mu, Sigma= S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping, iw_index=index)
    else:
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

    # note with DLR it is good do replace this with the Delta(tau) interface
    S.G0_iw << inverse(S.Sigma_iw + inverse(Gloc))
    
    if l_symmetrize:
        S.G0_iw['up'] << .5*(S.G0_iw['up'] + S.G0_iw['down'])
        S.G0_iw['down'] << S.G0_iw['up']
        
        # all three orb are degenerate
        S.G0_iw['up'] << 0.0+0.0j
        for i_orb in range(n_orb):
            S.G0_iw['up'][0,0] << S.G0_iw['up'][0,0] + (S.G0_iw['down'][i_orb,i_orb]/n_orb)

        # write to all orbitals
        for block, gf in S.Sigma_iw:
            for i_orb in range(n_orb):
                S.G0_iw[block][i_orb,i_orb] << S.G0_iw['up'][0,0]

        for i, block in enumerate(block_lst,1):
            for i_orb in range(n_orb):
                for j_orb in range(n_orb):
                    symmetrize(S.G0_iw[block].data[:,i_orb,j_orb])
                    
    if l_DLR_g02g0:
        DLR_g02g0(S)
    
    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
        ar['iterations'] = it
        ar['G_0'] = S.G0_iw
        ar['G_0-%s'%it] = S.G0_iw
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
    
    if l_DLR_gtau2giw:
        DLR_gtau2giw(S)    
        S.Sigma_iw << inverse(S.G0_iw) - inverse(S.G_iw)

    if mpi.is_master_node():
        print('impurity density matrix:')
        for block, gf in S.G_iw:
            print(block)
            print(gf.density().real)
            print('--------------')
        print('total occupation {:.4f}'.format(S.G_iw.total_density().real))

    # a manual dyson equation would look like this
    # S.Sigma_iw << inverse(S.G0_iw) - inverse(S.G_iw)
    # TODO step 2:
    # replace this dyson eq by a fitting S.G_tau with a DLR fit and then extract Sigma
    # on DLR frequency grid to get rid of tail-fitting
    # write Fourier() replacement method with DLR

    #force self energy obtained from solver to be hermitian
    for name, s_iw in S.Sigma_iw:
        S.Sigma_iw[name] = make_hermitian(s_iw)
    
    S.Sigma_iw['up'] << .5*(S.Sigma_iw['up'] + S.Sigma_iw['down'])
    S.Sigma_iw['down'] << S.Sigma_iw['up']

    # all three orb are degenerate
    S.Sigma_iw['up'] << 0.0+0.0j
    for i_orb in range(n_orb):
        S.Sigma_iw['up'][0,0] << S.Sigma_iw['up'][0,0] + (S.Sigma_iw['down'][i_orb,i_orb]/n_orb)

    # write to all orbitals
    for block, gf in S.Sigma_iw:
        for i_orb in range(n_orb):
            S.Sigma_iw[block][i_orb,i_orb] << S.Sigma_iw['up'][0,0]
                
    nimp = S.G_iw.total_density().real  #impurity density

    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
#         ar['iterations'] = it
#         ar['G_0'] = S.G0_iw
#         ar['G_0-%s'%it] = S.G0_iw
        ar['Delta_tau'] = S.Delta_tau
        ar['Delta_tau-%s'%it] = S.Delta_tau
        ar['Delta_infty'] = S.Delta_infty
        ar['Delta_infty-%s'%it] = S.Delta_infty
        ar['G_tau'] = S.G_tau
        ar['G_tau-%s'%it] = S.G_tau
        ar['G_iw'] = S.G_iw
        ar['G_iw-%s'%it] = S.G_iw
        ar['Sigma_iw'] = S.Sigma_iw
        ar['Sigma_iw-%s'%it] = S.Sigma_iw
#         ar['Gloc'] = Gloc
#         ar['Gloc-%s'%it] = Gloc
        ar['nimp-%s'%it] = nimp
        ar['nlat-%s'%it] = nlat
        ar['mu-%s'%it] = mu
        del ar




