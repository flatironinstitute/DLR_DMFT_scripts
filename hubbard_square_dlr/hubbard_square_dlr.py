#DMFT calculation for a hubbard model on a 2D square lattice
#uses Triqs version 3.0.x
import numpy as np
import sys, os, time, random
from triqs.sumk import *
from triqs.gf import *
from triqs.lattice.tight_binding import TBLattice
import triqs.utility.mpi as mpi
from triqs.utility.dichotomy import dichotomy
from triqs_cthyb import *
from h5 import HDFArchive
from triqs.operators import *

from timeit import default_timer as timer

beta = 40.
t = -1.                   #nearest neighbor hopping
tp = 0.                   #next nearest neighbor hopping
U = 3                     # hubbard U parameter
nloops = 50               # number of DMFT loops
nk = 30                   # number of k points in each dimension
density_required = 1.     # target density for setting the chemical potential

eps = 10**-6              # DLR accuracy

outfile = f'U{U:.1F}_{eps:.1E}'

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 200
p["n_warmup_cycles"] = int(1e4)
p["n_cycles"] = int(1e7/mpi.size)

# tail fit
# turn this off to get the raw QMC results without fitting
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4
p["fit_min_w"] = 5
p["fit_max_w"] = 15

l_DLR_gloc2gloc = False

if l_DLR_gloc2gloc:
    outfile += '_gloc2gloc'

l_symmetrize = False
if l_symmetrize:
    outfile += '_symmetrize'

if p["perform_tail_fit"]:
    outfile += '_fit'
    
l_read = False
if l_read:
    readfile = outfile
    outfile += '_test'
    
l_previous_runs = False
if l_previous_runs:
    previous_runs = 7    

S = Solver(beta=beta, gf_struct = [('up', [0]), ('down', [0])])

if l_DLR_gloc2gloc:
    rf = np.loadtxt(f'dlrrf_{round(beta)}_{eps:.1E}.dat')
    rf.sort()
    mf = np.loadtxt(f'dlrmf_{round(beta)}_{eps:.1E}.dat',dtype=int)
    mf.sort()
    index = mf - (-S.n_iw)

h_int = U * n('up',0) * n('down',0) #local interating hamiltonian

Gloc = S.G_iw.copy() #local Green's function

hop= {  (1,0)  :  [[ t]],
        (-1,0) :  [[ t]],
        (0,1)  :  [[ t]],
        (0,-1) :  [[ t]],
        (1,1)  :  [[ tp]],
        (-1,-1):  [[ tp]],
        (1,-1) :  [[ tp]],
        (-1,1) :  [[ tp]]}

L = TBLattice(units = [(1, 0, 0) , (0, 1, 0)], hopping = hop, orbital_names= range(1), orbital_positions= [(0., 0., 0.)]*1)

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
        DLR_gloc2gloc(Gloc,beta,eps)

    return Gloc

#function to extract density for a given mu, to be used by dichotomy function to determine mu
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

def kernel_it(τ,ω): 
    assert τ >= 0.
    if ω > 0.:
        return np.exp(-ω*τ)/(1+np.exp(-ω))
    else:
        return np.exp(ω*(1-τ))/(1+np.exp(ω))
    
def kernel_mf(n,ω):    
    return 1/(ω-(2*n+1)*np.pi*1j)

def symmetrize(vec):
    
    assert len(vec) % 2 == 0
    
    n_iw = round(len(vec)/2)
    
    for n in range(n_iw):
        real = np.real((vec[n_iw + n] + vec[n_iw - 1 - n])/2)
        imag = np.imag((vec[n_iw + n] - vec[n_iw - 1 - n])/2)
        vec[n_iw + n] = real + imag * 1j
        vec[n_iw - 1 - n] = real - imag * 1j
    
    return

# def tail_fit(vec,vec_ref):
    
#     assert np.shape(vec) == np.shape(vec_ref)
    
#     assert len(vec) % 2 == 0
    
#     n_iw = round(len(vec)/2)
    
# #     for n in range(n_iw):
# #         if np.abs(n) > 320:
# #             vec[n_iw + n] = vec_ref[n_iw + n]
# #             vec[n_iw - 1 - n] = vec_ref[n_iw - 1 - n]

#     thres = 321
#     vec[(n_iw + thres):].imag = vec_ref[(n_iw + thres):].imag
#     vec[:(n_iw - thres)].imag = vec_ref[:(n_iw - thres)].imag
    
#     return   

def DLR_gtau2giw(S,beta,eps):
    
    α = {}
    
    rf = np.loadtxt(f'dlrrf_{round(beta)}_{eps:.1E}.dat')
    rf.sort()
    
    block_lst = list(['up','down'])
    
    for i, block in enumerate(block_lst,1):

        it = np.array([i / beta for i in S.G_tau[block].mesh.values()])

        κ_it = np.zeros((len(it),len(rf)))
        for i,τ in enumerate(it):
            for j,ω in enumerate(rf):
                κ_it[i,j] = kernel_it(τ,ω)

        from scipy.linalg import lstsq
        α[block] = np.array(lstsq(κ_it,S.G_tau[block].data[:, 0, 0].real)[0])
        
        tail_shift = ( np.sum(α[block]) + 1.0 ) / 2 
        α[block][0] -= tail_shift
        α[block][-1] -= tail_shift
        
        S.G_tau[block].data[:, 0, 0] = (κ_it@α[block]).astype(complex)
        
    for i, block in enumerate(block_lst,1):
        
        mf = np.array([round((iω.imag*beta/np.pi-1)/2) for iω in S.G_iw[block].mesh.values()])
        
        κ_mf = np.zeros((len(mf),len(rf)),dtype=complex)
        for i,n in enumerate(mf):
            for j,ω in enumerate(rf):
                κ_mf[i,j] = kernel_mf(n,ω)
        
        S.G_iw[block].data[:, 0, 0] = (κ_mf@α[block]*beta).astype(complex)

    return

def DLR_g02g0(S,beta,eps):
    
    α = {}
    
    rf = np.loadtxt(f'dlrrf_{round(beta)}_{eps:.1E}.dat')
    rf.sort()
    
    block_lst = list(['up','down'])
    
    for i, block in enumerate(block_lst,1):
        
        mf = np.array([round((iω.imag*beta/np.pi-1)/2) for iω in S.G0_iw[block].mesh.values()])
        
        κ_mf = np.zeros((len(mf),len(rf)),dtype=complex)
        for i,n in enumerate(mf):
            for j,ω in enumerate(rf):
                κ_mf[i,j] = kernel_mf(n,ω)
        
        from scipy.linalg import lstsq
        α[block] = np.array(lstsq(κ_mf,S.G0_iw[block].data[:, 0, 0])[0])
        
        S.G0_iw[block].data[:, 0, 0] = (κ_mf@α[block]).astype(complex)
    
    return

def DLR_gloc2gloc(Gloc,beta,eps):
    
    α = {}
    
#     rf = np.loadtxt(f'dlrrf_{round(beta)}_{eps:.1E}.dat')
#     rf.sort()
    
#     mf = np.loadtxt(f'dlrmf_{round(beta)}_{eps:.1E}.dat',dtype=int)
#     mf.sort()
#     index = mf - (-S.n_iw)
    
    assert len(rf) == len(mf)
    
    block_lst = list(['up','down'])
    
    for i, block in enumerate(block_lst,1):
        
        κ_mf = np.zeros((len(mf),len(rf)),dtype=complex)
        for i,n in enumerate(mf):
            for j,ω in enumerate(rf):
                κ_mf[i,j] = kernel_mf(n,ω)
        
#         from scipy.linalg import lstsq
#         α[block] = np.array(lstsq(κ_mf,Gloc[block].data[index, 0, 0])[0])
        
#         min ||A @ x - b||
#         subject to  C @ x = d
#         can be obtained using the Python code

#         from scipy.linalg import lapack

#         # Define the matrices as usual, then
#         x = lapack.dgglse(A, C, b, d)[3]
    
#         from scipy.linalg import lapack
#         x = lapack.zgglse(κ_mf, np.ones([1,len(rf)]), Gloc[block].data[index, 0, 0], np.array([[1]]))[3]
    
        α[block] = np.linalg.solve(κ_mf,Gloc[block].data[index, 0, 0])
    
        tail_shift = ( np.sum(α[block]) + beta ) / 2 
        α[block][0] -= tail_shift
        α[block][-1] -= tail_shift
        
        mf_all = np.array([round((iω.imag*beta/np.pi-1)/2) for iω in Gloc[block].mesh.values()])
        
        assert np.abs(mf_all[0]) == S.n_iw
        
        κ_mf_all = np.zeros((len(mf_all),len(rf)),dtype=complex)
        for i,n in enumerate(mf_all):
            for j,ω in enumerate(rf):
                κ_mf_all[i,j] = kernel_mf(n,ω)
        
        Gloc[block].data[:, 0, 0] = (κ_mf_all@α[block]).astype(complex)
        
#         symmetrize(Gloc[block].data[:, 0, 0])
    
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
    
    if l_DLR_gloc2gloc:
        Gloc << sumk(mu = mu, Sigma= S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping, iw_index=index)
    else:
        Gloc << SK(mu = mu, Sigma = S.Sigma_iw)
    
    mpi.barrier()
    mpi.report('time for k sum: {:.2f} s'.format(timer() - start_time))
    
    # TODO step 1
    # extract special Gloc(iw) points and do the DLR and back
    # iw_freq = [1, 10, 55, ..]
    # Gloc_spec = Gloc['up].data[iw_freq,:,:] #Gloc_spec is now a np array
    # DLR fit
    # evaluate DLR fit at all iw frequencies
    # see if the result is the same
    
#     if it > 1 and l_DLR_gloc2gloc:
#         DLR_gloc2gloc(Gloc,beta,eps)
    
#     if mpi.is_master_node():
#         ar = HDFArchive(outfile+'.h5','a')
#         ar['iterations'] = it
# #
#         ar['Gloc'] = Gloc
#         ar['Gloc-%s'%it] = Gloc
# #         ar['nimp-%s'%it] = nimp
# #         ar['nlat-%s'%it] = nlat
# #         ar['mu-%s'%it] = mu
#         del ar
    
    nlat = Gloc.total_density().real # lattice density
    mpi.report('Gloc density: {:.3f}'.format(nlat))
    # set starting guess for Sigma = U/2 at first iteration
    if it == 1:
        S.Sigma_iw << .5*U
    
    S.G0_iw << inverse(S.Sigma_iw + inverse(Gloc))
    
    if l_symmetrize:
        S.G0_iw['up'] << .5*(S.G0_iw['up'] + S.G0_iw['down'])
        S.G0_iw['down'] << S.G0_iw['up']

        block_lst = list(['up','down'])
        for i, block in enumerate(block_lst,1):
            symmetrize(S.G0_iw[block].data[:,0,0])
    
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
    
#    DLR_g02g0(S,beta,eps)
    
    # solve the impurity problem. The solver is performing the dyson equation as postprocessing
    S.solve(h_int=h_int, **p)
        
#     DLR_gtau2giw(S,beta,eps)    
#     S.Sigma_iw << inverse(S.G0_iw) - inverse(S.G_iw)
    
    # a manual dyson equation would look like this:
    # S.Sigma_iw << inverse(S.G0_iw) - inverse(S.G_iw)

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
#         ar['G_0'] = S.G0_iw
#         ar['G_0-%s'%it] = S.G0_iw
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

