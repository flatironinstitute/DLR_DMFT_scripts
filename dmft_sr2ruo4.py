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
U = 2.3                   # hubbard U parameter
J = 0.40                  # hubbard J parameter
nloops = 10               # number of DMFT loops needs 5-10 loops to converge
nk = 30                   # number of k points in each dimension
density_required = 4.     # target density for setting the chemical potential
n_orb = 3                 # number of orbitals
mu = 11.3715              # chemical potential
add_spin = False
w90_seedname = 'sr2ruo4'
w90_pathname = './'

outfile = 'sro_U%.1f'%U

p = {}
# solver
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 200
p["n_warmup_cycles"] = int(1e4)
p["n_cycles"] = int(4e7/mpi.size)
# tail fit
# turn this off to get the raw QMC results without fitting
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4
p["fit_min_w"] = 3
p["fit_max_w"] = 10

S = Solver(beta=beta, gf_struct = [('up', 3), ('down', 3)], n_iw = 1025, n_tau=10001)

# set up interaction Hamiltonian
Umat, Upmat = util.U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=J)
h_int = util.h_int_kanamori(['up', 'down'], range(n_orb), off_diag=True, U=Umat, Uprime=Upmat, J_hund=J)

Gloc = S.G_iw.copy() #local Green's function

# set up Wannier Hamiltonian
H_add_loc = np.zeros((n_orb, n_orb), dtype=complex)
H_add_loc += np.diag([-mu]*n_orb)

L = get_TBL(path=w90_pathname, name=w90_seedname, extend_to_spin=add_spin, add_local=H_add_loc)

SK = SumkDiscreteFromLattice(lattice=L, n_points=nk)

def sumk(mu, Sigma, bz_weights, hopping, iw_vec = None):
    '''
    calc Gloc with mpi parallelism
    '''
    Gloc = Sigma.copy()
    Gloc << 0.0+0.0j

    n_orb = Gloc['up'].target_shape[0]

    if not iw_vec:
        iw_mat = np.array([iw.value * np.eye(n_orb) for iw in Gloc.mesh])
    else:
        iw_values =  np.array(list(Gloc.mesh.values()))[vals]
        iw_mat = np.array([iw * np.eye(n_orb) for iw in iw_values])

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
    dens =  sumk(mu = mu, Sigma = S.Sigma_iw, bz_weights=SK.bz_weights, hopping=SK.hopping).total_density()
    if abs(dens.imag) > 1e-20:
            mpi.report("Warning: Imaginary part of density will be ignored ({})".format(str(abs(dens.imag))))
    return dens.real

#check if there are previous runs in the outfile and if so restart from there
previous_runs = 0
previous_present = False
mu = 0.
if mpi.is_master_node():
    ar = HDFArchive(outfile+'.h5','a')
    if 'iterations' in ar:
        previous_present = True
        previous_runs = ar['iterations']
        S.Sigma_iw = ar['Sigma_iw']
        mu = ar['mu-%d'%previous_runs]
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

    # note with DLR it is good do replace this with the Delta(tau) interface
    S.G0_iw << inverse(S.Sigma_iw + inverse(Gloc))
    # solve the impurity problem. The solver is performing the dyson equation as postprocessing
    S.solve(h_int=h_int, **p)

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

    nimp = S.G_iw.total_density().real  #impurity density
    mpi.report('Impurity density is {:.4f}'.format(nimp))

    if mpi.is_master_node():
        ar = HDFArchive(outfile+'.h5','a')
        ar['iterations'] = it
        ar['G_0'] = S.G0_iw
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



