#DMFT calculation for a hubbard model on a 2D square lattice
#uses Triqs version 3.0.x

from triqs.sumk import *
from triqs.gf import *
from triqs.lattice.tight_binding import TBLattice
import triqs.utility.mpi as mpi
from triqs.utility.dichotomy import dichotomy
from triqs_cthyb import *
from h5 import HDFArchive
from triqs.operators import *

beta = 40.
t = -1.                   #nearest neighbor hopping
tp = 0.                   #next nearest neighbor hopping
U = 3                     # hubbard U parameter
nloops = 10               # number of DMFT loops needs 5-10 loops to converge
nk = 30                   # number of k points in each dimension
density_required = 1.     # target density for setting the chemical potential

outfile = 'U%.1f'%U

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

S = Solver(beta=beta, gf_struct = [('up', [0]), ('down', [0])])
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

#function to extract density for a given mu, to be used by dichotomy function to determine mu
def Dens(mu):
    # calling the k sum here
    # github.com/TRIQS/triqs/blob/3.0.x/python/triqs/sumk/sumk_discrete.py#L73
    dens =  SK(mu = mu, Sigma = S.Sigma_iw).total_density()
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
    mu, density = dichotomy(Dens, mu, density_required, 1e-4, .5, max_loops = 100, x_name="chemical potential", y_name="density", verbosity=3)
    # calling the k sum here which you need to manually implement or change
    # but it is all python, even the invert is done on a numpy array
    # github.com/TRIQS/triqs/blob/3.0.x/python/triqs/sumk/sumk_discrete.py#L73

    # TODO step 3 write a new python function which replaces the SK() call which accepts a iw_vector and does the k sum only on these frequencies
    # include timing! 
    Gloc << SK(mu = mu, Sigma = S.Sigma_iw)

    # TODO step 1
    # extract special Gloc(iw) points and do the DLR and back
    # iw_freq = [1, 10, 55, ..]
    # Gloc_spec = Gloc['up].data[iw_freq,:,:] #Gloc_spec is now a np array
    # DLR fit
    # evaluate DLR fit at all iw frequencies
    # see if the result is the same
        
    nlat = Gloc.total_density().real # lattice density

    # set starting guess for Sigma = U/2 at first iteration
    if it == 1:
        S.Sigma_iw << .5*U

    # note with DLR it is good do replace this with the Delta(tau) interface
    S.G0_iw << inverse(S.Sigma_iw + inverse(Gloc))
    # solve the impurity problem. The solver is performing the dyson equation as postprocessing
    S.solve(h_int=h_int, **p)

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



