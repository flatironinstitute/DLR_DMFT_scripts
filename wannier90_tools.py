from triqs.lattice.tight_binding import TBLattice
from triqs_tprf.wannier90 import *

def load_data_generic(path, name='w2w'):
    hopping, num_wann = parse_hopping_from_wannier90_hr_dat(path + name +'_hr.dat')
    units = parse_lattice_vectors_from_wannier90_wout(path + name +'.wout')
    return hopping, units, num_wann

def get_TBL(path, name='w2w', extend_to_spin=False, add_local=None, add_field=None, renormalize=None):

    hopping, units, num_wann = load_data_generic(path, name=name)
    if extend_to_spin:
    	hopping, num_wann = extend_wannier90_to_spin(hopping, num_wann)
    if add_local is not None:
        hopping[(0,0,0)] += add_local
    if renormalize is not None:
        assert len(np.shape(renormalize)) == 1, 'Give Z as a vector'
        assert len(renormalize) == num_wann, 'Give Z as a vector of size n_orb (times two if SOC)'
        
        Z_mat = np.diag(np.sqrt(renormalize))
        for R in hopping:
            hopping[R] = np.dot(np.dot(Z_mat, hopping[R]), Z_mat)

    if add_field is not None:
        hopping[(0,0,0)] += add_field

    TBL = TBLattice(units = units, hopping = hopping, orbital_positions = [(0,0,0)]*num_wann,
                    orbital_names = [str(i) for i in range(num_wann)])
    return TBL

