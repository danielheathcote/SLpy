import numpy as np
import pyshtools as pysh
import RFmod as RF
import SLmod as SL
import xarray as xr

from numpy import pi as pi


class SeaLevelSolver:


    def __init__(self, truncation_degree):
        self.truncation_degree = truncation_degree
        self.current_sea_level, self.current_ice_thickness = SL.get_sl_ice_data(truncation_degree)
        self.ocean_function = SL.ocean_function(self.current_sea_level, self.current_ice_thickness)

    def solve_fingerprint(self, zeta_glq):

        return SL.fingerprint(self.ocean_function, zeta_glq)
    
    def solve_adjoint_fingerprint(self, zeta_d, zeta_u_d, zeta_phi_d, kk_d):

        return SL.generalised_fingerprint(self.ocean_function, zeta_d, zeta_u_d, zeta_phi_d, kk_d)
    
    def grid_to_vector(self, grid):
        ## Converts GLQGrid to a vector
        return grid.to_array().reshape(-1)
        
    def vector_to_grid(self, vector):
        ## Converts vector to GLQGrid
        return pysh.SHGrid.from_array(vector.reshape(self.truncation_degree+1,2*(self.truncation_degree+1)),grid='GLQ')
        
    def plot_from_grid(self, grid):
        ## Plots a GLQGrid
        SL.plot(grid)
        

class GraceSolver(SeaLevelSolver):


    def __init__(self, truncation_degree, observation_degree ):
        super().__init__(truncation_degree)
        self.observation_degree = observation_degree

    def solve_for_grace_coefficients(self,zeta):

        return self.solve_fingerprint(zeta)

    def observation_operator(self, solution):

        coeffs = pysh.expand.SHExpandGLQ(solution[2].to_array(),lmax_calc=self.observation_degree)
        coeffs_vec = np.zeros(((self.observation_degree+1)**2)-4)
        for l in range(2,self.observation_degree+1):
            coeffs_vec[((l)**2)-4:((l+1)**2)-4] = np.concatenate((coeffs[1,l,1:l+1][::-1],coeffs[0,l,0:l+1]))
        return coeffs_vec
    
    def __matmul__(self,zeta_vector):
        return self.observation_operator(self.solve_for_grace_coefficients(self.vector_to_grid(zeta_vector)))


