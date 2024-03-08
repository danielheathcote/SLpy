import numpy as np
import pyshtools as pysh
import RFmod as RF
import SLmod as SL
import xarray as xr

from numpy import pi as pi

#########################################################
# set some constants
b    = 6368000.          # mean radius of the solid Earth 
g    = 9.825652323       # mean surface gravity
G    = 6.6723e-11        # gravitational constant
rhow = 1000.0            # density of water
rhoi =  917.0            # density of ice
rhos = 2600.0            # surface density of solid Earth
CC = 8.038e37            # polar moment of inertia 
AA = 8.012e37            # equatorial moment of inertia
Om = 2.*pi/(24.*3600.)   # mean rotation rate
Me = 5.974e24            # mass of the Earth
##########################################################


class SeaLevelSolver:


    def __init__(self, truncation_degree):
        self.truncation_degree = truncation_degree
        self.current_sea_level, self.current_ice_thickness = SL.get_sl_ice_data(truncation_degree)
        self.ocean_function = SL.ocean_function(self.current_sea_level, self.current_ice_thickness)

    def solve_fingerprint(self, zeta_glq):

        return SL.fingerprint(self.ocean_function, zeta_glq, verbose=False)
    
    def solve_adjoint_fingerprint(self, zeta_d, zeta_u_d, zeta_phi_d, kk_d):

        return SL.generalised_fingerprint(self.ocean_function, zeta_d, -1*g*zeta_u_d, -1*g*zeta_phi_d, -1*g*(kk_d+SL.rotation_vector_from_zeta_phi(zeta_phi_d)), verbose=False)
    
    def grid_to_vector(self, glq_grid):
        ## Converts GLQGrid to a vector
        return glq_grid.to_array().reshape(-1)
        
    def vector_to_grid(self, vector):
        ## Converts vector to GLQGrid
        return pysh.SHGrid.from_array(vector.reshape(self.truncation_degree+1,2*(self.truncation_degree+1)),grid='GLQ')
        
    def plot_from_grid(self, glq_grid):
        ## Plots a GLQGrid
        SL.plot(glq_grid)
        

class GraceSolver(SeaLevelSolver):


    def __init__(self, truncation_degree, observation_degree):
        super().__init__(truncation_degree)
        self.observation_degree = observation_degree

    def observation_operator(self, solution_of_fingerprint_problem):
        ## Converts a solution of the fingerprint problem to a vector of SH coefficients for phi
        
        ## For reference: Clm = phi_coeffs[0,l,m>=0] and Slm = phi_coeffs[1,l,m>=1]/
        ## phi_coeffs = pysh.expand.SHExpandGLQ(solution_of_fingerprint_problem[2].to_array(),lmax_calc=self.observation_degree)
        phi_coeffs = solution_of_fingerprint_problem[2].expand().to_array()
        ## THESE OPTIONS ARE 10^14 DIFFERENT, DUNNO WHY

        coeffs_vec = phi_coeffs[:,2:self.observation_degree+1,:self.observation_degree+1].reshape(-1)

        return coeffs_vec
    
    def observation_operator_adjoint(self, phi_coeffs_vec):
        ## Converts a vector of phi coefficients to glq grids for the generalised fingerprint problem

        phi_coeffs = np.zeros([2,self.truncation_degree+1,self.truncation_degree+1])
        phi_coeffs[:,2:self.observation_degree+1,:self.observation_degree+1] = phi_coeffs_vec.reshape(2,self.observation_degree-1,self.observation_degree+1)

        zeta_phi_d = pysh.SHGrid.from_array(pysh.expand.MakeGridGLQ(phi_coeffs, extend=1), grid='GLQ')

        ## Create a null grid with the same dimensions as zeta_phi_d
        null_grid = pysh.SHGrid.from_zeros(lmax=self.truncation_degree,grid = 'GLQ')

        ## Return a quadruple of zeta_d, t_d, zeta_phi_d, kk_d
        return null_grid, null_grid, zeta_phi_d, np.array([0,0])
    
    def __matmul__(self,zeta_vector):
        return self.observation_operator(self.solve_fingerprint(self.vector_to_grid(zeta_vector)))


