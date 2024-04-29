import numpy as np
import pyshtools as pysh
import RFmod as RF
import SLmod as SL
import xarray as xr
from abc import ABC, abstractmethod

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

        return SL.fingerprint(self.ocean_function, zeta_glq, verbose=False, rotation = True)
    
    def solve_adjoint_fingerprint(self, zeta_d, zeta_u_d, zeta_phi_d, kk_d):

        return SL.generalised_fingerprint(self.ocean_function, zeta_d, -1*zeta_u_d, -1*g*zeta_phi_d, -1*g*(kk_d-SL.rotation_vector_from_zeta_phi(zeta_phi_d)), verbose=False, rotation = True)
    
    def load_inner_product(self, zeta1, zeta2):
        
        return SL.surface_integral(zeta1*zeta2)

    def solution_inner_product(self, sl1, u1, phi1, om1, sl2, u2, phi2, om2):

        return SL.surface_integral(sl1*sl2 + u1*u2 + phi1*phi2) + np.inner(om1,om2)

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


    @staticmethod
    def size_of_data_vector(observation_degree):
        return ((observation_degree+1)**2)-4
    
    def __init__(self, truncation_degree, observation_degree):
        super().__init__(truncation_degree)
        self.observation_degree = observation_degree
        size = self.size_of_data_vector(observation_degree)
        self.measurement_error_covariance_matrix = np.zeros((size,size))

    def observation_operator(self, sl, u, phi, om, psi=0):
        ## Converts a solution of the fingerprint problem to a vector of SH coefficients for phi
        ## For reference: Clm = phi_coeffs[0,l,m>=0] and Slm = phi_coeffs[1,l,m>=1]/
        phi_coeffs = phi.expand(normalization = 'ortho').to_array()

        phi_coeffs_vec = np.zeros(((self.observation_degree+1)**2)-4)       
        for l in range(2,self.observation_degree+1):
            phi_coeffs_vec[((l)**2)-4:((l+1)**2)-4] = np.concatenate((phi_coeffs[1,l,1:l+1][::-1],phi_coeffs[0,l,0:l+1]))
        
        return phi_coeffs_vec
    
    def observation_operator_adjoint(self, phi_coeffs_vec):
        ## Converts a vector of phi coefficients to glq grids for the generalised fingerprint problem

        phi_coeffs = np.zeros((2,self.truncation_degree+1,self.truncation_degree+1))
        ## For reference: Clm = phi_coeffs[0,l,m>=0] and Slm = phi_coeffs[1,l,m>=1]
        for l in range(2,self.observation_degree+1):
            phi_coeffs[1,l,1:l+1] = phi_coeffs_vec[(l**2)-4:(l**2)-4+l][::-1]
            phi_coeffs[0,l,0:l+1] = phi_coeffs_vec[(l**2)-4+l:((l+1)**2)-4]

        zeta_phi_d = pysh.SHGrid.from_array(pysh.expand.MakeGridGLQ(phi_coeffs, extend=1, norm = 4), grid='GLQ')/(b**2)

        ## Create a null grid with the same dimensions as zeta_phi_d
        null_grid = pysh.SHGrid.from_zeros(lmax=self.truncation_degree,grid = 'GLQ')

        ## Return a quadruple of zeta_d, t_d, zeta_phi_d, kk_d
        return null_grid, null_grid, zeta_phi_d, np.array([0,0])
    
    def forward_operator(self, zeta_glq):
        ## The operator A: takes a glq grid of zeta and returns a vector of SH coefficients for phi
        ## Could be extended to take a vector representing zeta directly

        return self.observation_operator(*self.solve_fingerprint(zeta_glq))
        
    def adjoint_operator(self, phi_coeffs_vec):
        ## The adjoint operator A*: takes a vector of SH coefficients for phi and returns a glq grid of zeta

        return self.solve_adjoint_fingerprint(*self.observation_operator_adjoint(phi_coeffs_vec))
    
    def data_inner_product(self, phi1, phi2):

        return np.inner(phi1,phi2)
    
    def set_measurement_error_covariance_matrix(self, matrix):

        self.measurement_error_covariance_matrix = matrix
    
    def scale_measurement_error_covariance_matrix(self, factor):

        self.measurement_error_covariance_matrix = factor*self.measurement_error_covariance_matrix

    def add_to_measurement_error_covariance_matrix(self, matrix):

        self.measurement_error_covariance_matrix += matrix

    def R_covariance_operator(self, vector):

        pass

    def generate_sample_from_covariance(self):

        pass
    

class PropertyClass(ABC):


    def __init__(self, truncation_degree):
        self.truncation_degree = truncation_degree

    @abstractmethod
    def weighting_function(self, i):
        pass

    @abstractmethod
    def length_of_property_vector(self):
        pass

    def forward_property_operator(self, zeta_glq):
        ## The operator P:

        property_vector = np.zeros(self.length_of_property_vector())

        for i in range(self.length_of_property_vector()):
            property_vector[i] = SL.surface_integral(self.weighting_function(i)*zeta_glq)

        return property_vector
    
    def adjoint_property_operator(self, property_vector):
        ## The adjoint operator P*:

        adjoint_property_grid = pysh.SHGrid.from_zeros(lmax = self.truncation_degree, grid = 'GLQ')

        for i in range(self.length_of_property_vector()):
            adjoint_property_grid += property_vector[i]*self.weighting_function(i)

        return adjoint_property_grid
    
    def property_inner_product(self, p1, p2):

        return np.inner(p1,p2)
    

class PropertyClassGaussian(PropertyClass):


    def __init__(self, truncation_degree, gaussian_params = [(100,0,0)]):
        ## gaussian_params is a list of tuples (width, lat0, lon0) for each Gaussian averaging function

        super().__init__(truncation_degree)
        self.vector_of_weighting_functions = [SL.gaussian_averaging_function(L = self.truncation_degree, r = width, lat0 = lat, lon0 = lon) for width, lat, lon in gaussian_params]

    def generate_gaussian_averaging_function(self, index, width, lat0, lon0):
        ## Generates a Gaussian averaging function centred at lat0, lon0

        self.vector_of_weighting_functions[index] = SL.gaussian_averaging_function(L = self.truncation_degree, r = width, lat0 = lat0, lon0 = lon0)  

    def weighting_function(self, i):
        ## Returns the ith Gaussian averaging function

        return self.vector_of_weighting_functions[i]
    
    def length_of_property_vector(self):
        ## Returns the length of the property vector

        return len(self.vector_of_weighting_functions)

class PriorClass:


    def __init__(self, truncation_degree):
        self.truncation_degree = truncation_degree
        self.prior_zeta = pysh.SHGrid.from_zeros(lmax = truncation_degree, grid = 'GLQ')
        self.prior_covariance_Q = np.zeros(truncation_degree)

    def set_prior_zeta(self, zeta_glq):
        self.prior_zeta = zeta_glq

    def set_prior_covariance_Q(self, vector):
        self.prior_covariance_Q = vector

    def add_to_prior_covariance_Q(self, vector):
        self.prior_covariance_Q += vector

    def Q_covariance_operator(self, fun):
        return RF.apply_covariance(self.prior_covariance_Q, fun)


class InferenceClass(GraceSolver, PropertyClassGaussian, PriorClass):


    def __init__(self, truncation_degree, observation_degree, length_of_property_vector):
        GraceSolver.__init__(self, truncation_degree, observation_degree)
        PropertyClassGaussian.__init__(self, truncation_degree, length_of_property_vector)
        PriorClass.__init__(self, truncation_degree)
        self.joint_covariance_matrix = np.zeros((2,2))

    def top_left_operator(self, data_vector):
        ## The operator AQA* + R

        return self.forward_operator(self.Q_covariance_operator(self.adjoint_operator(data_vector)[0])) + self.measurement_error_covariance_matrix @ data_vector
    
# What have I done:
# - Created forward and adjoint operators in GraceSolver (coiuld be extended to take vectors directly)
# - Ability to set and edit data covariance matrix
# - Made the abstract PropertyClass which I have successfully tested
# - Initialise gaussian weighting thing using lists of lat lon and width. New abstract method which returns the lenght of property vector


# What I need to do:
# - (as above) should I adapt forward_operator and adjoint_operator to take vectors?
# - adjoint_operator spits out 5 things - where should i put the [0] index to get SL?
# - Generate rotationally invariant random fields, try masking it with ice function, and then try multiplying it with some mean field (ubar)
# - Then will need ways to sample from it and act the covariance operator on things








    


                                                    

