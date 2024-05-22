import numpy as np
import pyshtools as pysh
import RFmod as RF
import SLmod as SL
import xarray as xr
from abc import ABC, abstractmethod
from multiprocessing import Pool
from functools import partial
from scipy import optimize

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
        current_sea_level, current_ice_thickness = SL.get_sl_ice_data(truncation_degree)
        self.ocean_function = SL.ocean_function(current_sea_level, current_ice_thickness)

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
        self.measurement_error_covariance_matrix = np.eye(size)

    def convert_glq_to_vector_of_sh_coeffs(self, glq_grid):
        ## Converts a GLQ grid to a vector of SH coefficients, starting at l=2
        ## For reference: Clm = phi_coeffs[0,l,m>=0] and Slm = phi_coeffs[1,l,m>=1]/
        coeffs = glq_grid.expand(normalization = 'ortho').to_array()

        coeffs_vec = np.zeros(((self.observation_degree+1)**2)-4)       
        for l in range(2,self.observation_degree+1):
            coeffs_vec[((l)**2)-4:((l+1)**2)-4] = np.concatenate((coeffs[1,l,1:l+1][::-1],coeffs[0,l,0:l+1]))
        
        return coeffs_vec        

    def observation_operator(self, sl, u, phi, om, psi=0):
        ## Converts a solution of the fingerprint problem to a vector of SH coefficients for phi
        ## For reference: Clm = phi_coeffs[0,l,m>=0] and Slm = phi_coeffs[1,l,m>=1]/
        ## [EDIT REQUEST]: SHOULD USE ABOVE FUNCTION!!
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

        return self.solve_adjoint_fingerprint(*self.observation_operator_adjoint(phi_coeffs_vec))[0]
    
    def data_inner_product(self, phi1, phi2):

        return np.inner(phi1,phi2)
    
    def set_measurement_error_covariance_matrix(self, matrix):

        self.measurement_error_covariance_matrix = matrix
    
    def scale_measurement_error_covariance_matrix(self, factor):

        self.measurement_error_covariance_matrix = factor*self.measurement_error_covariance_matrix

    def add_to_measurement_error_covariance_matrix(self, matrix):

        self.measurement_error_covariance_matrix += matrix

    def apply_measurement_error_covariance(self, vector):

        return self.measurement_error_covariance_matrix @ vector

    def generate_sample_of_measurement_error(self, num_samples):
        ## Generates num_samples random samples from the measurement error covariance matrix

        size = self.size_of_data_vector(self.observation_degree)
        return np.random.multivariate_normal(np.zeros(size), self.measurement_error_covariance_matrix, num_samples)

        
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

        ## Get the ice prior
        arr = SL.get_sl_ice_data(truncation_degree)[1].to_array()
        self.prior_ice_load = pysh.SHGrid.from_array(np.where(arr > 0, np.log(4000/(arr+400)), 0), grid='GLQ')

        ## Get the ocean prior
        sl_anomaly_dataset = xr.open_mfdataset('/home/dah94/space/ssh/MON/dt_global_allsat_msla_h_y1993_m*.nc')
        da = SL.format_latlon(sl_anomaly_dataset.sla.mean('time'))
        da = da.fillna(0)        
        self.prior_ocean_load = 10*SL.interpolate_xarray(truncation_degree, da)

        ## Get the total prior load
        self.prior_total_load = self.prior_ice_load + self.prior_ocean_load

        ## Set the covariances
        self.ice_covariance_Q = RF.sobolev_covariance(truncation_degree, s=2, mu=0.2)
        self.ocean_covariance_Q = RF.sobolev_covariance(truncation_degree, s=2, mu=0.2)


    def sample_individual_load(self, mean_field, Q):
        ## For a given mean field and covariance operator Q, samples a random field

        return mean_field*(RF.random_field(Q) + 1)

    def apply_individual_load_covariance(self, mean_field, Q, fun):
        ## For a given mean field, covariance operator Q and function, applies the covariance operator to the function

        return mean_field*RF.apply_covariance(Q, mean_field*fun)
    
    def sample_full_load(self):
        ## Samples a full load field

        return self.sample_individual_load(self.prior_ice_load, self.ice_covariance_Q) + self.sample_individual_load(self.prior_ocean_load, self.ocean_covariance_Q)
    
    def apply_full_load_covariance(self, fun):
        ## Applies the full load covariance operator to a function

        return self.apply_individual_load_covariance(self.prior_ice_load, self.ice_covariance_Q, fun) + self.apply_individual_load_covariance(self.prior_ocean_load, self.ocean_covariance_Q, fun)

    def set_ice_covariance(self, Q):
        ## Sets the ice covariance operator

        self.ice_covariance_Q = Q

    def set_ocean_covariance(self, Q):
        ## Sets the ocean covariance operator

        self.ocean_covariance_Q = Q

class InferenceClass(GraceSolver, PropertyClassGaussian, PriorClass):


    def __init__(self, truncation_degree, observation_degree, gaussian_params=[(100,0,0)]):
        GraceSolver.__init__(self, truncation_degree, observation_degree)
        PropertyClassGaussian.__init__(self, truncation_degree, gaussian_params)
        PriorClass.__init__(self, truncation_degree)

    def generate_synthetic_dataset(self, num_samples):
        ## Generates a synthetic dataset of phi coefficients with associated errors

        synthetic_dataset = np.zeros((num_samples, self.size_of_data_vector(self.observation_degree)))
        synthetic_dataset_errors = synthetic_dataset.copy()

        ## Set R
        self.scale_measurement_error_covariance_matrix(1e-5)

        ## Initialise the full loads
        load_samples = [self.sample_full_load() for _ in range(num_samples)]

        for i in range(num_samples):
            synthetic_dataset[i] = self.forward_operator(load_samples[i])
            synthetic_dataset_errors[i] = self.generate_sample_of_measurement_error(1)
            synthetic_dataset[i] += synthetic_dataset_errors[i]
        
        return synthetic_dataset, synthetic_dataset_errors, load_samples

    def wahr_method(self, phi_coeffs):

        ## Load the love numbers up to truncation_degree, and define the output vector
        _,k,_,_ = SL.love_numbers(self.truncation_degree)
        result = np.zeros(self.length_of_property_vector())

        ## Loop over the weighting functions:
        for i in range(self.length_of_property_vector()):

            ## Express the ith weighting function as a vector of SH coefficients
            w_coeffs = self.convert_glq_to_vector_of_sh_coeffs(self.vector_of_weighting_functions[i])

            ## Loop over l and m, and compute k^-1*phi_lm*w_lm
            for l in range(2,self.observation_degree+1):
                for m in range(-1*l,l+1):
                    vec_index = (l**2)-4+m+l
                    result[i] += (1/k[l]) * phi_coeffs[vec_index] * w_coeffs[vec_index]

            result = result*b**2

        return result

    def top_left_operator(self, data_vector):
        ## The operator AQA* + R

        return self.forward_operator(self.apply_full_load_covariance(self.adjoint_operator(data_vector))) + self.apply_measurement_error_covariance(data_vector)
    
    def top_right_operator(self, property_vector):
        ## The operator AQB*

        return self.forward_operator(self.apply_full_load_covariance(self.adjoint_property_operator(property_vector)))
                                     
    def bottom_left_operator(self, data_vector):
        ## The operator BGA*

        return self.forward_property_operator(self.apply_full_load_covariance(self.adjoint_operator(data_vector)))

    def bottom_right_operator(self, property_vector):
        ## The operator BQB*

        return self.forward_property_operator(self.apply_full_load_covariance(self.adjoint_property_operator(property_vector)))
    
    def compute_top_left_matrix(self):

        data_size =  self.size_of_data_vector(self.observation_degree)
        matrix = np.zeros((data_size,data_size))

        for i in np.arange(data_size):
            basis_vector = np.zeros(data_size)
            basis_vector[i] = 1

            matrix[:,i] = self.top_left_operator(basis_vector)

        return matrix
    
    def compute_top_right_matrix(self):

        data_size = self.size_of_data_vector(self.observation_degree)
        property_size = self.length_of_property_vector()
        matrix = np.zeros((data_size, property_size))

        for i in np.arange(property_size):
            basis_vector = np.zeros(property_size)
            basis_vector[i] = 1

            matrix[:,i] = self.top_right_operator(basis_vector)

        return matrix
    
    def compute_bottom_left_matrix(self):

        data_size = self.size_of_data_vector(self.observation_degree)
        property_size = self.length_of_property_vector()
        matrix = np.zeros((property_size, data_size))

        for i in np.arange(data_size):
            basis_vector = np.zeros(data_size)
            basis_vector[i] = 1

            matrix[:,i] = self.bottom_left_operator(basis_vector)

        return matrix    
    
    def compute_bottom_right_matrix(self):

        property_size = self.length_of_property_vector()
        matrix = np.zeros((property_size,property_size))

        for i in np.arange(property_size):
            basis_vector = np.zeros(property_size)
            basis_vector[i] = 1

            matrix[:,i] = self.bottom_right_operator(basis_vector)

        return matrix
    
    def compute_column(self, i, basis_size, operation):
        basis_vector = np.zeros(basis_size)
        basis_vector[i] = 1

        # Dynamically call the appropriate method based on 'operation'
        operator_method = getattr(self, f"{operation}_operator")
        return operator_method(basis_vector)

    def compute_matrix_parallel(self, operation):

        data_size = self.size_of_data_vector(self.observation_degree)
        property_size = self.length_of_property_vector()
        
        if operation == "top_left":
            matrix_shape = (data_size, data_size)
        elif operation == "top_right":
            matrix_shape = (data_size, property_size)
        elif operation == "bottom_left":
            matrix_shape = (property_size, data_size)
        elif operation == "bottom_right":
            matrix_shape = (property_size, property_size)

        matrix = np.zeros(matrix_shape)

        # Define a partial function to pass additional arguments
        partial_compute_column = partial(self.compute_column, basis_size=matrix_shape[1], operation=operation)

        # Use multiprocessing Pool to parallelize the computation
        with Pool(processes=4) as pool:
            results = pool.map(partial_compute_column, range(matrix_shape[1]))

        # Assign the results to the matrix
        for i, result in enumerate(results):
            matrix[:, i] = result

        return matrix

    def compute_posterior_property_vector(self, data_vector):
        ## Computing w_p

        # Compute AQA* + R
        top_left_matrix = self.compute_top_left_matrix()
        # Invert the matrix
        top_left_matrix_inv = np.linalg.inv(top_left_matrix)

        v_minus_vbar = data_vector - self.forward_operator(self.prior_total_load)

        return self.forward_property_operator(self.prior_total_load) + self.bottom_left_operator(top_left_matrix_inv @ v_minus_vbar)
    
    def compute_posterior_covariance_matrix(self):
        ## Computing s

        # Compute AQA* + R
        top_left_matrix = self.compute_top_left_matrix()
        # Invert the matrix
        top_left_matrix_inv = np.linalg.inv(top_left_matrix)
        # Compute AQB*
        top_right_matrix = self.compute_top_right_matrix()
        # Compute BQA*
        bottom_left_matrix = self.compute_bottom_left_matrix()
        # Compute BQB*
        bottom_right_matrix = self.compute_bottom_right_matrix()

        # Compute the matrix
        return bottom_right_matrix - bottom_left_matrix @ top_left_matrix_inv @ top_right_matrix

    def compute_posterior_load(self, data_vector):
        ## Computing u_p

        # Compute AQA* + R
        top_left_matrix = self.compute_top_left_matrix()
        # Invert the matrix
        top_left_matrix_inv = np.linalg.inv(top_left_matrix)
        
        v_minus_vbar = data_vector - self.forward_operator(self.prior_total_load)
        
        return self.prior_total_load + self.apply_full_load_covariance(self.adjoint_operator(top_left_matrix_inv @ v_minus_vbar))
    
    def apply_posterior_load_covariance_matrix(self, vector):
        ## Computing Q_p

        # Compute AQA* + R
        top_left_matrix = self.compute_top_left_matrix()
        # Invert the matrix
        top_left_matrix_inv = np.linalg.inv(top_left_matrix)

        # Compute the action of Q_p
        return self.apply_full_load_covariance(vector) - self.apply_full_load_covariance(self.adjoint_operator(top_left_matrix_inv @ self.forward_operator(self.apply_full_load_covariance(vector))))
        
    def loss_function(self, guess, data_vec):
        ## Loss fucntion for top left operator AQA* + R

        return np.linalg.norm(self.top_left_operator(guess) - data_vec)
    
    def approximate_top_left_inverse(self, data_vec):
        ## Function which approximates the inverse of AQA* + R

        initial_guess = data_vec
        return optimize.minimize(self.loss_function, initial_guess, args=(data_vec,))




    
# What have I done:
# - Created forward and adjoint operators in GraceSolver (coiuld be extended to take vectors directly)
# - Ability to set and edit data covariance matrix
# - Made the abstract PropertyClass which I have successfully tested
# - Initialise gaussian weighting thing using lists of lat lon and width. New abstract method which returns the length of property vector
# - Work out a way to downsample sea level data to L=64 (either by initialising some grid and interpolating manually, or by pyshtools expansions)
# - Appraise the current framework for doing the priors (I don't think it needs to be particularly interactive)
# - Make some new matrices w_p and s
# - Compute the posterior load


# What I need to do:
# - Try sumI(b^2 phi_lm w_lm) = int(phi*w)dS
# - Start working on computing the block matrices
# - Look at the errors in the Wahr method
# - Decide on length vs size naming 
# - Try out some linear solvers
# - Work out multiprocessing
# - Make some cool plots of w
        







    


                                                    

