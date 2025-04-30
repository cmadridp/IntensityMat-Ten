import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli

class gaussian_mixture():
    def __init__(self, dim, mean, cov):
        self.dim = dim
        self.mean_1 = [mean[0] for __ in range(dim)]
        self.mean_2 = [mean[1] for __ in range(dim)]
        self.cov_1 = np.diag(np.ones(dim) * cov[0])
        self.cov_2 = np.diag(np.ones(dim) * cov[1])
        self.prob = 0.5

    def density_value(self, x_input):
        return self.prob * multivariate_normal.pdf(x_input, self.mean_1, self.cov_1) + (
                    1 - self.prob) * multivariate_normal.pdf(x_input, self.mean_2, self.cov_2)


    def generate(self, N):
        rr = np.tensordot([bernoulli.rvs(self.prob, size=N)], [np.ones(self.dim)], axes=[[0],[0]])
        #data1= np.multiply(rr,np.random.uniform(0,1, N*self.dim).reshape((N, self.dim)))
        data1 = np.multiply( rr,np.random.multivariate_normal(self.mean_1,self.cov_1,N ) )

        data2 = np.multiply( -1*rr+1,np.random.multivariate_normal(self.mean_2,self.cov_2,N ))
        return data1 + data2


class SinusoidalDensity:
    def __init__(self, amplitude=1.0, frequency=2 * np.pi, phase=0.0):
        """
        Parameters:
         - amplitude: maximum value of the sinusoidal density.
         - frequency: frequency of the sinusoid.
         - phase: phase shift.
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def density_value(self, x_input):
        """
        For an array of points x_input of shape (n, d), compute the sinusoidal density for each point.
        f(x) = amplitude * ∏_{d=1}^d [0.5 * (1 + sin(frequency * x[d] + phase))]

        Parameters:
        x_input (np.array): An array of shape (n, d) where each row is a point in R^d.

        Returns:
        np.array: A 1D array of length n containing the density values at each point.
        """
        # Ensure x_input is a 2D array.
        x = np.atleast_2d(x_input)
        # Compute the sinusoidal component for each coordinate.
        components = 0.5 * (1 + np.sin(self.frequency * x + self.phase))
        # Compute the product over dimensions for each point.
        return self.amplitude * np.prod(components, axis=1)


class poisson_pp():
    def __init__(self, density, dim, scale=1.0, lambda_max=None, grid_size=100, region_lower=None, region_upper=None):
        """
        Parameters:
         - density: a density function.
         - dim: the dimensionality of the density function.
         - scale: a positive constant such that the intensity is lambda(x) = scale * f(x).
         - lambda_max: Optional fixed upper bound for the intensity. If None, the code estimates it.
         - grid_size: used for grid search when estimating lambda_max (if needed).
         - region_lower: lower bounds for candidate region (array-like of length dim). If None, defaults to [-1, ..., -1].
         - region_upper: upper bounds for candidate region (array-like of length dim). If None, defaults to [1, ..., 1].
        """
        self.density = density
        self.scale = scale
        self.dim = dim

        # Set candidate sampling region
        if region_lower is None:
            self.region_lower = np.zeros(dim, dtype=float)
        else:
            self.region_lower = np.array(region_lower)
        if region_upper is None:
            self.region_upper = np.ones(dim, dtype=float)
        else:
            self.region_upper = np.array(region_upper)

        # Estimate lambda_max by sampling candidate points from the specified region.
        if lambda_max is None:
            if self.dim == 1:
                pts = np.linspace(self.region_lower[0], self.region_upper[0], grid_size).reshape(-1, 1)
            else:
                total_points = grid_size ** self.dim
                if total_points > 10000:
                    pts = np.random.uniform(low=self.region_lower, high=self.region_upper, size=(10000, self.dim))
                else:
                    grids = [np.linspace(self.region_lower[d], self.region_upper[d], grid_size) for d in
                                 range(self.dim)]
                    mesh = np.meshgrid(*grids)
                    pts = np.vstack([m.flatten() for m in mesh]).T
            vals = [self.density.density_value(pt) for pt in pts]
            lambda_max_est = max(vals)
            self.lambda_max = self.scale * lambda_max_est
        else:
            self.lambda_max = lambda_max

    def generate(self, N_ppp):
        """
        Generate N_ppp number of i.i.d. Poisson point processes on the region using vectorized thinning.
        Returns:
         - point_processes: a list of NumPy arrays, each of shape (n_points, dim) containing the accepted points.
        """
        point_processes = []
        for i in range(N_ppp):
            # Generate candidate points from a homogeneous Poisson process with mean lambda_max.
            N_candidates = int(np.random.poisson(self.lambda_max))
            if N_candidates <= 0:
                point_processes.append(np.empty((0, self.dim)))
                continue
            # Generate candidates uniformly from the specified region.
            candidates = np.random.random(size=(N_candidates, self.dim)) * (self.region_upper - self.region_lower) + self.region_lower
            # Compute intensities for all candidates at once using the vectorized density function.
            intensities = self.scale * np.array([self.density.density_value(x) for x in candidates]).flatten()
            # Generate uniform random numbers for all candidates.
            U = np.random.uniform(0, 1, N_candidates)
            # Accept candidates where U < intensity / lambda_max.
            accepted = candidates[U < intensities / self.lambda_max]
            point_processes.append(accepted)
        return point_processes


class domain:
    def __init__(self, X_train, factor = 0.0001):
        self.X_train = X_train
        self.dim = X_train.shape[1]
        Y = X_train.transpose()
        self.upper = []
        self.lower = []
        for dd in range(self.dim):
            self.upper.append(np.quantile(Y[dd], 1 - factor))
            self.lower.append(np.quantile(Y[dd], factor))
        self.upper = np.array(self.upper) + 1e-07
        self.lower = np.array(self.lower) - 1e-07
        # print(self.upper)
        # print(self.lower)
        self.difference = self.upper - self.lower
        self.density_factor = np.prod(self.difference)

    def transform_to_0_1(self, x_input):
        # slope=1/(upper-lower)
        # intercept=-1*lower*slope
        return (x_input - self.lower) / self.difference


    def transform_from_0_1(self, data_0_1):
        return data_0_1 * self.difference + self.lower

    def transform_density_val(self, val):
        return val/self.density_factor



# # Example usage:
# # Define parameters for the Gaussian mixture.
# dim = 5
# mean = [0, 0]
# cov = [0.1, 0.1]
# # Create a Gaussian mixture instance.
# gmm = gaussian_mixture(dim, mean, cov)
# # Create a Poisson point process instance with intensity proportional to the Gaussian mixture density.
# # Here, scale controls the overall expected number of points.
# ppp_instance = poisson_pp(gmm, dim, scale=5, region_lower=[-0.5]*dim, region_upper=[0.5]*dim)
# # Generate 100 independent point processes.
# processes = ppp_instance.generate(5000)
# # Concatenate all point processes into a single point process.
# X_train = np.concatenate(processes, axis=0)
# print('X_train shape:', X_train.shape)



# dim = 5
# amplitude=2.0
# # Create a Gaussian mixture instance.
# density = SinusoidalDensity(amplitude)
# ppp_instance = poisson_pp(density, dim, scale=10, region_lower=[-0.5]*dim, region_upper=[0.5]*dim)
# # Generate 100 independent point processes.
# processes = ppp_instance.generate(10)
# # Concatenate all point processes into a single point process.
# X_train = np.concatenate(processes, axis=0)
# print(X_train)
# print('X_train shape:', X_train.shape)
#
# data11 = domain(X_train)
# data_0_1 = data11.transform_to_0_1(X_train)
# print(data_0_1)
# print(data11.transform_from_0_1(data_0_1))