import numpy as np
import math


class Indexing:
    def generate(self, max_level):
        """Generate a dictionary mapping indices to (level, k) pairs.
        For index in 1 to 2**max_level, level = floor(log2(index)) and k = index - 2**(level).
        """
        seen = {}
        # Iterate over indices from 1 to 2**max_level (inclusive)
        for index in range(1, 2**max_level + 1):
            level = int(math.floor(math.log2(index)))
            k = index - 2**level
            seen[index] = (level, k)
        return seen


class Wavelet:
    def __init__(self):
        # Precompute the indexing dictionary up to level 6
        self.seen = Indexing().generate(6)
        # Precompute powers: 2^(n/2) for n in 0,...,29
        self.power = np.power(2, np.arange(30) / 2.0)

    def generator(self, x):
        """Simple generator function: returns 1 if x in [0,0.5), -1 if in [0.5,1), and 0 otherwise."""
        if x < 0 or x >= 1:
            return 0
        return 1 if x < 0.5 else -1

    def compute(self, l_index, x):
        """Compute the wavelet basis function for a given index and input x."""
        if l_index == 0:
            return 1 if 0 <= x < 1 else 0
        level, k = self.seen[l_index]
        # Compute argument for generator and scale it with the precomputed power
        return self.power[level] * self.generator((self.power[level]**2) * x - k)

    def compute_single_x_all_basis(self, number_of_basis, x_input):
        """Compute all basis function values up to a specified number for a given x_input."""
        return np.array([self.compute(i, x_input) for i in range(number_of_basis)])


class WaveletNewBasis:
    def __init__(self, M):
        """Initialize with M basis functions."""
        self.M = M
        self.wavelet = Wavelet()

    def compute_single_x_all_new_basis(self, U, x_input):
        """Compute the new basis by taking the inner product of the standard basis vector and U.
        U should be a vector of length M.
        """
        vec_temp = self.wavelet.compute_single_x_all_basis(self.M, x_input)
        return np.inner(vec_temp, U)


# Uncomment the following lines to test the functions:
# wavelet_instance = Wavelet()
# print(wavelet_instance.compute_single_x_all_basis(10, 0.4))

# wavelet_new = WaveletNewBasis(3)
# print(wavelet_new.compute_single_x_all_new_basis([1, 0, 0], 0.4))
