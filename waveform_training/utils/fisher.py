from jax import hessian
from utils.likelihood import get_phase_and_time_marginalized_log_likelihood_FFT
import numpy as np

def get_observed_fisher_matrix(parameters, data_array):
    """
    Calculate the observed Fisher information matrix using automatic differentiation.
    """
    def logl(parameters):
        ll = get_phase_and_time_marginalized_log_likelihood_FFT(parameters, data_array)
        return ll
    hessian_logl = hessian(logl)
    return -1.0 * hessian_logl(parameters)

def is_eigenvector(A, eigen_vectors, eigen_values, tol=1e-6):
    """
    Check if the given `eigen_vectors` are eigenvectors of the matrix `A`
    with respective eigenvalues `eigen_values`.
    """
    for i in range(eigen_vectors.shape[1]):
        v = eigen_vectors[:, i]
        lambda_i = eigen_values[i]
        if not np.allclose(np.dot(A, v), lambda_i * v, atol=tol):
            return False
    return True