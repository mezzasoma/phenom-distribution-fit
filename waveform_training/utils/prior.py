import numpy as np

def get_domain_from_domain_width(lambda_sigma, domain_width):
    """
    Creates a prior domain centered at given `lambda_sigma` value, with a set width in each dimension.
    
    Parameters:
    - lambda_sigma (numpy array): Center values for each dimension.
    - domain_width (numpy array): Width of the domain for each dimension.
    
    Returns:
    - numpy array: Array with the prior domain for each dimension.
    """
    doubled_lambda_sigma = np.repeat(lambda_sigma, 2)
    doubled_lambda_sigma[::2] -= domain_width / 2
    doubled_lambda_sigma[1::2] += domain_width / 2

    return np.split(doubled_lambda_sigma, len(lambda_sigma))