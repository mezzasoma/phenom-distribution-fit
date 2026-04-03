print('Started execution of Python script.')

import argparse
import os
# Limit JAX/XLA GPU preallocation to 94%. Reduced from 99% to avoid XLA memory error.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=str(.94)
import pickle
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, lax

# ------------------------------------------------------------------
# Laptop settings:
# Force JAX to use the CPU, since a compatible GPU may be unavailable
# or may not have enough memory for this inference.
# jax.config.update("jax_platforms", "cpu")
# ------------------------------------------------------------------

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

from utils.constants_jax import lambda_sigma_IMRPhenomD_33
from utils.h_plus import get_h_plus_ripple_lambda_sigma
from utils.waveform_dictionary import (get_desired_snr_squared_of_each_waveform_in_data,
                                       make_waveform_dictionary,
                                       convert_dictionary_to_jax_array)
from utils.prior import get_domain_from_domain_width
from utils.likelihood import get_phase_and_time_marginalized_log_likelihood_FFT_no_vmap

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Perform inference with FlowMC."
        )
    )
    parser.add_argument(
        "injection_attempt",
        type=int,
        help="Injection number: 0, 1, 2, ...",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="Absolute path to the inference output directory.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    output_directory = args.output_directory
    injection_attempt = args.injection_attempt

    data_dir = "outdir/NRHybSur3dq8_training_set_frequency_domain"
    number_of_waveforms_in_data = len([f for f in os.listdir(data_dir) if f.endswith('.npz')])

    print(f'{number_of_waveforms_in_data} waveforms in the training set')

    desired_average_mismatch = 1e-4
    desired_standard_deviation_mismatch = 1e-4

    n_dim = len(lambda_sigma_IMRPhenomD_33)
    snr_squared_of_each_waveform_in_data = get_desired_snr_squared_of_each_waveform_in_data(n_dim, desired_average_mismatch, desired_standard_deviation_mismatch, number_of_waveforms_in_data)

    get_h_plus_ripple_lambda_sigma_jit = jit(get_h_plus_ripple_lambda_sigma)
    NRHybSur3dq8_data_dictionary = make_waveform_dictionary(data_dir, snr_squared_of_each_waveform_in_data)
    NRHybSur3dq8_data_array = convert_dictionary_to_jax_array(NRHybSur3dq8_data_dictionary)

    def custom_transform(params, linear_transformation_matrix):
        return jnp.dot(params, linear_transformation_matrix)

    def inverse_custom_transform(transformed_params, linear_transformation_matrix):
        return jnp.dot(transformed_params, linear_transformation_matrix.T)

    parameter_transformation_dir = 'outdir/parameter_transformation'
    with open(parameter_transformation_dir + '/injection_' + str(injection_attempt) + '_parameter_transformation.pkl', 'rb') as f:
        linear_transformation_matrix = pickle.load(f)
        print(f'Loaded custom transformation.')

    with open(parameter_transformation_dir + '/injection_' + str(injection_attempt) + '_point.pkl', 'rb') as f:
        loaded_injection_point = pickle.load(f)
        print(f'Loaded injection point.')

    if injection_attempt == 0:
        injection_point_for_NRSur_data = lambda_sigma_IMRPhenomD_33
    else:
        injection_point_for_NRSur_data = loaded_injection_point

    by_hand_mean = custom_transform(injection_point_for_NRSur_data, linear_transformation_matrix)
    domain_width = 3.68e+7 * np.array([   1. ,   18.4,    1.6,   30. ,  113.6,    2.2,   23.1,  171.8,
    10.9,   10.3,   59.4,   19.6,   28.3,   21.7,  173.9,  173.9,
    32.6,  173.9,  195.7,  173.9,  173.9,  173.9,  161.1,  161.1,
    161.1,  161.1,  161.1,  161.1,  161.1,  161.1,  161.1,  161.1,
    161.1])
    domain = jnp.array(get_domain_from_domain_width(np.array(injection_point_for_NRSur_data), domain_width))

    def flat_prior(lambda_sigma):
        '''
        Jax-friendly log flat prior.
        '''
        is_in_domain = ((domain[:,0]<=(lambda_sigma)) & (lambda_sigma<=domain[:,1])).all()
        prior_value = lax.cond(is_in_domain, lambda: 0.0, lambda: -jnp.inf)
        return prior_value

    def log_probability(transformed_lambda_sigma, data):
        '''
        Jax-friendly log probability: log(prior) + log(likelihood)
        '''
        lambda_sigma = inverse_custom_transform(transformed_lambda_sigma,linear_transformation_matrix)
        lp = flat_prior(lambda_sigma)
        ll = get_phase_and_time_marginalized_log_likelihood_FFT_no_vmap(lambda_sigma, data)
        log_prob_value = lax.cond(jnp.isinf(lp), lambda: -jnp.inf, lambda: lp + ll)
        return log_prob_value

    sigma_values = jnp.array([
        24.03104749, 192.97617831, 48.54935367, 602.23262256, 2378.5119922, 
        56.34204978, 646.87945548, 2518.19124078, 34.96144059, 403.1120487, 
        1256.81324082, 93.21788141, 767.54582729, 198.53009895, 2832.45681808, 
        10604.44014435, 149.95604306, 2331.93813933, 10323.65610185, 105.28417374, 
        1287.30870028, 4471.36036942, 1878.55886219, 1878.55886219, 1878.55886219, 
        1878.55886219, 1878.55886219, 1878.55886219, 1878.55886219, 1878.55886219, 
        1878.55886219, 1878.55886219, 1878.55886219
    ])

    matrix_sigma_squared = jnp.diag(sigma_values ** 2)
    by_hand_sigmas = jnp.sqrt(jnp.diag(jnp.dot(linear_transformation_matrix,jnp.dot(matrix_sigma_squared,linear_transformation_matrix.T))))
    tentative_sigmas = jnp.diag(by_hand_sigmas)

    data = NRHybSur3dq8_data_array
    n_dim = 33
    n_chains = 800
    n_loop_training = 100
    if injection_attempt in [0, 1]:
        n_loop_production = 1
    else:
        n_loop_production = 50
    n_local_steps = 100
    n_global_steps = 40
    num_epochs = 20

    n_layer = 10 
    n_hidden = 128

    learning_rate = 0.0001
    momentum = 0.9
    batch_size = 50_000
    max_samples = 50_000
    use_global=True
    train_thinning=1

    step_size = 0.3e-4
    mass_matrix = tentative_sigmas
    sampler_params = {'step_size': mass_matrix*step_size}

    rng_key_set = initialize_rng_keys(n_chains, seed=99)

    mean = jnp.tile(by_hand_mean,(n_chains,1))
    cov = jnp.tile(jnp.diag(jnp.array((1e-10*jnp.abs(by_hand_mean)**2))),(n_chains,1,1))
    initial_position = jax.random.multivariate_normal(rng_key_set[0], mean=mean, cov=cov, dtype=jnp.float64)

    model = RQSpline(n_dim, n_layer, [n_hidden, n_hidden], 8)
    MALA_Sampler = MALA(log_probability, True, sampler_params, use_autotune=False)
    nf_sampler = Sampler(
        n_dim,
        rng_key_set,
        data,
        MALA_Sampler,
        model,
        n_loop_training=n_loop_training,
        n_loop_production=n_loop_production,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_chains=n_chains,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        batch_size=batch_size,
        use_global=use_global,
        train_thinning=train_thinning
    )

    configuration_content = f"""
n_dim = {n_dim}
n_chains = {n_chains}
n_loop_training = {n_loop_training}
n_loop_production = {n_loop_production}
n_local_steps = {n_local_steps}
n_global_steps = {n_global_steps}
num_epochs = {num_epochs}
n_layer = {n_layer}
n_hidden = {n_hidden}
learning_rate = {learning_rate}
momentum = {momentum}
batch_size = {batch_size}
max_samples = {max_samples}
step_size = {step_size}
use_global = {use_global}
train_thinning = {train_thinning}

injection_attempt = {injection_attempt}
data_waveform_set = NRHybSur3dq8
snr_squared_of_each_waveform_in_data = {snr_squared_of_each_waveform_in_data}
snr_of_each_waveform_in_data = {np.sqrt(snr_squared_of_each_waveform_in_data)}
desired_average_mismatch = {desired_average_mismatch}
desired_standard_deviation_mismatch = {desired_standard_deviation_mismatch}
    """

    print(configuration_content)

    print('Sampling starts.')
    nf_sampler.sample(initial_position, data)
    print('Sampling ended.')

    print(f"About to write the FlowMC sampler configuration and samples under {output_directory}.")
    os.makedirs(output_directory, exist_ok=True)

    with open(os.path.join(output_directory, "configuration.txt"), "w") as file:
        file.write(configuration_content.strip())

    summary = nf_sampler.get_sampler_state()

    with open(os.path.join(output_directory, "summary.pkl"), "wb") as file:
        pickle.dump(summary, file)

    with open(os.path.join(output_directory, "params.pkl"), "wb") as file:
        pickle.dump(nf_sampler.state.params, file)

    with open(os.path.join(output_directory, "variables.pkl"), "wb") as file:
        pickle.dump(nf_sampler.variables, file)

    out_train = nf_sampler.get_sampler_state(training=True)
    with open(os.path.join(output_directory, "train.pkl"), "wb") as file:
        pickle.dump(out_train, file)

    with open(
        os.path.join(output_directory, "parameter_transformation.pkl"),
        "wb",
    ) as file:
        pickle.dump(linear_transformation_matrix, file)

    with open(
        os.path.join(
            output_directory,
            "evaluation_point_of_parameter_transformation.pkl",
        ),
        "wb",
    ) as file:
        pickle.dump(loaded_injection_point, file)

    with open(os.path.join(output_directory, "injection_point.pkl"), "wb") as file:
        pickle.dump(injection_point_for_NRSur_data, file)

    print('Writing samples ended.')
    quit()

if __name__ == "__main__":
    main()