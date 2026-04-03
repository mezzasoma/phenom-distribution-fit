import argparse
import os
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
# ------------------------------------------------------------------
# Laptop settings:
# Force JAX to use the CPU, since a compatible GPU may be unavailable
# or may not have enough memory.
jax.config.update("jax_platforms", "cpu")
# ------------------------------------------------------------------
import pickle
import warnings

from utils.path_utils import create_and_set_outdir
from utils.constants_jax import lambda_sigma_IMRPhenomD_33
from utils.waveform_dictionary import (get_desired_snr_squared_of_each_waveform_in_data,
                                       make_waveform_dictionary,
                                       convert_dictionary_to_jax_array)

from utils.h_plus import get_h_plus_ripple_lambda_sigma
from utils.fisher import (get_observed_fisher_matrix,
                          is_eigenvector)

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute and save the parameter transformation for a given injection. "
            "For injection 0, do not provide a previous run label. "
            "For injection >= 1, provide the previous run label."
        )
    )

    parser.add_argument(
        "injection_attempt",
        type=int,
        help="Injection number: 0, 1, 2, ...",
    )

    parser.add_argument(
        "previous_run_label",
        nargs="?",
        default=None,
        type=str,
        help=(
            "Run label of the previous run, e.g. flowmc-20240722113944. "
            "Required only if injection_attempt > 0."
        ),
    )

    args = parser.parse_args()

    if args.injection_attempt < 0:
        parser.error("injection_attempt must be 0, 1, 2, ...")

    if args.injection_attempt == 0 and args.previous_run_label is not None:
        parser.error(
            "Do not provide previous_run_label when injection_attempt is 0."
        )

    if args.injection_attempt > 0 and args.previous_run_label is None:
        parser.error(
            "previous_run_label is required when injection_attempt > 0."
        )

    return args


def get_injection_point(injection_attempt, previous_run_label):
    if injection_attempt == 0:
        stable_22_dim_injection = jnp.array(
            [
                -9.65664758e+03, -4.60098674e+04, -6.95209760e+03,
                -2.58057367e+05,  6.61823414e+05,  2.00675596e+03,
                -4.19271547e+05,  1.58679105e+06, -8.08679957e+03,
                -1.06534577e+05,  6.55541261e+05,  1.98384348e+04,
                 2.42327675e+05,  1.59788540e+04,  1.13091563e+06,
                -2.87741191e+06,  3.71306547e+03,  1.74748678e+06,
                -6.93309626e+06,  4.58693315e+04,  4.15452148e+05,
                -2.91769735e+06,
            ],
            dtype=jnp.float64,
        )
        new_injection = jnp.concatenate(
            (stable_22_dim_injection, lambda_sigma_IMRPhenomD_33[-11:]),
            dtype=jnp.float64,
        )
        return new_injection

    path_to_new_injection = os.path.join(
        "outdir",
        "inspect_inference",
        f"{previous_run_label}_train_mean.pkl",
    )
    with open(path_to_new_injection, "rb") as f:
        new_injection = pickle.load(f)

    return jnp.array(new_injection, dtype=jnp.float64)


def save_transformation_and_point(
    outdir,
    injection_attempt,
    parameter_transformation,
    new_injection,
):
    transformation_path = os.path.join(
        outdir,
        f"injection_{injection_attempt}_parameter_transformation.pkl",
    )
    point_path = os.path.join(
        outdir,
        f"injection_{injection_attempt}_point.pkl",
    )

    with open(transformation_path, "wb") as f:
        pickle.dump(parameter_transformation, f)

    with open(point_path, "wb") as f:
        pickle.dump(new_injection, f)


def main():
    args = parse_args()

    outdir = create_and_set_outdir("parameter_transformation")

    data_dir = "outdir/NRHybSur3dq8_training_set_frequency_domain"
    number_of_waveforms_in_data = len(
        [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    )
    desired_average_mismatch = 1e-4
    desired_standard_deviation_mismatch = 1e-4
    n_dim = len(lambda_sigma_IMRPhenomD_33)
    snr_squared_of_each_waveform_in_data = get_desired_snr_squared_of_each_waveform_in_data(
        n_dim,
        desired_average_mismatch,
        desired_standard_deviation_mismatch,
        number_of_waveforms_in_data,
    )
    get_h_plus_ripple_lambda_sigma_jit = jit(get_h_plus_ripple_lambda_sigma)
    get_observed_fisher_matrix_jit = jit(get_observed_fisher_matrix)
    NRHybSur3dq8_data_dictionary = make_waveform_dictionary(
        data_dir,
        snr_squared_of_each_waveform_in_data,
    )
    NRHybSur3dq8_data_array = convert_dictionary_to_jax_array(
        NRHybSur3dq8_data_dictionary
    )
    parameters = lambda_sigma_IMRPhenomD_33
    data = NRHybSur3dq8_data_array
    get_observed_fisher_matrix_jit(parameters, data).block_until_ready()

    new_injection = get_injection_point(
        args.injection_attempt,
        args.previous_run_label,
    )
    total_sum_observed_fisher_matrix = get_observed_fisher_matrix_jit(
        new_injection,
        data,
    )

    eig_values, eig_vectors = np.linalg.eigh(total_sum_observed_fisher_matrix)
    tolerances = [10.0**(-k) for k in range(9, 17)]
    for tol in tolerances:
        if is_eigenvector(
            total_sum_observed_fisher_matrix,
            eig_vectors,
            eig_values,
            tol=tol,
        ):
            print(f"Eigenvector check passed at tolerance {tol:.1e}")
        else:
            warnings.warn(
                f"Eigenvector check failed at tolerance {tol:.1e}. "
                "A lower tolerance corresponds to a better diagonalization of the Fisher matrix. "
                "This warning can be ignored, but if the tolerance is greater than machine precision "
                "the resulting parameter transformation may not be able to decorrelate "
                "the sampled parameters."
            )
            break

    parameter_transformation = jnp.array(eig_vectors.astype(jnp.float64))
    save_transformation_and_point(
        outdir,
        args.injection_attempt,
        parameter_transformation,
        new_injection,
    )

if __name__ == "__main__":
    main()