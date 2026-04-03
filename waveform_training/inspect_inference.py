import argparse
from utils.path_utils import create_and_set_outdir
from utils.samples_processing import (save_mean_of_last_train_samples,
                                      save_plot_final_r_hat,
                                      save_plot_corner_with_reference_no_ticks,
                                      save_plot_log_likelihood_along_train_chains)
from utils.constants_jax import lambda_sigma_IMRPhenomD_33
from utils.waveform_dictionary import (get_desired_snr_squared_of_each_waveform_in_data,
                                       make_waveform_dictionary,
                                       convert_dictionary_to_jax_array)
from utils.likelihood import get_phase_and_time_marginalized_log_likelihood_FFT
from jax import jit
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect inference run and save the mean of the last 100 training samples."
        )
    )
    parser.add_argument(
        "path_to_run",
        type=str,
        help="Path to the run directory.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    path_to_run = args.path_to_run

    outdir = create_and_set_outdir("inspect_inference")
    save_mean_of_last_train_samples(path_to_run, outdir, n_last_samples=100)
    save_plot_final_r_hat(path_to_run, outdir)
    selected_indices = [0,1,2,3,4,5]
    save_plot_corner_with_reference_no_ticks(path_to_run, outdir, selected_indices)

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
    NRHybSur3dq8_data_dictionary = make_waveform_dictionary(
        data_dir,
        snr_squared_of_each_waveform_in_data,
    )
    NRHybSur3dq8_data_array = convert_dictionary_to_jax_array(
        NRHybSur3dq8_data_dictionary
    )
    parameters = lambda_sigma_IMRPhenomD_33
    data = NRHybSur3dq8_data_array
    get_phase_and_time_marginalized_log_likelihood_FFT_jit = jit(get_phase_and_time_marginalized_log_likelihood_FFT)
    get_phase_and_time_marginalized_log_likelihood_FFT_jit(parameters, data).block_until_ready
    save_plot_log_likelihood_along_train_chains(path_to_run, outdir, get_phase_and_time_marginalized_log_likelihood_FFT_jit, data)

if __name__ == "__main__":
    main()