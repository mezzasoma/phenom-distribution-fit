from os import listdir
from os.path import join
import numpy as np
import jax.numpy as jnp

def get_desired_snr_squared_of_each_waveform_in_data(n_dim, desired_avg_mismatch, desired_std_dev_mismatch, number_of_waveforms_in_data):
    """
    Return a conservative per-waveform signal-to-noise ratio squared (SNR^2) required to
    meet the desired mismatch statistics.

    Two SNR^2 values are computed:
    - one from the desired mean mismatch
    - one from the desired standard deviation of the mismatch

    The larger of the two is returned, so that both constraints are satisfied.
    """
    snr_squared_avg_mismatch = (n_dim - 1) / (2.0 * desired_avg_mismatch * number_of_waveforms_in_data)
    snr_squared_std_dev_mismatch = (n_dim - 1) / (np.sqrt(2) * desired_std_dev_mismatch * number_of_waveforms_in_data)
    snr_squared_per_waveform = max(snr_squared_avg_mismatch, snr_squared_std_dev_mismatch)
    snr_per_waveform = np.sqrt(snr_squared_per_waveform)

    print(f"SNR squared of each waveform in data, from average mismatch, is: {snr_squared_avg_mismatch:.2f}")
    print(f"SNR squared of each waveform in data, from standard deviation mismatch, is: {snr_squared_std_dev_mismatch:.2f}")
    print(f"Conservative choice for SNR squared of each waveform in data is: {snr_squared_per_waveform:.2f}")
    print(f"Choosing per-waveform SNR = sqrt({snr_squared_per_waveform:.2f}) = {snr_per_waveform:.2f}" )

    return snr_squared_per_waveform

def make_waveform_dictionary(dir, snr_squared):
    """
    Create waveform dictionary with user-defined SNR squared for each waveform (.npz) found
    in the directory `dir`.
    Each waveform comes with its own (assumed constant) PSD S_n, whose value is 
    chosen to produce the desired SNR squared.
    The dictionary entries are jax.numpy arrays.
    """
    data_dictionary = {}
    for filename in listdir(dir):
        full_path = join(dir, filename)
        if full_path.endswith('.npz'):
            data = np.load(full_path)
            xi = jnp.array(data['xi'])
            freqs = jnp.array(data['freqs_ftt'])
            fft = data['aligned_data_fft_tukey']
            amplitude_squared = np.abs(fft)**2
            df = freqs[1] - freqs[0]
            snr_squared_before_scaling = 4 * np.trapz(amplitude_squared,data['freqs_ftt'])
            S_n = jnp.array(snr_squared_before_scaling/snr_squared)
            data_subdictionary = {'xi':xi, 'freqs' : freqs, 'fft': jnp.array(fft), 'S_n': S_n}
            data_dictionary[filename[:-4]] = data_subdictionary
            print('Loaded data (q,chi1,chi2) = ({:.3f},{:.1f},{:.1f}), [fmin,fmax] = [{:.3f},{:.3f}] Hz, df = {:.3e}, # samples = {}'.format(*xi,freqs[0],freqs[-1],df,len(freqs)))
    data_dictionary = dict(sorted(data_dictionary.items(), key=lambda item: item[1].get('xi')[0]))
    return data_dictionary

def convert_dictionary_to_jax_array(data_dictionary):
    """
    Convert waveform data dictionary into a homogeneus jax array. 
    The structure of the array is:
    [ [freqs_and_xi_and_S_n, re_fft + 4-zeros pad, im_fft + 4-zeros pad ], [same for each waveform], ... ]
    """
    data_list = []
    for data in data_dictionary.values():
        xi = data['xi']
        freqs = data['freqs']
        fft = data['fft']
        S_n = data['S_n']
        freqs_and_xi_and_S_n = jnp.zeros(len(freqs)+4, dtype=jnp.float64)
        re_fft = jnp.zeros(len(freqs)+4, dtype=jnp.float64)
        im_fft = jnp.zeros(len(freqs)+4, dtype=jnp.float64)
        freqs_and_xi_and_S_n = freqs_and_xi_and_S_n.at[0:len(freqs)].set(freqs)
        freqs_and_xi_and_S_n = freqs_and_xi_and_S_n.at[-4:-1].set(xi)
        freqs_and_xi_and_S_n = freqs_and_xi_and_S_n.at[-1:].set(S_n)
        re_fft = re_fft.at[0:len(freqs)].set(jnp.real(fft))
        im_fft = im_fft.at[0:len(freqs)].set(jnp.imag(fft))
        data_list.append([freqs_and_xi_and_S_n, re_fft, im_fft])
    data_array = jnp.array(data_list)
    return data_array