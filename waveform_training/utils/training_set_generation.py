import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({"text.usetex": True})
import os
import lal
import lalsimulation as lalsim
from pycbc.waveform import get_fd_waveform
from tqdm import tqdm
import scipy
from scipy.optimize import fmin
import random
from utils.mismatch import get_minimized_mismatch

def create_uniform_3d_grid(q_min, q_max, chi_min, chi_max, n_points_per_dim):
    """
    Create a uniform grid in the 3D space defined by q, chi_A, and chi_B.

    Parameters:
    q_min (float): Minimum value of q.
    q_max (float): Maximum value of q.
    chi_min (float): Minimum value of chi_A and chi_B.
    chi_max (float): Maximum value of chi_A and chi_B.
    n_points_per_dim (int): Number of points per dimension.

    Returns:
    np.ndarray: Array of shape (n_points_per_dim**3, 3) with the grid points.
    """
    q_values = np.linspace(q_min, q_max, n_points_per_dim)
    chi_values = np.linspace(chi_min, chi_max, n_points_per_dim)

    q_grid, chi_A_grid, chi_B_grid = np.meshgrid(q_values, chi_values, chi_values, indexing='ij')
    q_grid = q_grid.flatten()
    chi_A_grid = chi_A_grid.flatten()
    chi_B_grid = chi_B_grid.flatten()

    grid_points = np.vstack((q_grid, chi_A_grid, chi_B_grid)).T

    return grid_points

def save_training_set_3d_plot(samples, outdir):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        samples[:, 2],
        marker="o",
        s=10,
        alpha=0.75,
        depthshade=True,
        linewidths=0.8,
    )

    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\chi_1$")
    ax.set_zlabel(r"$\chi_2$")

    ax.set_xlim(1.0, 8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.8, 0.8)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ax.set_yticks([-0.8, -0.4, 0.0, 0.4, 0.8])
    ax.set_zticks([-0.8, -0.4, 0.0, 0.4, 0.8])

    ax.set_title(f"{len(samples)} waveforms in the training set")

    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.08, top=0.90)
    plt.savefig(os.path.join(outdir, "training_set.png"), dpi=250)
    plt.close(fig)
    return None

def get_chirptime(xi, M, f_min):
    q = xi[0] 
    s1z = xi[1]
    s2z = xi[2]
    q_IMRPhenomD = 1.0 / q
    m_1 = M / (1 + q_IMRPhenomD)
    m_2 = M * q_IMRPhenomD / (1 + q_IMRPhenomD)
    m_1_SI = m_1*lal.MSUN_SI
    m_2_SI = m_2*lal.MSUN_SI

    return lalsim.SimIMRPhenomDChirpTime(m_1_SI, m_2_SI, s1z, s2z, f_min)

def get_idx_and_duration_of_longest_signal(xi_values, M, f_min):
    """
    Return index and approximate duration T (seconds) corresponding 
    to the longest signal among the xi values.
    """
    chirptimes = np.zeros(len(xi_values))
    for i, xi in enumerate(xi_values):
        chirptimes[i] = get_chirptime(xi, M, f_min)
    idx = np.argmax(chirptimes)
    return idx, chirptimes[idx]

def get_f_mins_given_duration(xi_values , M, T):
    """
    Return array of f_min (Hz) that approximately produce a signal of duration T (seconds)
    for each xi value.
    """
    f_mins = np.zeros(len(xi_values))
    for i, xi in enumerate(xi_values):
        q = xi[0] 
        q_IMRPhenomD = 1.0 / q
        m_1 = M / (1 + q_IMRPhenomD)
        m_2 = M * q_IMRPhenomD / (1 + q_IMRPhenomD)
        eta = (m_1 * m_2) / M**2
        chirp_mass = eta ** (3/5) * M
        f_mins[i] = 134 * (1.21 / chirp_mass) ** (5/8) * (1.0 / T) ** (3/8)
    return f_mins

def make_time_domain_dictionary(loaded_surrogate, xi_values, M, f_mins):
    """
    Create dictionary of time-domain surrogate waveforms with harmonics 
    restricted to (l,|m|)=(2,2), without zero-padding or windowing.
    """
    time_domain_dictionary = {}
    for i, xi in enumerate(tqdm(xi_values)):
        q = xi[0] 
        chi_1 = [0, 0, xi[1]]
        chi_2 = [0, 0, xi[2]]
        dist_mpc = 100
        dt = 1/(4096)
        fs = 1/dt
        f_min = f_mins[i]
        inclination = 0.0
        phi_ref = 0.0

        t, h, _ = loaded_surrogate(q, chi_1, chi_2, dt=dt, f_low=f_min, f_ref=f_min, mode_list=[(2,2)], M=M, dist_mpc=dist_mpc, 
                   inclination=inclination, phi_ref=phi_ref, units='mks')
        h_plus = h.real
        time_domain_dictionary[str(xi)] = {'xi': xi,'t': t, 'h': h_plus}
    return time_domain_dictionary

def time_crop_dictionary_to_shortest_signal(time_domain_dictionary):
    min_len = min(len(data["t"]) for data in time_domain_dictionary.values())
    cropped_time_domain_dictionary = {}
    for data in time_domain_dictionary.values():
        xi = data['xi']
        t = data['t']
        h = data['h']
        len_diff = len(h) - min_len
        h = h[len_diff:]
        t = t[len_diff:]
        cropped_time_domain_dictionary[str(xi)] = {'xi': xi,'t': t, 'h': h}
    return cropped_time_domain_dictionary

def zeropad_and_window_time_domain_dictionary(time_domain_dictionary):
    """
    Add 2 seconds of zero padding after the ringdown, then window the 
    signal with Tukey (roll off = 0.4 s).
    """
    windowed_time_domain_dictionary = {}
    for data in tqdm(time_domain_dictionary.values()):
        xi = data['xi']
        t = data['t']
        h = data['h']
        dt = t[1] - t[0]
        zero_pad = 2.0
        zero_t = np.arange(t[-1] + dt, t[-1] + zero_pad, dt)
        number_of_zeros = len(zero_t)
        t_pad = np.concatenate((t,zero_t))
        h_pad = np.pad(h,(0,number_of_zeros))
        duration_pad = t_pad[-1] - t_pad[0]
        roll_off = 0.4
        alpha = 2.0 *roll_off/duration_pad
        tukey_window = scipy.signal.windows.tukey(len(h_pad), alpha=alpha)
        h_tukey = h_pad * tukey_window
        windowed_time_domain_dictionary[str(xi)] = {'xi': xi,'t': t_pad, 'h': h_tukey}
               
    additional_crop_time_domain_dictionary = time_crop_dictionary_to_shortest_signal(windowed_time_domain_dictionary)
    return additional_crop_time_domain_dictionary

def get_constant_psd_mismatch(h1, h2, df):
    """
    Compute mismatch without dephasing or minimization over dephasing. 
    This implementation only works for constant PSD S_n (any real value).
    Requires already cropped frequency-domain waveforms (e.g. already in 
    the domain [30, 366] Hz).
    """
    def get_constant_psd_inner_product(h1, h2, df):
        """
        Compute inner product. This implementation only works for constant 
        PSD S_n (any real value).
        """
        integrand = np.conj(h1) * h2
        integral = np.trapz(integrand, dx=df)
        return 4.0 * np.real(integral)
    
    h1_norm = get_constant_psd_inner_product(h1,h1,df)
    h2_norm = get_constant_psd_inner_product(h2,h2,df)
    match = get_constant_psd_inner_product(h1,h2,df) / np.sqrt(h1_norm*h2_norm)
    return (1.0 - match)

def get_constant_psd_mismatch_with_dephasing(alignment_params, freqs, h1, h2):
    """
    Compute mismatch after applying a time and phase shift.
    This implementation only works for constant PSD S_n (any real value).
    Requires already cropped frequency-domain waveforms
    (e.g. already in the domain [30, 366] Hz).
    """
    time_shift = alignment_params[0]
    phase_shift = alignment_params[1]
    dephasing = np.exp(2 * np.pi * 1j * freqs * time_shift + 1j * phase_shift)
    df = freqs[1] - freqs[0]
    return get_constant_psd_mismatch(h1, dephasing * h2, df)

def get_lalsuite_IMRPhenomD_h_plus(M, xi, freqs_fft, f_min, f_max):
    delta_f = freqs_fft[1] - freqs_fft[0]
    f_lower = freqs_fft[0] # This NEEDS to be freqs_fft[0] and not f_min, or the alignment won't hold for ripple's IMRPhenomD
    f_ref = freqs_fft[0] # This NEEDS to be freqs_fft[0] and not f_min, or the alignment won't hold for ripple's IMRPhenomD
    q = xi[0] 
    chiA = [0, 0, xi[1]]
    chiB = [0, 0, xi[2]]
    dist_mpc = 100
    inclination = 0.0
    coa_phase = 0.0
    q_IMRPhenomD = 1 / q
    m_1 = M / (1+q_IMRPhenomD)
    m_2 = M * q_IMRPhenomD / (1+q_IMRPhenomD)
    hp_tilde, hc_tilde = get_fd_waveform(approximant='IMRPhenomD',
                                        mass1=m_1,mass2=m_2,
                                        spin1z=chiA[2],spin2z=chiB[2],
                                        distance=dist_mpc,
                                        inclination=inclination,
                                        f_lower=f_lower,
                                        delta_f=delta_f,
                                        f_ref=f_ref,
                                        coa_phase= coa_phase
                                        )
    mask = (hp_tilde.sample_frequencies>=f_min) & (hp_tilde.sample_frequencies<= f_max)    
    freqs = hp_tilde.sample_frequencies
    freqs = freqs[mask]
    lalsuite_IMRPhenomD_h_plus = hp_tilde[mask]

    if len(freqs) != len(freqs_fft):
        raise ValueError(
            f"Frequency-grid length mismatch: len(freqs)={len(freqs)} "
            f"but len(freqs_fft)={len(freqs_fft)}."
        )

    cumulative_error_from_frequency_grid_misalignment = np.sum(freqs - freqs_fft)
    if cumulative_error_from_frequency_grid_misalignment >= delta_f / 100:
        raise ValueError(
            "Frequency-grid misalignment between LAL and surrogate waveforms is too large: "
            f"sum(freqs - freqs_fft) ={cumulative_error_from_frequency_grid_misalignment:.5e}, "
            f"which is greater than delta_f/100 ={delta_f/100:.5e} "
            f"(with delta_f={delta_f:.5e})."
        )
    
    return freqs, lalsuite_IMRPhenomD_h_plus

def get_constant_psd_aligned_waveform_and_minimized_mismatch(guess_t_c, freqs_fft, lalsuite_IMRPhenomD_h_plus, data_fft_tukey, mismatch_tolerance):
    """
    Align a frequency-domain waveform to a reference waveform by minimizing the
    constant-PSD mismatch over a time shift and phase shift.

    The function repeatedly runs a Nelder-Mead minimization of the mismatch with
    respect to:
    - t_c: time shift
    - phi_c: phase shift

    The minimizer is initialized with the provided time-shit guess and a sequence
    of phase guesses. If no minimum below the requested mismatch tolerance is found
    after several attempts, the tolerance is relaxed slightly until convergence is
    achieved.

    Parameters
    ----------
    guess_t_c : float
        Initial guess for the time shift used by the minimizer.

    freqs_fft : np.ndarray
        Frequency grid associated with the input frequency-domain waveforms.

    lalsuite_IMRPhenomD_h_plus : np.ndarray
        Reference frequency-domain waveform to which the data waveform is aligned.

    data_fft_tukey : np.ndarray
        Frequency-domain waveform to be aligned to the reference waveform.

    mismatch_tolerance : float
        Target upper bound for the minimized mismatch. If the optimizer fails to
        reach this value after repeated attempts, the tolerance is increased
        gradually.

    Returns
    -------
    aligned_data_fft_tukey : np.ndarray
        The input waveform after applying the optimal time and phase shift.

    f_min_value : float
        The minimized mismatch value obtained after alignment.

    Note:
    This function assumes a constant noise power spectral density (PSD), as required
    by `get_constant_psd_mismatch_with_dephasing`.
    """
    f_min_value = 1.0
    phi_guess = -30.0
    attempt_number = 0
    original_mismatch_tolerance = mismatch_tolerance
    while f_min_value > mismatch_tolerance:
        min_guess = [guess_t_c, phi_guess]
        f_min_point, f_min_value, i0 , i1, i2 = fmin(get_constant_psd_mismatch_with_dephasing, min_guess, args=(freqs_fft, lalsuite_IMRPhenomD_h_plus, data_fft_tukey), full_output=1, disp=0)
        phi_guess += 5.00
        attempt_number += 1
        if attempt_number > 10:
            mismatch_tolerance *= 1.1
            print(f"Minimizer could not find a minimum below the given mismatch tolerance after {attempt_number} attempts")
            print(f"Mismatch tolerance increased from {original_mismatch_tolerance} to {mismatch_tolerance}")
    mismatch_tolerance = original_mismatch_tolerance

    # print(f"Alignment parameters t_c: {f_min_point[0]:.2f}, phi_c: {f_min_point[1]:.2f}")
    dephasing = np.exp(2*np.pi*1j*freqs_fft*(f_min_point[0]) + 1j*f_min_point[1])
    aligned_data_fft_tukey = dephasing * data_fft_tukey
    return aligned_data_fft_tukey, f_min_value

def get_thinned_waveform(freqs, freqs_fft, lalsuite_IMRPhenomD_h_plus, aligned_data_fft_tukey, chosen_len = None):
    """
    Reduce the number of samples in all input arrays by applying the
    same thinning factor, chosen so that the output has approximately `chosen_len` samples.
    """
    current_len = len(freqs_fft)
    print(f"{current_len} samples before thinning")
    if chosen_len == None or chosen_len < 0:
        chosen_len = current_len
    closest_integer_slicing_every = int(np.ceil(current_len / chosen_len))
    
    thinned_freqs = freqs[::closest_integer_slicing_every]
    thinned_freqs_fft = freqs_fft[::closest_integer_slicing_every]
    thinned_lalsuite_IMRPhenomD_h_plus = lalsuite_IMRPhenomD_h_plus[::closest_integer_slicing_every]
    thinned_aligned_data_fft_tukey = aligned_data_fft_tukey[::closest_integer_slicing_every]
    
    is_evenly_spaced = np.allclose(np.diff(freqs_fft), np.diff(freqs_fft)[0])
    assert is_evenly_spaced
    print(f"{len(thinned_freqs_fft)} samples after thinning")
    return thinned_freqs, thinned_freqs_fft, thinned_lalsuite_IMRPhenomD_h_plus, thinned_aligned_data_fft_tukey

def make_thinned_frequency_domain_dictionary(M, f_min, f_max, chosen_len , dt, time_domain_dictionary, mismatch_tolerance = 0.01):
    """
    Create dictionary of frequency-domain waveforms from a time-domain dictionary, aligned with lalsuite IMRPhenomD.
    Requires waveforms in the time-domain dictionary to be already padded and windowed.
    """
    frequency_domain_dictionary = {}
    for key, data in tqdm(time_domain_dictionary.items()):
        xi = data['xi']
        t = data['t']
        h = data['h']
        
        freqs_fft = np.fft.fftfreq(len(h), dt)
        data_fft_tukey = np.fft.fft(h) * dt
        
        frequency_mask = (freqs_fft >= f_min) & (freqs_fft <= f_max)
        data_fft_tukey = data_fft_tukey[frequency_mask]
        freqs_fft = freqs_fft[frequency_mask]

        freqs, lalsuite_IMRPhenomD_h_plus = get_lalsuite_IMRPhenomD_h_plus(M, xi, freqs_fft, f_min, f_max)

        guess_t_c = -t[-1]
        aligned_data_fft_tukey, minimized_mismatch = get_constant_psd_aligned_waveform_and_minimized_mismatch(guess_t_c, freqs_fft, lalsuite_IMRPhenomD_h_plus, data_fft_tukey, mismatch_tolerance)
        print(
            f"For (q, chi_1, chi_2) = ({xi[0]:.3f}, {xi[1]:.3f}, {xi[2]:.3f})\n"
            f"Surrogate vs LAL IMRPhenomD comparison\n"
            f"Minimized mismatch before thinning: {minimized_mismatch:.6e}"
        )

        df = freqs_fft[1] - freqs_fft[0]
        mismatch_before_thinning = get_constant_psd_mismatch(lalsuite_IMRPhenomD_h_plus, aligned_data_fft_tukey, df)
        print(
            f"Non-minimized mismatch before thinning, after alignment: {mismatch_before_thinning:.6e}"
        )

        thinned_freqs, thinned_freqs_fft, thinned_lalsuite_IMRPhenomD_h_plus, thinned_aligned_data_fft_tukey = get_thinned_waveform(freqs, freqs_fft, lalsuite_IMRPhenomD_h_plus, aligned_data_fft_tukey, chosen_len)

        thinned_df = thinned_freqs_fft[1] - thinned_freqs_fft[0]
        mismatch_after_thinning = get_constant_psd_mismatch(thinned_lalsuite_IMRPhenomD_h_plus, thinned_aligned_data_fft_tukey, thinned_df)
        print(f"Non-minimized mismatch after thinning, after alignment: {mismatch_after_thinning:.6e}")
        
        frequency_domain_dictionary[key] ={'xi': xi, 'freqs_fft': thinned_freqs_fft, 'aligned_data_fft_tukey': thinned_aligned_data_fft_tukey, 'freqs': thinned_freqs, 'lalsuite_IMRPhenomD_h_plus': thinned_lalsuite_IMRPhenomD_h_plus}
        print("")
    return frequency_domain_dictionary

# def get_minimized_mismatch(h1, h2, psd, freq):
#     """
#     Return the mismatch minimized over a time and phase difference between h1 and h2.
#     Requires already cropped frequency-domain waveforms
#     (e.g. already in the domain [30, 366] Hz).
#     Adapted from an implementation provided by Carl-Johan Haster (https://github.com/cjhaster).
#     """
#     def innprod_max(h1, h2, psd, freq):
#         """
#         Return the inner product between h1 and h2 maximized over a time and phase 
#         difference, using the FFT trick.
#         """
#         dot1  = h1*np.conjugate(h2) / psd
#         dot2 = 1j*dot1

#         transform1  = np.fft.irfft(dot1)
#         transform2  = np.fft.irfft(dot2)

#         transform = np.sqrt(np.square(transform1) + np.square(transform2))

#         dt = 1.0 / (2.0 * freq[-1])
#         transform_0 = transform.max()

#         if transform.argmax() == transform.size - 1:
#             transform_p = transform[0]
#             transform_m = transform[transform.argmax() - 1]
#         elif transform.argmax()==0:
#             transform_p = transform[transform.argmax() + 1]
#             transform_m = transform[-1]
#         else:
#             transform_p=transform[transform.argmax()+1]
#             transform_m=transform[transform.argmax()-1]

#         t_peak = -dt*(transform_p - transform_m)/(2*transform_p+2*transform_m - 4*transform_0)
#         transform_max= transform_0  + t_peak*(transform_p - transform_m)/(2*dt) + \
#                     t_peak**2. * (0.5*transform_p + 0.5*transform_m - transform_0)/(dt**2)

#         return 4.0 * transform_max*(freq[-1] - freq[0])
#     return (1.0 - (innprod_max(h1, h2, psd, freq) / np.sqrt(innprod_max(h1, h1, psd, freq)*innprod_max(h2, h2, psd, freq))))

def plot_aligned_training_waveforms(thinned_frequency_domain_dictionary, number_of_waveforms_to_plot, outdir):
    random.seed(99)
    keys = list(thinned_frequency_domain_dictionary.keys())
    random_keys = random.sample(keys, number_of_waveforms_to_plot)

    plt.rcParams["figure.figsize"] = [10,8]
    fig, axs = plt.subplots(number_of_waveforms_to_plot, dpi=250, sharex=True)

    for i, key in enumerate(random_keys):
        data = thinned_frequency_domain_dictionary[key]
        xi, freqs_fft, aligned_data_fft_tukey = data['xi'], data['freqs_fft'], data['aligned_data_fft_tukey']
        freqs, lalsuite_IMRPhenomD_h_plus = data['freqs'], data['lalsuite_IMRPhenomD_h_plus']
        
        axs[i].plot(freqs_fft, np.real(aligned_data_fft_tukey), label='NRHybSur3dq8', alpha=0.5)
        axs[i].plot(freqs, np.real(lalsuite_IMRPhenomD_h_plus), label='IMRPhenomD', ls=':', color='k', alpha=.5)
        axs[i].set_xlim(freqs_fft[0] - 2, freqs_fft[-1] + 4)
        axs[i].set_ylabel(r"$\mathrm{Re}(\tilde{h}_+)$", fontsize=13)
        axs[i].tick_params(axis="both", labelsize=11)
        
        psd_unity = np.ones_like(aligned_data_fft_tukey)
        this_mismatch_min = get_minimized_mismatch(lalsuite_IMRPhenomD_h_plus, aligned_data_fft_tukey, psd_unity, freqs_fft)
        df = freqs_fft[1] - freqs_fft[0]
        this_mismatch = get_constant_psd_mismatch(lalsuite_IMRPhenomD_h_plus, aligned_data_fft_tukey, df)
        axs[i].set_title(
        fr"$(q,\chi_1,\chi_2)=({xi[0]:.3f}, {xi[1]:.3f}, {xi[2]:.3f})$, "
        fr"minimized mismatch: {this_mismatch_min:.6e}, "
        fr"non-minimized mismatch: {this_mismatch:.6e}",
        fontsize=11,
        )
    axs[-1].set_xlabel(r"$f\;[\mathrm{Hz}]$", fontsize=13)
    plt.legend(fontsize=11)

    fig.suptitle(
        f"{number_of_waveforms_to_plot} of the {len(thinned_frequency_domain_dictionary)} training waveforms",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outdir + '/aligned_training_waveforms.png')
    plt.close()
    return None

def save_training_set(thinned_frequency_domain_dictionary, outdir):
    for j, (key, data) in enumerate(tqdm(thinned_frequency_domain_dictionary.items())):
        xi, freqs_fft, aligned_data_fft_tukey = data['xi'], data['freqs_fft'], data['aligned_data_fft_tukey']
        name_label = [['q_','chi1_','chi2_'][i] + format(xi[i], '.5f') for i in range(3)]
        np.savez(outdir +'/' + '_'.join(name_label)+ '.npz', xi=xi, freqs_ftt=freqs_fft, aligned_data_fft_tukey=aligned_data_fft_tukey)
    return None