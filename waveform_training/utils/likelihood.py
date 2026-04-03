from utils.h_plus import get_h_plus_ripple_lambda_sigma
from jax import lax, vmap
import jax.numpy as jnp
from jax.scipy.special import i0e, logsumexp

def get_phase_and_time_marginalized_log_likelihood_FFT_no_vmap(lambda_sigma, data_array):
    """
    Return Log of [Likelihood marginalized analytically over phi_c and numerically via FFT over t_c],
    summed over all waveforms in the training dataset.
    
    Note
    ----
    Instead of using vmap, a loop (lax.scan) over the training waveforms is used. 
    This slows down the inference but requires less GPU memory to be allocated, hence allowing
    to load a bigger training set.
    """
    def slim(a):
        """
        Turn Array([[...]]) into Array([...]).
        """
        return a.reshape((a.shape[1],))

    def unpack_params_and_data(a):
        """
        Unpack xi, PSD, frequency array, waveform array from a single element of the data array.
        """
        xi = slim(lax.dynamic_slice(a,(0,a.shape[1]-4),(1,3)))
        Sn = slim(lax.dynamic_slice(a,(0,a.shape[1]-1),(1,1)))
        freqs = slim(lax.dynamic_slice(a,(0,0),(1,a.shape[1]-4)))
        re_fft = slim(lax.dynamic_slice(a,(1,0),(1,a.shape[1]-4)))
        im_fft = slim(lax.dynamic_slice(a,(2,0),(1,a.shape[1]-4)))
        fft = re_fft + 1j * im_fft
        return xi, Sn, freqs, fft

    def get_ln_i0(value):
        """
        Compute log(I_0(x)), I_0(x) where is the modified Bessel function of order 0.
        """
        return jnp.log(i0e(value)) + value

    def get_phase_marginalized_log_likelihood_element(lambda_sigma, data_array_element):
        """
        Note:
        The inner product of h with itself is not included, because it does not
        depend on `lambda_sigma` or on the coalescence time `t_c`.
        """
        xi, Sn, freqs, fft = unpack_params_and_data(data_array_element)
        df = freqs[1] - freqs[0]
        model = get_h_plus_ripple_lambda_sigma(lambda_sigma, xi, freqs)
        dot = model * jnp.conjugate(fft) / Sn
        fft_of_dot = jnp.fft.fftshift(jnp.fft.fft(dot))* df
        return get_ln_i0(4.0*jnp.abs(fft_of_dot))

    def get_phase_and_time_marginalized_log_likelihood_element(lambda_sigma, data_array_element):
        """
        Return the log of the integral of I_0 over t_c. The interval dt_c is dropped since
        it is an overall additive constant.
        """
        logl_at_t_c_array = get_phase_marginalized_log_likelihood_element(lambda_sigma, data_array_element)
        return logsumexp(logl_at_t_c_array)

    def scan_func(carry, data_array_element):
        return carry + get_phase_and_time_marginalized_log_likelihood_element(lambda_sigma, data_array_element), None

    total_log_likelihood, _ = lax.scan(scan_func, 0.0, data_array)
    return total_log_likelihood

def get_phase_and_time_marginalized_log_likelihood_FFT(lambda_sigma, data_array):
    """
    Return Log of [Likelihood marginalized analytically over phi_c and numerically via FFT over t_c],
    summed over all waveforms.
    The likelihood evaluation across waveforms is parallelized with `vmap`.

    Note
    ----
    Using `vmap` significantly speeds up the likelihood evaluation, but at the cost of
    increasing GPU memory usage because each waveform requires an FFT.
    """
    def slim(a):
        """
        Turn Array([[...]]) into Array([...]).
        """
        return a.reshape((a.shape[1],))

    def unpack_params_and_data(a):
        """
        Unpack xi, PSD, frequency array, waveform array from a single element of the data array.
        """
        xi = slim(lax.dynamic_slice(a,(0,a.shape[1]-4),(1,3)))
        Sn = slim(lax.dynamic_slice(a,(0,a.shape[1]-1),(1,1)))
        freqs = slim(lax.dynamic_slice(a,(0,0),(1,a.shape[1]-4)))
        re_fft = slim(lax.dynamic_slice(a,(1,0),(1,a.shape[1]-4)))
        im_fft = slim(lax.dynamic_slice(a,(2,0),(1,a.shape[1]-4)))
        fft = re_fft + 1j * im_fft
        return xi, Sn, freqs, fft

    def get_ln_i0(value):
        """
        Compute log(I_0(x)), I_0(x) where is the modified Bessel function of order 0.
        """
        return jnp.log(i0e(value)) + value

    def get_phase_marginalized_log_likelihood_element(lambda_sigma, data_array_element):
        """
        Note:
        The inner product of h with itself is not included, because it does not
        depend on `lambda_sigma` or on the coalescence time `t_c`.
        """
        xi, Sn, freqs, fft = unpack_params_and_data(data_array_element)
        df = freqs[1] - freqs[0]
        model = get_h_plus_ripple_lambda_sigma(lambda_sigma, xi, freqs)
        dot = model * jnp.conjugate(fft) / Sn
        fft_of_dot = jnp.fft.fftshift(jnp.fft.fft(dot))* df
        return get_ln_i0(4.0*jnp.abs(fft_of_dot))

    def get_phase_and_time_marginalized_log_likelihood_element(lambda_sigma, data_array_element):
        """
        Return the log of the integral of I_0 over t_c. The interval dt_c is dropped since
        it is an overall additive constant.
        """
        logl_at_t_c_array = get_phase_marginalized_log_likelihood_element(lambda_sigma, data_array_element)
        return logsumexp(logl_at_t_c_array)

    log_likelihood_array = vmap(get_phase_and_time_marginalized_log_likelihood_element, in_axes=(None, 0))
    return jnp.sum(log_likelihood_array(lambda_sigma, data_array))