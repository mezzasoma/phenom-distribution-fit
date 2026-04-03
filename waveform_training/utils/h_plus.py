from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
import jax.numpy as jnp

def get_h_plus_ripple_lambda_sigma(lambda_sigma, xi, freqs):
    """
    Return the frequency-domain plus polarization of IMRPhenomD for a
    10-solar-mass binary, with intrinsic training parameters xi
        xi = (q, chi_1, chi_2)
    where
        q      = m_1 / m_2 >= 1
        chi_1  = dimensionless aligned spin of the heavier mass
        chi_2  = dimensionless aligned spin of the lighter mass
    over the frequency array `freqs`, for a set of fitting coefficients 
    `lambda_sigma`.
    `lambda_sigma` is a jax array containing the fitting coefficients (lambdas) 
    entering the pseudo-post-Newtonian coefficients sigma_2, sigma_3, and sigma_4.
    Their default IMRPhenomD values are:
    lambda_sigma[0] ~ -10114.05, ..., lambda_sigma[10] ~ 674402.46, 
    lambda_sigma[11] ~ 22933.65, ..., lambda_sigma[21] ~ -3.06e6,
    lambda_sigma[22] ~ -14621.7, ..., lambda_sigma[32] ~ 4.39e6

    Fixed reference parameters:
    total mass            = 10 MSUN
    luminosity distance   = 100 Mpc
    coalescence time      = 0
    coalescence phase     = 0
    inclination           = 0
    polarization angle    = 0
    """
    total_mass = 10 
    q = xi[0]
    chi_1 = xi[1]
    chi_2 = xi[2]    
    inverse_mass_ratio = 1.0 / q
    distance_mpc = 100 
    m_1 = total_mass / (1 + inverse_mass_ratio) 
    m_2 = total_mass * inverse_mass_ratio / (1 + inverse_mass_ratio)
    coalescence_time = 0.0 
    coalescence_phase = 0.0
    inclination = 0.0
    polarization_angle = 0.0
    chirp_mass, symmetric_mass_ratio = ms_to_Mc_eta(jnp.array([m_1, m_2]))

    waveform_astrophysical_parameters = jnp.array([
        chirp_mass,
        symmetric_mass_ratio,
        chi_1,
        chi_2,
        distance_mpc,
        coalescence_time,
        coalescence_phase,
        inclination,
        polarization_angle,
    ])

    h_plus, h_cross = IMRPhenomD.gen_IMRPhenomD_polar(
        freqs,
        waveform_astrophysical_parameters,
        freqs[0],
        lambda_sigma,
    )
    
    return h_plus