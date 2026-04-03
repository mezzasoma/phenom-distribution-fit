from astropy.constants import M_sun, G, c, kpc

solar_mass_in_SI = M_sun.value
gravitaional_constant_in_SI = G.value
speed_of_light_in_SI = c.value
kiloparsec_in_SI = kpc.value
solar_mass_in_seconds = solar_mass_in_SI*gravitaional_constant_in_SI/speed_of_light_in_SI**3
megaparsec_in_seconds = 1e3*kiloparsec_in_SI/speed_of_light_in_SI