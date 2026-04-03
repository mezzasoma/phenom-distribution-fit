import gwsurrogate
from utils.constants import solar_mass_in_seconds
from utils.path_utils import create_and_set_outdir
from utils.training_set_generation import (create_uniform_3d_grid,
                                           save_training_set_3d_plot,
                                           get_idx_and_duration_of_longest_signal,
                                           get_f_mins_given_duration,
                                           make_time_domain_dictionary,
                                           time_crop_dictionary_to_shortest_signal,
                                           zeropad_and_window_time_domain_dictionary,
                                           make_thinned_frequency_domain_dictionary,
                                           plot_aligned_training_waveforms,
                                           save_training_set)
import numpy as np

loaded_surrogate = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
outdir = create_and_set_outdir("NRHybSur3dq8_training_set_frequency_domain")

# Definitions:
# m_1, m_2: component masses, with m_1 >= m_2
# q: mass ratio, defined as q = m_1 / m_2 >= 1
# chi_1, chi_2: z-components of the spins associated with m_1 and m_2, respectively
# M: total mass
# xi: intrinsic training parameters (q, chi_1, chi_2)

q_min, q_max = 1.0000001, 7.95 
chi_min, chi_max = -0.79, 0.79 
n_points_per_dim = 4
initial_64_grid_points = create_uniform_3d_grid(q_min, q_max, chi_min, chi_max, n_points_per_dim)

additional_30_handpicked_points = np.array([
        [ 3.6062500625,  0.79        , -0.5925      ],
        [ 2.737500075 ,  0.79        ,  0.          ],
        [ 1.8687500875,  0.79        , -0.395       ],
        [ 3.6062500625,  0.79        , -0.79        ],
        [ 1.8687500875,  0.79        , -0.5925      ],
        [ 1.8687500875,  0.79        , -0.79        ],
        [ 2.737500075 ,  0.79        , -0.1975      ],
        [ 2.737500075 ,  0.79        , -0.395       ],
        [ 2.737500075 ,  0.79        , -0.5925      ],
        [ 2.737500075 ,  0.79        , -0.79        ],
        [ 3.6062500625,  0.1975      ,  0.79        ],
        [ 4.47500005  ,  0.1975      , -0.79        ],
        [ 3.6062500625,  0.5925      ,  0.5925      ],
        [ 5.3437500375,  0.395       , -0.79        ],
        [ 4.47500005  ,  0.395       ,  0.395       ],
        [ 2.737500075 ,  0.1975      ,  0.79        ],
        [ 2.737500075 ,  0.395       ,  0.5925      ],
        [ 3.6062500625,  0.395       ,  0.395       ],
        [ 4.47500005  ,  0.395       , -0.79        ],
        [ 5.3437500375,  0.5925      ,  0.79        ],
        [ 1.8687500875,  0.1975      ,  0.79        ],
        [ 4.47500005  ,  0.5925      ,  0.5925      ],
        [ 4.47500005  ,  0.395       ,  0.5925      ],
        [ 1.8687500875,  0.395       ,  0.79        ],
        [ 3.6062500625,  0.395       ,  0.5925      ],
        [ 3.6062500625,  0.5925      ,  0.79        ],
        [ 2.737500075 ,  0.395       ,  0.79        ],
        [ 4.47500005  ,  0.395       ,  0.79        ],
        [ 4.47500005  ,  0.5925      ,  0.79        ],
        [ 3.6062500625,  0.395       ,  0.79        ]])

xi_values = np.vstack([initial_64_grid_points, additional_30_handpicked_points])
save_training_set_3d_plot(xi_values, outdir)

M = 10.0
f_min_for_fft = 10.0

idx, max_duration = get_idx_and_duration_of_longest_signal(xi_values, M, f_min_for_fft)
f_mins = get_f_mins_given_duration(xi_values, M, max_duration)

time_domain_dictionary = make_time_domain_dictionary(loaded_surrogate, xi_values, M, f_mins)
cropped_time_domain_dictionary = time_crop_dictionary_to_shortest_signal(time_domain_dictionary)
zeropadded_and_windowed_time_domain_dictionary = zeropad_and_window_time_domain_dictionary(cropped_time_domain_dictionary)

f_min = 30.0
f_max = 0.018 / (M * solar_mass_in_seconds)
desired_number_of_thinned_frequency_samples = 20_000
dt = 1 / 4096.0
mismatch_tolerance = 0.03
thinned_frequency_domain_dictionary = make_thinned_frequency_domain_dictionary(M, f_min, f_max, desired_number_of_thinned_frequency_samples, dt, zeropadded_and_windowed_time_domain_dictionary, mismatch_tolerance)

number_of_waveforms_to_plot = 5
plot_aligned_training_waveforms(thinned_frequency_domain_dictionary, number_of_waveforms_to_plot, outdir)
save_training_set(thinned_frequency_domain_dictionary, outdir)

