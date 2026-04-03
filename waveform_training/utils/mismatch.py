import numpy as np

def get_minimized_mismatch(h1, h2, psd, freq):
    """
    Return the mismatch minimized over a time and phase difference between h1 and h2.
    Requires already cropped frequency-domain waveforms
    (e.g. already in the domain [30, 366] Hz).
    Adapted from an implementation provided by Carl-Johan Haster (https://github.com/cjhaster).
    """
    def innprod_max(h1, h2, psd, freq):
        """
        Return the inner product between h1 and h2 maximized over a time and phase 
        difference, using the FFT trick.
        """
        dot1  = h1*np.conjugate(h2) / psd
        dot2 = 1j*dot1

        transform1  = np.fft.irfft(dot1)
        transform2  = np.fft.irfft(dot2)

        transform = np.sqrt(np.square(transform1) + np.square(transform2))

        dt = 1.0 / (2.0 * freq[-1])
        transform_0 = transform.max()

        if transform.argmax() == transform.size - 1:
            transform_p = transform[0]
            transform_m = transform[transform.argmax() - 1]
        elif transform.argmax()==0:
            transform_p = transform[transform.argmax() + 1]
            transform_m = transform[-1]
        else:
            transform_p=transform[transform.argmax()+1]
            transform_m=transform[transform.argmax()-1]

        t_peak = -dt*(transform_p - transform_m)/(2*transform_p+2*transform_m - 4*transform_0)
        transform_max= transform_0  + t_peak*(transform_p - transform_m)/(2*dt) + \
                    t_peak**2. * (0.5*transform_p + 0.5*transform_m - transform_0)/(dt**2)

        return 4.0 * transform_max*(freq[-1] - freq[0])
    return (1.0 - (innprod_max(h1, h2, psd, freq) / np.sqrt(innprod_max(h1, h1, psd, freq)*innprod_max(h2, h2, psd, freq))))
