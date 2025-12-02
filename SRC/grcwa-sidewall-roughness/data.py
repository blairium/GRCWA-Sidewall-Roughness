from periodictable.xsf import index_of_refraction


def get_optical_constants(compound: str, wavelength: float):
    wavelength *= 1e1  # Convert to Angstrom
    nk = index_of_refraction(compound, wavelength=wavelength)
    return nk.real + abs(nk.imag) * 1j
