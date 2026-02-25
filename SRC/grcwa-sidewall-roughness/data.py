from periodictable.xsf import index_of_refraction


def get_optical_constants(compound: str, wavelength: float) -> complex:
    """
    Get the complex refractive index of a compound at a given wavelength.

    Args:
        compound (str): The compound name (e.g., "Ni").
        wavelength (float): The wavelength in nm.

    Returns
    -------
        complex: The complex refractive index.
    """
    wavelength_angstrom = wavelength * 1e1  # Convert to Angstrom
    nk = index_of_refraction(compound, wavelength=wavelength_angstrom)
    return nk.real + abs(nk.imag) * 1j
