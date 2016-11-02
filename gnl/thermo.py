"""
Thermodynamics utility functions
"""
import numpy as np

rd = 287.1
cpd = 1004
gr = 9.81
kappa = rd / cpd


def hydrostatic_pressure_thz(th, z, ps=1000e2):
    """Hydrostatic pressure obtained by finite difference integration

    Parameters
    ----------
    th :
		potential temperature (deg K)
	z :
		height values (meters)
	ps : Optional[float]
		surface pressure (Pa)
		
	Returns
	-------
	Hydrostatic pressure (Pa)

    """
    p = np.zeros_like(z)
    p[0] = ps

    for i in range(1, z.shape[0]):
        rho = ps**kappa / rd / th[i - 1] * p[i - 1]**(1 - kappa)
        p[i] = p[i - 1] - rho * gr * (z[i] - z[i - 1])

    return p
