"""
Read data from the TOGA COARE soundings available at
http://tornado.atmos.colostate.edu/togadata/ifa_data.html#DATA
"""
import os
import datetime
import xarray as xray
import numpy as np
from numpy import linspace, arange, concatenate
from scipy.interpolate import interp1d
from pylab import *
import scipy.sparse.linalg as sla

close('all')
# ion()

rd = 287.1
cpd = 1004
gr = 9.81
kappa = rd / cpd


class EigOmega(sla.LinearOperator):
    """Linear operator for the vertical velocity eigenfunction problem"""
    def __init__(self, p, T, *args, **kwargs):
        super(EigOmega, self).__init__(*args, **kwargs)

        self._p = p.copy()
        self._T = T.copy()

    def _matvec(self, v):

        kappa = 7 / 2

        T = self._T
        p = self._p

        p = np.hstack((2 * p[0] - p[1], p, 2 * p[-1] - p[-2]))
        T = np.hstack((T[0], T, 2 * T[-1] - T[-2]))
        self._pg = p

        # Boundary condition at surface
        v_top = -v[0]  # Dirichlet
        v_bot = -v[-1]

        v = np.hstack((v_top, v, v_bot))

        dp = p[2:] - p[:-2]
        dT = T[2:] - T[:-2]

        beta = p[1:-1]**2 / (kappa * T[1:-1] - p[1:-1] * dT / dp)

        dVdp = (v[1:] - v[:-1]) / (p[1:] - p[:-1])
        ph = (p[1:] + p[:-1]) / 2

        d2Vdp2 = (dVdp[1:] - dVdp[:-1]) / (ph[1:] - ph[:-1])

        y = d2Vdp2 / beta

        return y


class Eig(sla.LinearOperator):
    """Hermitian Finite difference operator for Sturm Liouville problem"""

    def __init__(self, p, th, *args, **kwargs):
        super(Eig, self).__init__(*args, **kwargs)

        self._p = p.copy()
        self._th = th.copy()

    def _matvec(self, v):

        th = self._th
        p = self._p

        p = np.hstack((2 * p[0] - p[1], p, 2 * p[-1] - p[-2]))
        th = np.hstack((2 * th[0] - th[1], th, 2 * th[-1] - th[-2]))

        self._pg = p
        temp = th * (p / 1000e2)**(kappa)
        rho = p / rd / temp

        #  Weight the columns as in Kasahari
        w = 1 / sqrt(.5 * (p[2:] - p[:-2]))
        v *= w

        # No normal flow boundary conditions
        v = np.hstack((v[0], v, v[-1]))

        # Calculate N2
        ph = (p[1:] + p[:-1]) / 2
        rh = (rho[1:] + rho[:-1]) / 2

        dp = p[1:] - p[:-1]
        dth = th[1:] - th[:-1]

        dV = v[1:] - v[:-1]

        N2 = -rh * 9.81**2 * 2 / (th[1:] + th[:-1]) * dth / dp
        beta = ph**2 * N2

        df = beta * dV / dp

        # Weight the rows as in Kasahari
        y = (df[1:] - df[:-1])
        y /= w

        return y


def linop_to_dense(sl):
    """Convert linop to dense matrix"""
    m, n = sl.shape

    A = np.vstack([sl.dot(np.eye(n)[i, :]) for i in range(m)])

    return A


def dot_prod(u, v, pbound):
    dp = pbound[1:] - pbound[:-1]
    return sum(u * v * dp) / (pbound[-1] - pbound[0])


def vert_int(u, pbound):
    dp = pbound[1:] - pbound[:-1]
    return hstack((0, cumsum(u * dp) / (pbound[-1] - pbound[0])))


def u_modes(th, p, pbound, inds=range(3, 8)):
    """Calculate baroclinic modes using sturm lioville problem from Kasahara
    and Puri

    Args
    -----
    th (nz): potential temperature
    p (nz): cell-centered pressure
    pbound (nz+1) : cell-boundary pressure

    Returns
    -------
    u_modes, u_proj, w_modes, w_proj

    """

    # convert to list just in case
    inds = list(inds)

    sl = Eig(p, th, np.float64, (len(th), len(th)))

    A = linop_to_dense(sl)

    S, P = np.linalg.eig(A)

    ind = np.argsort(S)
    from scipy.integrate import cumtrapz
    print(sqrt(-1 / S[ind]))

    Psi = P[:, ind[inds]]
    Phi = np.zeros((P.shape[0], Psi.shape[1]))
    dp = pbound[1:] - pbound[:-1]

    # Allocate output
    Psi_out = []
    Phi_out = [] 



    for i in inds:
        psi = P[:, ind[-i]]

        # Normalize and get right sign for psi
        psinorm = sqrt(dot_prod(psi, psi, pbound))
        psi = psi / psinorm * sign(psi[-1])

        # The magnitude of phi seems ok
        phi = - vert_int(psi, pbound)
        phi -= phi[-1]

        # Centered version of phi
        phih = (phi[1:] + phi[:-1]) / 2
        phih *= sign(phi[-3])
        # phih /= np.max(phih[p > 400e2])

        # Store in appropriate arrays
        Psi_out.append(psi)  # u-modes
        Phi_out.append(phih)  # w-modes

        figure(1)
        # plot(ψ, sl._pg, label='Mode {0}'.format(i))
        plot(psi, p, label='Mode {0}'.format(i))
        yticks(linspace(0, 1000e2, 10))

        figure(2)
        # plot(ψ, sl._pg, label='Mode {0}'.format(i))
        plot(phih, p, label='Mode {0}'.format(i))
        yticks(linspace(0, 1000e2, 10))

    figure(1)
    gca().invert_yaxis()
    legend()

    figure(2)
    gca().invert_yaxis()
    axvline(0)
    legend(loc="center left")

    Psi_out[0] *= 6.25
    Phi_out[0] *= 6.25

    Psi_out[1] *= 12.5
    Phi_out[1] *= 12.5

    # Compute projection operators
    aPsi = pinv(Psi_out)

    # This should remove the mean first
    projMean = np.eye(th.shape[0]) - np.outer(th, th)/np.dot(th,th)
    aPhi = pinv(Phi_out)
    aPhiNoMean = projMean.T.dot(aPhi)

    # Expand columns out to lists
    aPsi = [aPsi[:, i] for i in range(aPsi.shape[1])]
    aPhi = [aPhiNoMean[:, i] for i in range(aPhiNoMean.shape[1])]

    return Psi_out, aPsi, Phi_out, aPhi


def w_modes(T, p):
    """Find the normal functions for the vertical velocity"""
    sl = EigOmega(p, T, np.float64, (len(T), len(T)))

    A = linop_to_dense(sl)

    S, P = np.linalg.eig(A)

    ind = np.argsort(S)

    for i in range(1, 4):
        psi = P[:, ind[-i]]
        # theta = psi * th
        psi = np.hstack((-psi[0], P[:, ind[-i]], -psi[-1]))
        phi = (psi[1:] - psi[:-1]) / (sl._pg[1:] - sl._pg[:-1])
        psi = psi[1:-1]
        ph = (sl._pg[1:] + sl._pg[:-1]) / 2

        figure(1)
        plot(psi, p, label='Mode {0}'.format(i))

        figure(2)
        plot(phi, ph, label='Mode {0}'.format(i))

        # figure(3)
        # plot(theta, p, label='Mode {0}'.format(i))

    figure(1)
    gca().invert_yaxis()
    axhline(0)

    figure(2)
    gca().invert_yaxis()
    axhline(0)

    # figure(3)
    # gca().invert_yaxis()

    return P


def pressure_grid():
    ntrop = 20
    nstrat = 8

    ptop = 25e2
    pbot = 1000e2
    ppause = 200e2

    dtrop = (pbot - ppause) / ntrop
    dstrat = (ptop - ppause) / nstrat

    # vertical boundaries
    p = concatenate((linspace(ptop, ppause, nstrat + 1), linspace(
        ppause, pbot, ntrop + 1)[1:]))

    # center values
    pcent = (p[1:] + p[:-1]) / 2
    return p, pcent


def getsound(sound):
    columns = ['p', 'z', 'T', 'q', 'rh', 'u', 'v']

    tdat = np.fromstring(sound[0], dtype=np.int, sep=' ')

    time = datetime.datetime(tdat[0] + 1900, *tdat[1:])

    arr = np.fromstring(''.join(sound[1:]), sep=' ')\
            .reshape((-1, len(columns)))

    da = xray.Dataset(
        {'p': (['i'], arr[:, 0]),
         'T': (['i'], arr[:, 2])},
        coords={'time': time,
                'i': np.arange(arr.shape[0])})

    return da


def load_sounds(fname):

    f = open(fname)
    lines = f.readlines()

    sounds = [lines[i:i + 42] for i in range(0, len(lines), 42)]
    return xray.concat([getsound(sound) for sound in sounds], dim='time')


def plot_sound(avg):
    T = avg['T']
    p = avg['p']
    plot(T, p)
    gca().invert_yaxis()
    grid()


def read_wrf(fname='input_sounding'):
    """Read input from a WRF idealized sounding text file"""
    import re
    with open(fname) as f:
        line = f.readline()
        ps, ts, us = [float(tok) for tok in re.split(r'\s+', line)
                      if tok != '']

        ps *= 100
        z, T, u = loadtxt("input_sounding",
                          skiprows=1,
                          usecols=(0, 1, 2),
                          unpack=True)

    T = np.hstack((ts, T))
    z = np.hstack((0, z))

    return dict(T=T, z=z, ps=ps)


def pressure_wrf(th, z, ps):
    """Integrate hydrostatic relation to give pressure"""
    p = np.zeros_like(z)
    p[0] = ps

    for i in range(1, z.shape[0]):
        rho = 1000e2**kappa / rd / th[i - 1] * p[i - 1]**(1 - kappa)
        p[i] = p[i - 1] - rho * 9.81 * (z[i] - z[i - 1])

    return p


def interpextrap(pcent, p, T0):
    Tcent = interp1d(p, T0, bounds_error=False)(pcent)
    Tcent[pcent < p[0]] = (T0[1] - T0[0]) / (p[1] - p[0]) * (
        pcent[pcent < p[0]] - p[0]) + T0[0]

    print(Tcent)
    return Tcent


def main(sounding='wrf'):

    # get grid for pressure
    pbound, pcent = pressure_grid()

    ## Get TOGA profiles
    if sounding == 'toga':
        if not os.path.exists('basic_flds'):
            os.system(
                'curl http://tornado.atmos.colostate.edu/togadata/data/ifa_averaged_v2/basic_flds.ifa_v2.1.gz | gunzip > basic_flds')

        toga = load_sounds('basic_flds')
        avg = toga.groupby('time.hour').mean('time')
        prof0 = avg.sel(hour=6)  # This is 3pm local solar time

        T0 = prof0['T'].values + 273.15
        p0 = prof0['p'].values * 100
        th = T0 * (1000e2 / p0)**(rd / cpd)

    elif sounding == 'wrf':
        d = read_wrf("./input_sounding")

        th = d['T']
        z0 = d['z']
        p0 = pressure_wrf(th, z0, d['ps'])
        print(p0)

    # Sort to make increasing in p
    psort = p0.argsort()
    p0 = p0[psort]
    th = th[psort]

    figure(3)
    plot(th, p0)
    gca().invert_yaxis()

    # interpolate onto model grid
    theta_c = interpextrap(pcent, p0, th)

    # generate eigenfunctions and output to text
    # w_modes(Tcent, pcent)
    Psi, aPsi, Phi, aPhi = u_modes(theta_c, pcent, pbound, inds=(3, 5))
    with open("modes.txt", "w") as f:
        for i in range(2):
            Psi[i].tofile(f, sep=" ")
            f.write('\n')
            aPsi[i].tofile(f, sep=" ")
            f.write('\n')
            Phi[i].tofile(f, sep=" ")
            f.write('\n')
            aPhi[i].tofile(f, sep=" ")
            f.write('\n')

    np.savetxt('t0.txt', theta_c)
    # np.savetxt('temp.txt', Tcent)
    np.savetxt('p0.txt', pcent)
    np.savetxt('pdiff.txt', pbound[1:] - pbound[:-1])
    np.savetxt('ph.txt', pbound)

    # save eigenfunctions

    show()


if __name__ == '__main__':
    main()
