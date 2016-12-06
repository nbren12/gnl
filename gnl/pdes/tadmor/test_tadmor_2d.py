import numpy as np
from .tadmor_2d import _slopes, _stagger_avg, Tadmor2D

def tadmor_error(n):
    uc = np.zeros((1, n+ 4, n + 4))

    L = 1.0
    dx = L /n

    x = np.arange(-2,n+2) * dx
    x, y =  np.meshgrid(x,x)

    initcond = lambda x, y: np.exp(-((x%L-.5)/.10)**2) * np.exp(-((y%L-.5)/.10)**2)

    uc[0,:] = initcond(x,y)

    dt = dx/5


    def fx(u):
        return u

    def fy(u):
        return u


    tend = .25
    t = 0

    tad = Tadmor2D()
    tad.fx=fx
    tad.fy=fy
    tad.geom.n_ghost = 8



    while (t < tend - 1e-10):
        dt = min(dt, tend-t)
        tad.central_scheme(uc, dx, dx, dt)
        t+=dt

    return np.mean(np.abs(initcond(x-t, y-t) - uc[0,...]).sum())


def test_tadmor_convergence(plot=False):
    """
    Create error convergence plots for 1d advection problem
    """
    nlist = [50, 100, 200, 400]

    err = [tadmor_error(n) for n in nlist]
    p = np.polyfit(np.log(nlist), np.log( err ), 1)

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(nlist, err)
        plt.title('Order of convergence p = %.2f'%p[0])
        plt.show()

    if abs(p[0]) < 1.9:
        raise ValueError('Order of convergence (p={p})is less than 2'.format(p=-p[0]))


def plot_advection2d():
    n = 200
    uc = np.zeros((1, n+ 8, n + 8))

    L = 1.0
    dx = L /n

    x = np.arange(-4,n+4) * dx
    x, y =  np.meshgrid(x,x)

    initcond = lambda x, y: np.exp(-((x%L-.5)/.10)**2) * np.exp(-((y%L-.5)/.10)**2)*10

    uc[0,:] = initcond(x,y)

    dt = dx/4


    def fx(u):
        return u

    def fy(u):
        return u
    tad = Tadmor2D()
    tad.fx=fx
    tad.fy=fy


    tend = 1.0
    t = 0

    M = uc.max()
    img = plt.plot(uc[0,:,100].T, label='tadmor')
    plt.axis('tight')

 

    while (t < tend - 1e-10):
        dt = min(dt, tend-t)
        uc = tad.central_scheme(uc, dx, dx, dt)
        t+=dt

    plt.plot(uc[0,:,100], label='tadmor')
    plt.axis('tight')
    plt.show()



import matplotlib.pyplot as plt
def plot_stagger_avg():

    n = 200
    m = 100
    uc = np.zeros((1, n+ 4, m + 4))

    L = 1.0
    dx = L /n
    dy = L/m

    x = np.arange(-2,n+2) * dx
    y = np.arange(-2,m+2) * dy
    x, y =  np.meshgrid(x,y, indexing='ij')

    uc[0,:] = np.exp(-((x-.5)/.10)**2)  * np.exp(-((y-.5)/.10)**2)

    zstag = _stagger_avg(uc)
    periodic_bc(zstag, 2, axes=(1,2))
    plt.pcolormesh(zstag[0,...])
    plt.show()



def plot_slopes():

    n = 200
    m = 100
    uc = np.zeros((1, n+ 4, m + 4))

    L = 1.0
    dx = L /n
    dy = L/m

    x = np.arange(-2,n+2) * dx
    y = np.arange(-2,m+2) * dy
    x, y =  np.meshgrid(x,y, indexing='ij')

    uc[0,:] = np.exp(-((x-.5)/.10)**2)  * np.exp(-((y-.5)/.10)**2)

    plt.subplot(132)
    uy = _slopes(uc, axis=2)
    plt.pcolormesh(uy[0,...])


    plt.subplot(131)
    ux = _slopes(uc, axis=1)
    plt.pcolormesh(ux[0,...])

    plt.subplot(133)
    plt.pcolormesh(uc[0,...])
    plt.show()

if __name__ == '__main__':
    pass
    test_tadmor_convergence(plot=True)
    # plot_advection2d()
    # test_slopes()
    # test_stagger_avg()
