import numpy as np

def test_convergence(f, nlist, plot=False, latex=True,
                     order_expected=2.0):

    nlist = np.array(nlist)
    err = np.array([f(n) for n in nlist.flat])
    p = np.polyfit(np.log(nlist), np.log(err), 1)
    rat = err[:-1]/err[1:]
    rat = list(rat) + ["-"]

    print("")
    print("Convergence test results")
    print("----------------------------")
    print("Grid Sizes: "+repr(nlist))
    print("Errors: "+repr(err))
    print("Ratio: "+repr(rat))
    print("Order of convergence:" +repr(-p[0]))
    print("")

    if latex:
        def format(it):
            if isinstance(it, str):
                return it
            elif isinstance(it, float):
                return "%.3e"%it
            else:
                return repr(it)
        print(r"n & $||err||_1$ & Ratio\\")
        print(r"\hline \\")
        for item in zip(nlist, err, rat):
            print(" & ".join(format(it) for it in item) + r"\\")

    if plot:
        import matplotlib.pyplot as plt
        plt.loglog(nlist, err)
        plt.title('Order of convergence p = %.2f'%p[0])
        plt.show()

    if  -p[0] < order_expected- .1:
        raise ValueError('Order of convergence (p={p})is less than 2'.format(p=-p[0]))
