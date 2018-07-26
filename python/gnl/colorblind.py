def colorblind():
    """Set colorblind friendly options for all installed plotting libraries"""
    for fun in [colorblind_matplotlib, colorblind_holoviews]:
        try:
            fun()
        except ImportError:
            pass


def colorblind_matplotlib():
    """Use colorblind pallette"""
    import matplotlib.pyplot as plt
    plt.style.use('tableau-colorblind10')


def colorblind_holoviews():
    """Use colorblind pallette"""
    import holoviews as hv
    hv.opts({'Curve': {'style': dict(color=hv.Cycle('Colorblind'))}})
