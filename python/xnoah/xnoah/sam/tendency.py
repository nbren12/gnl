from . import regrid as rg


def _interface_velocities(u, v, blocks):
    uw = rg.staggered_to_left(u, blocks['x'], 'x')
    ue = rg.staggered_to_right(u, blocks['x'], 'x')
    vs = rg.staggered_to_left(v, blocks['y'], 'y')
    vn = rg.staggered_to_right(v, blocks['y'], 'y')
    return uw, ue, vs, vn


def advect_scalar(u, v, f, blocks):
    uw, ue, vs, vn = _interface_velocities(u, v, blocks)

    if 'x' in f.dims:
        w = rg.centered_to_left(f, blocks['x'], 'x')
        e = rg.centered_to_right(f, blocks['x'], 'x')
    else:
        w, e = f, f

    if 'y' in f.dims:
        s = rg.centered_to_left(f, blocks['y'], 'y')
        n = rg.centered_to_right(f, blocks['y'], 'y')
    else:
        s, n = f, f

    # average fluxes over the interface
    w, e = [rg.coarsen_dim(x, blocks['y'], 'y') for x in [uw * w, ue * e]]
    s, n = [rg.coarsen_dim(x, blocks['x'], 'x') for x in [vs * s, vn * n]]

    # compute surface intergral
    x = u['x']
    y = u['y']
    Lx = float(x[1] - x[0]) * blocks['x']
    Ly = float(y[1] - y[0]) * blocks['y']

    sfc = (w - e) * Ly + (s - n) * Lx

    tend = sfc / Lx / Ly
    tend = tend.assign_attrs(units=f.units.strip() + '/s')

    return tend
