from functools import partial
import xarray as xr
import matplotlib.pyplot as plt
from xnoah.xarray import coarsen

def plot_2d(cld):
    fig, ax  = plt.subplots(figsize=(6,3))


    im= ax.pcolormesh(cld.x/1e6, cld.y/1e6, cld.values, cmap='Blues_r')
    ax.set_aspect(1.0)
    plt.colorbar(im)

    plt.xlabel('x [1000 km]')
    plt.ylabel('y [1000 km]')
    plt.title('Cloud Fraction')
    plt.savefig("cld.png", dpi=150)

path="/home/disk/eos13/guest/SAM6.10.6_NG"
file="NG_5120x2560x34_4km_10s_QOBS_EQX_1280_0000858600.2Dcom_001.nc"


f = xr.open_dataset(f"{path}/OUT_2D/{file}")

cld = coarsen(f.CLDC, x=5, y=5)
print("Plotting output")
plot_2d(cld.isel(time=0))
