import numpy as np
import pandas as pd
import netCDF4 as nc

from numpy.fft import fft, ifft, fftshift, fftfreq
from numpy import percentile
from numpy.random import rand, randn

from scipy.integrate import quadrature
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
