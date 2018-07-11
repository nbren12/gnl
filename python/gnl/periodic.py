import numpy as np
import copy
import scipy.fftpack as fft

class PeriodicVariable(object):
    """Convenience class for representing a variable with periodic BC in one direction"""
    def __init__(self, data, periodic_axis=0, dx=1.0):
        
        if  data.shape[periodic_axis] % 2 == 0:
            raise ValueError("Size of the periodic dimension must be odd")
            
        self._data = data
        self._periodic_axis=periodic_axis
        self._dx = dx
        
    @property
    def L(self):
        return self._dx * self.n
        
    @property
    def fk(self):
        return fft.fft(self._data, axis=self._periodic_axis)
    
    @fk.setter
    def fk(self, fk):
        data = fft.ifft(fk, axis=self._periodic_axis)
        
        if not np.allclose(np.imag(data), 0):
            raise ValueError("Output data is not real")
            
        self._data = np.real(data)
        
    @property
    def n(self):
        return self._data.shape[self._periodic_axis]
        
    @property
    def k(self):
        return 2*np.pi * fft.fftfreq(self.n, d=self._dx)
    
    @property
    def x(self):
         return np.arange(self.n) * self._dx
    
    def corr(self, other):
        dot = np.sum(other._data * self._data, axis=self._periodic_axis)
        return dot/self.L
    
    def diff(self, axis=None, order=1):
        
        if axis is None:
            axis=self._periodic_axis
        
        if axis == self._periodic_axis:
            # Take FFT derivative
            out = self.copy()
            out.fk = (1j * self.k)**order * self.fk
        else:
            raise NotImplementedError
            
        return out
        
    def copy(self):
        return copy.deepcopy(self)

def test_PeriodicVariable():
    
    pi = np.pi
    x = np.linspace(0, 2*pi, 102)[:-1]

    xp = PeriodicVariable(np.sin(2*x), dx=2*pi/101)
    np.testing.assert_allclose(xp.diff(order=1), 2*np.cos(2*x))
    
