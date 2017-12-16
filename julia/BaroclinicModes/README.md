# BaroclinicModes

This package implements the ENO-4 scheme for the four mode equations described in Khouider and Kacimi's papers.

## Numerical method

We follow the approach of Khouider and Moncrieff (2015), which uses a ENO scheme to estimate the derivatives.


    u_t + u'A u_x = F


do a weno-scheme with lax-friedrichs splitting.
    
    (u' A + alpha I ) u^+_y  + (u' A - alpha I)u^-_x = 0