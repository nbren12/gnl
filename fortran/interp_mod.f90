module interp_mod
  use state_mod, only: dp
contains

  !> Fifth order weno interpolation routine
  !! @returns interpolant to the right of the center point
  function weno5(phi)
    real(dp), intent(in) :: phi(-2:2)
    real(dp) :: weno5, p0, p1, p2, beta0, beta1, beta2,&
         alpha0, alpha1, alpha2, alpha_sum, w0, w1, w2

    p0 = (1.0/3.0) * phi(-2)  - (7.0/6.0)*phi(-1) + (11.0/6.0)*phi(0)
    p1 = (-1.0/6.0) * phi(-1) + (5.0/6.0)*phi(0) + (1.0/3.0)*phi(1)
    p2 = (1.0/3.0) * phi(0) + (5.0/6.0)*phi(1) - (1.0/6.0)*phi(2)

    beta2 = 13.0/12.0 * (phi(0) - 2.0 * phi(1) + phi(2))**2.0 + 0.25 * (3.0 * phi(0) - 4.0 * phi(1) + phi(2))**2.0
    beta1 = 13.0/12.0 * (phi(-1) - 2.0 * phi(0) + phi(1))**2.0 + 0.25 * (phi(-1) - phi(1))**2.0
    beta0 = 13.0/12.0 * (phi(-2) - 2.0 * phi(-1) + phi(0))**2.0 + 0.25 * (phi(-2) - 4.0 * phi(-1) + 3.0 * phi(0))**2.0

    alpha0 = 0.1/(beta0 + 1e-10)**2.0
    alpha1 = 0.6/(beta1 + 1e-10)**2.0
    alpha2 = 0.3/(beta2 + 1e-10)**2.0

    alpha_sum = alpha0 + alpha1 + alpha2

    w0 = alpha0/alpha_sum
    w1 = alpha1/alpha_sum
    w2 = alpha2/alpha_sum

    weno5 =  w0 * p0 + w1 * p1 + w2 * p2

  end function weno5

  function  sym4(phi)
    real(dp), intent(in) :: phi(4)
    real(dp) :: sym4

    sym4 = 7.0/12.0 * (phi(2) + phi(3)) - 1.0/12.0 * (phi(1) + phi(4) )

  end function sym4

end module interp_mod
