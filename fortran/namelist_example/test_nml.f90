program test_nml
  implicit none

  real :: a, b, c

  namelist /PARAM/ a,b,c

  a = -1
  b = -1
  c = -1

  ! write(unit=10, nml=PARAM)
  read(unit=10, nml=PARAM)

  print *, a, b, c

end program test_nml
