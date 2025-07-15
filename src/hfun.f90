! Landau
subroutine hfun(x, p, c, h)

    double precision, intent(in)  :: x(2), p(2), c(4)
    double precision, intent(out) :: h

    ! local variables
    double precision :: r, th, pr, pth
    double precision :: m2, a, g1, g2, g3, cr, sr, ct, st, mu1, mu2

    r   = x(1)
    th  = x(2)

    pr  = p(1)
    pth = p(2)

    cr = cos(r)
    sr = sin(r)
    ct = cos(th)
    st = sin(th)

    g1 = c(1)
    g2 = c(2)
    g3 = c(3)
    a  = c(4)

    mu1 = a*g1*cr*sr+(g2-g3)*ct*st*sr-a*(g2*ct**2+g3*st**2)*cr*sr 
    mu2 = a*(g2-g3)*ct*st-g1*cr+(g3*st**2+g2*ct**2)*cr

    m2   = sin(r)**2

    h   = mu1*pr + mu2*pth + sqrt(pr**2 + pth**2/m2)

end subroutine hfun