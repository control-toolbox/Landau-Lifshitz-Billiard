import os
import nutopy as nt
import numpy as np

def hamiltonian(c, compile=True, display=False):
    
    if display:
        out=''
    else:
        out='> /dev/null 2>&1'
    
    if compile:
        # Compilation of the Hamiltonian and associated derivatives up to order 3
        os.system('python -m numpy.f2py -c hfun.f90       -m hfun       ' + out)
        os.system('python -m numpy.f2py -c hfun_d.f90     -m hfun_d     ' + out)
        os.system('python -m numpy.f2py -c hfun_d_d.f90   -m hfun_d_d   ' + out)
        os.system('python -m numpy.f2py -c hfun_d_d_d.f90 -m hfun_d_d_d ' + out)
        
    from hfun       import hfun       as hf
    from hfun_d     import hfun_d     as hf_d
    from hfun_d_d   import hfun_d_d   as hf_d_d
    from hfun_d_d_d import hfun_d_d_d as hf_d_d_d

    # Hamiltonian and derivatives: the signature has to fit nutopy package requirements
    # See https://ct.gitlabpages.inria.fr/nutopy/api/ocp.html
    #
    # Note that the second and third increments d2x and d3x are reversed between tapenade and nutopy
    #
    hfun   = lambda t, x, p                   : hf(x, p, c)
    dhfun  = lambda t, x, dx, p, dp           : hf_d(x, dx, p, dp, c)
    d2hfun = lambda t, x, dx, d2x, p, dp, d2p : hf_d_d(x, d2x, dx, p, d2p, dp, c)
    d3hfun = lambda t, x, dx, d2x, d3x, p, dp, d2p, d3p : hf_d_d_d(x, d3x, d2x, dx, p, d3p, d2p, dp, c)

    hfun   = nt.tools.tensorize(dhfun, d2hfun, d3hfun, tvars=(2, 3), full=True)(hfun)
    h      = nt.ocp.Hamiltonian(hfun)
    
    return h