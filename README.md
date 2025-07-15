# Landau-Lifshitz-Billiard

Codes for the article [The Zermelo Navigation Problem on the 2-Sphere of Revolution: An Optimal Control Perspective with Applications to Micromagnetism](https://hal.science/hal-04996987), from Bernard Bonnard, Olivier Cots and Yannick Privat.

The code in this repository is in Python and is based upon [nutopy](https://ct.gitlabpages.inria.fr/nutopy). It provides tools to compute objects from Riemannian geometry and its extensions. 

The directory `geometry2d` contains generic functions to compute on the 2-sphere: geodesics, conjugate locus, wavefronts, spheres. It contains also the necessary functions to plot.

The directoy `src` contains the Hamiltonian associated to the Landau-Lifshitz equation (`hfun.f90`) and its derivatives. The files for the derivatives are computed thanks to the Tapenade software, see `differentiate.sh`. The Hamiltonian and its derivatives are written in Fortran and are wrapped in Python thanks to the file `wrapper.py`. 

The notebook `landau-finsler.ipynb` compute the optimal synthesis of the article in the Finsler case and the notebook `landau-billiard.ipynb` is used to illustrate the Landau-Lifshitz billiard phenomenom.
