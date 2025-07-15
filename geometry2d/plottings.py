from enum import Enum                     # for 2D vs 2D plots
import numpy as np                        # scientific computing tools

import matplotlib.pyplot as plt           # for plots
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D   # for 3D plots
plt.rcParams.update({"text.usetex":True, "font.family":"sans-serif", "font.sans-serif":["Helvetica"]}) # font properties
from matplotlib.colors import LightSource

from mpl_toolkits.mplot3d.proj3d import proj_transform
#from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

# Kind of coordinates
class Coords(Enum):
    PLANE=2   # 2D
    SPHERE=3  # 3D
    
# z orders
z_order_sphere     = 0;  zc = 100
z_order_conj_surf  = zc; zc = zc+1
z_order_axes       = zc; zc = zc+1
z_order_geodesics  = zc; zc = zc+1
z_order_wavefront  = zc; zc = zc+1
z_order_ball       = z_order_wavefront
z_order_conjugate  = zc; zc = zc+1
z_order_splitting  = zc; zc = zc+1
z_order_q0         = zc; zc = zc+1
delta_zo_back = 50

# Parameters for the 3D view
elevation__ = -10
azimuth__   = 20
dist__      = 10

# figure parameters
dpi__ = 200
figsize_2d__ = (3,3)
figsize_3d__ = (2,2)

#
alpha_sphere = 0.5

# 2D to 3D coordinates
def coord3d(theta, phi, epsilon):
    v = -theta
    u = phi
    coefs = (1., 1., epsilon)                   # Coefficients in (x/a)**2 + (y/b)**2 + (z/c)**2 = 1 
    rx, ry, rz = coefs                          # Radii corresponding to the coefficients
    x = rx * np.multiply(np.cos(u), np.cos(v))
    y = ry * np.multiply(np.cos(u), np.sin(v))
    z = rz * np.sin(u)
    return x, y, z

def get_cam(elev, azimuth, dist):
    ce   = np.cos(2*np.pi*elev/360)
    se   = np.sin(2*np.pi*elev/360)
    ca   = np.cos(2*np.pi*azimuth/360)
    sa   = np.sin(2*np.pi*azimuth/360)
    cam  = np.array([ dist*ca*ce, dist*sa*ce, dist*se])
    return cam
    
def decorate_2d(ax, q0=None):
    
    x   = [-np.pi, np.pi, np.pi, -np.pi]
    y   = [np.pi/2, np.pi/2, np.pi/2+1, np.pi/2+1]
    ax.fill(x, y, color=(0.95, 0.95, 0.95))
    y   = [-(np.pi/2+1), -(np.pi/2+1), -np.pi/2, -np.pi/2]
    ax.fill(x, y, color=(0.95, 0.95, 0.95))
    ax.set_xlabel(r'$\theta$', fontsize=10)
    ax.set_ylabel(r'$\varphi$', fontsize=10)
    ax.axvline(0, color='k', linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
    ax.axhline(0, color='k', linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
    ax.axhline(-np.pi/2, color='k', linewidth=0.5, zorder=z_order_axes)
    ax.axhline( np.pi/2, color='k', linewidth=0.5, zorder=z_order_axes)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi/2, np.pi/2)
    ax.set_aspect('equal', 'box')

    if not(q0 is None):
        r = q0[0]
        θ = q0[1]
        φ = r - np.pi/2
        ax.plot(θ, φ, marker="o", markersize=2, markeredgecolor="black", markerfacecolor="black", zorder=z_order_q0)  
        ax.plot(θ+2*np.pi, φ, marker="o", markersize=2, markeredgecolor="black", markerfacecolor="black", zorder=z_order_q0)  
        ax.plot(θ-2*np.pi, φ, marker="o", markersize=2, markeredgecolor="black", markerfacecolor="black", zorder=z_order_q0)  
    
def init_figure_2d(q0=None, *, dpi=dpi__, figsize=figsize_2d__):
    
    fig = Figure(dpi=dpi)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.patch.set_alpha(0.0)
    ax  = fig.add_subplot(111)
    ax.patch.set_alpha(0.0)
    decorate_2d(ax, q0)
    return fig

def plot_2d(fig, θ, φ, *, color='b', linewidth=0.5, zorder=1, linestyle='solid', show_arrow=False):
    ax = fig.axes[0]
    ax.plot(θ, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
    ax.plot(θ+2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
    ax.plot(θ-2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
    if show_arrow:
        adx = len(θ) // 5
        arrow = θ[adx+1], φ[adx+1], θ[adx+1]-θ[adx], φ[adx+1]-φ[adx]
        ax.arrow(*arrow, color=color, shape='full', lw=0, length_includes_head=True, head_width=0.08) 


def orthogonal_basis(v):
    """
    Trouve une base du plan orthogonal au vecteur v dans R^3.
    
    Paramètre:
    v : array-like de taille (3,), vecteur normal au plan.
    
    Retourne:
    (u1, u2) : deux vecteurs formant une base du plan orthogonal à v.
    """
    a, b, c = v

    # Choix de deux vecteurs orthogonaux à v
    if c != 0:
        u1 = np.array([1, 0, -a/c])
        u2 = np.array([0, 1, -b/c])
    elif b != 0:
        u1 = np.array([1, -a/b, 0])
        u2 = np.array([0, 0, 1])
    else:  # Cas où v est de la forme (a,0,0), son plan orthogonal est le plan yz
        u1 = np.array([0, 1, 0])
        u2 = np.array([0, 0, 1])

    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    
    return u1, u2

def determinant_of_vectors(v1, v2, v3):
    """
    Calcule le déterminant formé par trois vecteurs en R^3.
    
    Paramètres:
    v1, v2, v3 : array-like (3,) -> Trois vecteurs de R^3
    
    Retourne:
    det : float -> Déterminant de la matrice formée par ces vecteurs
    """
    A = np.column_stack((v1, v2, v3))  # Construction de la matrice avec les vecteurs en colonnes
    det = np.linalg.det(A)  # Calcul du déterminant
    return det

def plot3d_axis_on_sphere(ax, x, y, z, r, elevation, azimuth, color, linewidth, linestyle='solid', zorder=1):
    N = len(x)
    i = 0
    j = 1

    cam = get_cam(elevation, azimuth, dist__)

    # plan orthogonal
    u1, u2 = orthogonal_basis(cam)

    # déterminant
    # ps = (cam[0]-x[0])*cam[0]+(cam[1]-y[0])*cam[1]+(cam[2]-z[0])*cam[2] # ps < 0: behind sphere
    Point = np.array([x[0], y[0], z[0]])
    ps = determinant_of_vectors(u1, u2, Point)

    # projection sur le plan tangent et distance au centre
    dp = (np.dot(Point, u1)**2 + np.dot(Point, u2)**2) - r**2
    
    # distance
    dd = (x[0]**2+y[0]**2+z[0]**2)-r**2 # dd < 0: inside sphere

    STATE = 1 # visible
    if dd<0:
        STATE = 0
    else:
        if ps<0 and dp<0:
            STATE = 0
    
    while i<N-1:
        dd_j = (x[j]**2+y[j]**2+z[j]**2)-r**2
        #ps_j = (cam[0]-x[j])*cam[0]+(cam[1]-y[j])*cam[1]+(cam[2]-z[j])*cam[2]
        Point = np.array([x[j], y[j], z[j]])
        ps_j = determinant_of_vectors(u1, u2, Point)
        dp_j = (np.dot(Point, u1)**2 + np.dot(Point, u2)**2) - r**2
        STATE_j = 1 # visible
        if dd_j<0:
            STATE_j = 0
        else:
            if ps_j<0 and dp_j<0:
                STATE_j = 0
        if (STATE!=STATE_j) or (j==N-1):
            if STATE==0:
                ls = linestyle
                lw = linewidth/3.0
                al = 0.5
                zo = -zorder
            else:
                ls = linestyle
                lw = linewidth
                al = 1.0
                zo = zorder
            ax.plot(x[i:j+1], y[i:j+1], z[i:j+1], color=color, \
                    linewidth=lw, linestyle=ls, zorder=zo, alpha=al)
            i = j
            STATE = STATE_j
        j = j+1

def plot3d(ax, x, y, z, elevation, azimuth, color, linewidth, linestyle='solid', zorder=1):
    N = len(x)
    i = 0
    j = 1

    cam = get_cam(elevation, azimuth, dist__)
    ps = x[0]*cam[0]+y[0]*cam[1]+z[0]*cam[2]
    while i<N-1:
        ps_j = x[j]*cam[0]+y[j]*cam[1]+z[j]*cam[2]
        if (ps*ps_j<0) or (j==N-1):
            if ps>0:
                ls = linestyle
                lw = linewidth/3.0
                al = 0.5
                zo = -zorder
            else:
                ls = linestyle
                lw = linewidth
                al = 1.0
                zo = zorder
            ax.plot(x[i:j+1], y[i:j+1], z[i:j+1], color=color, \
                    linewidth=lw, linestyle=ls, zorder=zo, alpha=al)
            i = j
            ps = ps_j
        j = j+1

def decorate_3d(ax, epsilon, q0=None, elevation=elevation__, azimuth=azimuth__):
    
    ax.set_axis_off()
    coefs = (1., 1., epsilon)              # Coefficients in (x/a)**2 + (y/b)**2 + (z/c)**2 = 1 
    rx, ry, rz = coefs                     # Radii corresponding to the coefficients

    # Set of all spherical angles:
    v = np.linspace(-np.pi, np.pi, 100)
    u = np.linspace(-np.pi/2, np.pi/2, 100)

    # Cartesian coordinates that correspond to the spherical angles
    x = rx * np.outer(np.cos(u), np.cos(v))
    y = ry * np.outer(np.cos(u), np.sin(v))
    z = rz * np.outer(np.sin(u), np.ones_like(v))

    # Landscape
    ZSPHERE = z

    # this is used to set the graph color to blue
    blue = np.array([1, 1, 1])
    rgb = np.tile(blue, (ZSPHERE.shape[0], ZSPHERE.shape[1], 1))

    ls = LightSource(azdeg=azimuth, altdeg=elevation, hsv_min_val=0, hsv_max_val=1, hsv_min_sat=1, hsv_max_sat=0)
    illuminated_surface = ls.shade_rgb(rgb, ZSPHERE)

    # Plot:
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, \
                    color='lightblue', 
                    linewidth=0, facecolors=illuminated_surface,
                    alpha=alpha_sphere, antialiased=True, edgecolor='none', zorder=z_order_sphere)

    # initial point
    if not(q0 is None):
        r = q0[0]
        θ = q0[1]
        φ = r - np.pi/2
        x, y, z = coord3d(θ, φ, epsilon)
        cam = get_cam(elevation, azimuth, dist__)
        ps = x*cam[0]+y*cam[1]+z*cam[2]
        if ps>0: # back
            zo = -z_order_q0 # - delta_zo_back
            al = 0.5
        else:
            zo = z_order_q0
            al = 1.0
        ax.plot(x, y, z, marker="o", markersize=3, alpha=al, \
                markeredgecolor="black", markerfacecolor="black", zorder=zo)

    # 
    size_pole = np.pi/80 
    
    # North pole
    N = 20
    r = (np.pi-size_pole)*np.ones(N)
    θ = np.linspace(0, 2*np.pi, N)
    φ = r - np.pi/2
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, elevation, azimuth, color="black", linewidth=0.5, linestyle="dashed", zorder=z_order_axes)

    # South pole
    N = 20
    r = (0+size_pole)*np.ones(N)
    θ = np.linspace(0, 2*np.pi, N)
    φ = r - np.pi/2
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, elevation, azimuth, color="black", linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
    
    # add one meridian
    N = 100
    if not(q0 is None):
        θ = q0[1]*np.ones(N)
    else:
        θ = 0*np.ones(N)
    r = np.linspace(0+size_pole, np.pi-size_pole, N)
    φ = r - np.pi/2
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, elevation, azimuth, color="black", linewidth=0.5, linestyle="dashed", zorder=z_order_axes)

    r = np.linspace(np.pi+size_pole, 2*np.pi-size_pole, N)
    φ = r - np.pi/2
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, elevation, azimuth, color="black", linewidth=0.5, linestyle="dashed", zorder=z_order_axes)

    # add one meridian
    N = 100
    θ = np.pi/2*np.ones(N)
    
    r = np.linspace(0+size_pole, np.pi-size_pole, N)
    φ = r - np.pi/2
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, elevation, azimuth, color="black", linewidth=0.5, linestyle="dashed", zorder=z_order_axes)

    r = np.linspace(np.pi+size_pole, 2*np.pi-size_pole, N)
    φ = r - np.pi/2
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, elevation, azimuth, color="black", linewidth=0.5, linestyle="dashed", zorder=z_order_axes)
    
    # add equator
    N = 100
    θ = np.linspace(0, 2*np.pi, N)
    φ = 0*np.ones(N)
    x, y, z = coord3d(θ, φ, epsilon)
    plot3d(ax, x, y, z, elevation, azimuth, color="black", \
            linewidth=0.5, linestyle="dashed", zorder=z_order_axes)

    #
    ratio_axis = 2
    
    # axis
    N = 100
    x = np.linspace(-ratio_axis*rx, ratio_axis*rx, N)
    y = np.zeros(N)
    z = np.zeros(N)
    #plot3d_axis_on_sphere(ax, x, y, z, rx, elevation, azimuth, color="red", linewidth=0.5, zorder=z_order_axes)
    ax.plot(x, y, z, color='gray', linewidth=0.5, zorder=z_order_axes, alpha=0.5)

    # axis
    N = 100
    y = np.linspace(-ratio_axis*ry, ratio_axis*ry, N)
    x = np.zeros(N)
    z = np.zeros(N)
    #plot3d_axis_on_sphere(ax, x, y, z, rx, elevation, azimuth, color="red", linewidth=0.5, zorder=z_order_axes)
    ax.plot(x, y, z, color='gray', linewidth=0.5, zorder=z_order_axes, alpha=0.5)

    # axis
    N = 100
    z = np.linspace(-ratio_axis*rz, ratio_axis*rz, N)
    y = np.zeros(N)
    x = np.zeros(N)
    #plot3d_axis_on_sphere(ax, x, y, z, rx, elevation, azimuth, color="red", linewidth=0.5, zorder=z_order_axes)
    ax.plot(x, y, z, color='gray', linewidth=0.5, zorder=z_order_axes, alpha=0.5)

    # Adjustment of the axes, so that they all have the same span:
    max_radius = max(rx, ry, rz)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

    ax.view_init(elev=elevation, azim=azimuth) # Reproduce view
    #ax.dist = dist__
    ax.set_box_aspect(None, zoom=dist__)
    
    ax.set_xlim(np.array([-rx,rx])*.67)
    ax.set_ylim(np.array([-ry,ry])*.67)
    ax.set_zlim(np.array([-rz,rz])*.67)

    # 
    ax.set_aspect('equal', 'box')   

def init_figure_3d(epsilon, q0=None, elevation=elevation__, azimuth=azimuth__, *, 
                   dpi=dpi__, figsize=figsize_3d__):
    
    fig = Figure(dpi=dpi)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)
    plt.tight_layout()
    decorate_3d(ax, epsilon, q0, elevation, azimuth)
    return fig

class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)

def plot_3d(fig, x, y, z, *, color='b', linewidth=1, zorder=1, linestyle='solid', show_arrow=False):
    #
    ax = fig.axes[0]
    # get azimuth and elevation
    azimuth   = ax.azim
    elevation = ax.elev
    #
    N = len(x)
    i = 0
    j = 1
    #
    cam = get_cam(elevation, azimuth, dist__)
    ps = x[0]*cam[0]+y[0]*cam[1]+z[0]*cam[2]
    #
    while i<N-1:
        if np.isnan(x[i]):
            i = i+1
        else:
            if np.isnan(x[j]):
                ps_j = ps
            else:
                ps_j = x[j]*cam[0]+y[j]*cam[1]+z[j]*cam[2]
            if (ps*ps_j<0) or (j==N-1):
                if ps>0:
                    ls = linestyle #'solid'
                    lw = linewidth/3.0
                    al = 0.5
                    zo = -zorder
                else:
                    ls = linestyle #'solid'
                    lw = linewidth
                    al = 1.0
                    zo = zorder
                # we do not plot nan at the end of the array
                k = j
                while np.isnan(x[k]) or i==k:
                    k = k-1
                if k>i:
                    ax.plot(x[i:k+1], y[i:k+1], z[i:k+1], color=color, \
                                linewidth=lw, linestyle=ls, zorder=zo, alpha=al)
                i = j
            ps = ps_j
        j = j+1
    #
    if show_arrow:
        adx = len(x) // 5
        arrow = x[adx+1], y[adx+1], z[adx+1], x[adx+1]-x[adx], y[adx+1]-y[adx], z[adx+1]-z[adx]
        ps = x[adx]*cam[0]+y[adx]*cam[1]+z[adx]*cam[2]
        if ps>0: # back
            zo = -zorder # - delta_zo_back
            al = 0.2
            ms = 1
        else:
            zo = zorder
            al = 1.0
            ms = 5
        ax.arrow3D(*arrow, color=color, mutation_scale=ms, arrowstyle='-|>', zorder=zo, alpha=al) 
        

# get surface coordinates from a closed curve defined by spherical coordinates
def surface_from_spherical_curve(x, y, epsilon):
    N = 100
    xmin = np.min(x)
    xmax = np.max(x)
    
    X = np.zeros((N, N))
    Y = np.zeros((N, N))
    
    xs = np.linspace(xmin, xmax, N)
    for i in range(N):
        x_current = xs[i]
        # find the two intersections of the curve with x_current
        ii  = np.argwhere(np.multiply(x[1:]-x_current, \
                                      x[0:-1]-x_current)<=0)
        #
        k   = ii[0][0]
        xk  = x[k]
        xkp = x[k+1]
        λ   = (x_current-xk)/(xkp-xk)
        y1  = y[k]+λ*(y[k+1]-y[k])
        #
        k   = ii[1][0]
        xk  = x[k]
        xkp = x[k+1]
        if abs(xkp-xk)>1e-12:
            λ = (x_current-xk)/(xkp-xk)
        else:
            λ = 0
        y2  = y[k]+λ*(y[k+1]-y[k])
        #
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        ys = np.linspace(ymin, ymax, N)
        X[:, i] = x_current*np.ones(N)
        Y[:, i] = ys
    
    # cartesian
    XX = np.zeros((N, N))
    YY = np.zeros((N, N))
    ZZ = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            XX[i,j], YY[i,j], ZZ[i,j] = coord3d(X[i,j], Y[i,j], \
                                                epsilon)

    return XX, YY, ZZ
        
def plot_surface(fig, X, Y, Z, *, color='b', alpha=0.5, zorder=1):
    #
    ax = fig.axes[0]
    ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, \
                color=color, alpha=alpha, antialiased=True, zorder=zorder)