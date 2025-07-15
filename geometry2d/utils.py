import numpy as np
import geometry2d.plottings

# Function to computes self-intersections 
# of a curve in a 2-dimensional space
def get_self_intersections(curve, *, modulo=None):
    n = curve[:,0].size
    intersections = []
    for i in range(n-3):
        A = curve[i,:]
        B = curve[i+1,:]
        for j in range(i+2,n-1):
            if modulo is not None:
                C = curve[j,:]   % modulo
                D = curve[j+1,:] % modulo
            else:
                C = curve[j,:]
                D = curve[j+1,:]
            # Matrice M : M z = b
            m11 = B[0] - A[0]
            m12 = C[0] - D[0]
            m21 = B[1] - A[1]
            m22 = C[1] - D[1]
            det = m11*m22-m12*m21
            if(np.abs(det)>1e-8):
                b1 = C[0] - A[0]
                b2 = C[1] - A[1]
                la = (m22*b1-m12*b2)/det
                mu = (m11*b2-m21*b1)/det
                if(la>=0. and la<=1. and mu>=0. and mu<=1.):
                    xx = {'i': i, 'j': j, \
                          'x': np.array(A + la * (B-A)), \
                          'la': la, 'mu': mu}
                    intersections.append(xx)
    return intersections

# function to plot the domain of strong current between two angles r1 and r2
def plot_2d_domain(fig, r1, r2, color):
    # from r to phi
    def phi(r):
        return r - np.pi/2.0
    ph1 = phi(r1)
    ph2 = phi(r2)
    # plot the domain: a 2d surface
    ax = fig.axes[0]
    xlims = ax.get_xlim()
    x  = np.linspace(xlims[0], xlims[1], 100)
    y1 = np.ones_like(x)*ph1
    y2 = np.ones_like(x)*ph2
    ax.fill_between(x, y1, y2, color=color, alpha=0.5)
    return fig  

def plot_3d_domain(fig, r1, r2, color, *, elevation, azimuth):
    #
    color = color
    alpha = 0.5
    # from r to phi
    def phi(r):
        return r - np.pi/2.0
    φ1 = phi(r1)
    φ2 = phi(r2)
    # plot the domain: a 2d surface between the parallel of latitude φ1 and φ2 on the sphere
    ax = fig.axes[0]
    #
    N = 100
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(φ1, φ2, N)
    x = np.outer(np.cos(u), np.cos(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.ones(np.size(u)), np.sin(v))
    #
    x_front = np.zeros((N, N))
    y_front = np.zeros((N, N))
    z_front = np.zeros((N, N))
    x_back  = np.zeros((N, N))
    y_back  = np.zeros((N, N))
    z_back  = np.zeros((N, N))
    cam = geometry2d.plottings.get_cam(elevation, azimuth, geometry2d.plottings.dist__)
    gap = 1e-1
    for i in range(N):
        for j in range(N):
            u = np.array([x[i,j], y[i,j], z[i,j]])
            v = np.array([cam[0], cam[1], cam[2]])
            ps = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
            #ps = x[i,j]*cam[0]+y[i,j]*cam[1]+z[i,j]*cam[2]
            
            if ps >= -gap:
                x_back[i,j] = x[i,j]
                y_back[i,j] = y[i,j]
                z_back[i,j] = z[i,j]
                x_front[i,j] = np.nan
                y_front[i,j] = np.nan
                z_front[i,j] = np.nan
            if ps <= gap:
                x_front[i,j] = x[i,j]
                y_front[i,j] = y[i,j]
                z_front[i,j] = z[i,j]
                x_back[i,j] = np.nan
                y_back[i,j] = np.nan
                z_back[i,j] = np.nan
    #
    ax.plot_surface(x_front, y_front, z_front, color=color, alpha=1) #, zorder=z_order_surface)
    ax.plot_surface(x_back, y_back, z_back, color=color, alpha=alpha) #, zorder=-z_order_surface)
    return fig