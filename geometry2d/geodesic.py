import numpy as np   # scientific computing tools
import geometry2d.plottings
import scipy as sc  # for integration with event detection
import geometry2d.errors

class Geodesic():
    
    def __init__(self, problem):
        self.problem = problem
        self.color   = 'b'
        self.linewidth = 0.5
        self.zorder = geometry2d.plottings.z_order_geodesics

    def compute(self, t, α0):
        return np.array(self.problem.geodesic(t, α0))
    
    # Function to plot one given geodesic or a list of geodesics
    def plot_2d(self, geodesics, *, color=None, linewidth=None, zorder=None, linestyle='solid', figure=None, 
                dpi=geometry2d.plottings.dpi__, figsize=geometry2d.plottings.figsize_2d__, show_arrow=True):
        
        # initialize the parameters
        if color is None:
            color = self.color
        if linewidth is None:
            linewidth = self.linewidth
        if zorder is None:
            zorder = self.zorder
        
        # initialize the figure
        if figure is None:
            figure = geometry2d.plottings.init_figure_2d(self.problem.initial_point, dpi=dpi, figsize=figsize)

        # check if geodesics is a list
        if not isinstance(geodesics, list):
            geodesics = [geodesics]

        # iterate over the geodesics
        for q in geodesics:

            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = q[:, 0]
            θ = q[:, 1]
            φ = r - np.pi/2
            
            # plot the geodesic
            geometry2d.plottings.plot_2d(figure, θ,         φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle, show_arrow=show_arrow)
            geometry2d.plottings.plot_2d(figure, θ+2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle, show_arrow=show_arrow)
            geometry2d.plottings.plot_2d(figure, θ-2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle, show_arrow=show_arrow)
        
        return figure
            
    # Function to plot one given geodesic or a list of geodesics
    def plot_3d(self, geodesics, *, 
                elevation=geometry2d.plottings.elevation__,
                azimuth=geometry2d.plottings.azimuth__,
                color=None, linewidth=None, zorder=None, linestyle='solid', figure=None, 
                dpi=geometry2d.plottings.dpi__, figsize=geometry2d.plottings.figsize_3d__, show_arrow=True):
        
        # initialize the parameters
        if color is None:
            color = self.color
        if linewidth is None:
            linewidth = self.linewidth
        if zorder is None:
            zorder = self.zorder
        
        # initialize the figure
        if figure is None:
            figure = geometry2d.plottings.init_figure_3d(self.problem.epsilon, 
                                                    self.problem.initial_point,
                                                    elevation,
                                                    azimuth, dpi=dpi, figsize=figsize)

        # check if geodesics is a list
        if not isinstance(geodesics, list):
            geodesics = [geodesics]
            
        # iterate over the geodesics
        for q in geodesics:

            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = q[:, 0]
            θ = q[:, 1]
            φ = r - np.pi/2
            
            #
            x, y, z = geometry2d.plottings.coord3d(θ, φ, self.problem.epsilon)
            geometry2d.plottings.plot_3d(figure, x, y, z, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle, show_arrow=show_arrow)
            
        return figure
    
    def return_to_equator__(self, α0, *, first=True):

        #
        def Hvec(t, z):
            return self.problem.Hamiltonian.vec(t, z[0:2], z[2:4])
        #
        # equator: r = π/2
        v0 = Hvec(0, self.problem.initial_cotangent_point(α0))
        s  = np.sign(v0[0])
        if first:
            s = -s
        if s == 0:
            s = np.sign(v0[1])
        def hit_equator(t, z):
            if np.abs(z[0] - np.pi) <= 1e-4:
                return z[0] - (np.pi-1e-5)
            elif np.abs(z[0] - 0) <= 1e-4:
                return z[0] - (1e-5)
            elif np.abs(z[1] - 5*np.pi) <= 1e-1:
                return z[1] - 5*np.pi
            elif np.abs(z[1] - (-5*np.pi)) <= 1e-1:
                return -(z[1] - (-5*np.pi))
            # if np.abs(α0 % (2*np.pi) - np.pi/2) <= 1e-8: # going up
            #     return z[0] - (np.pi-1e-3) # we stop before north/south pole
            # elif np.abs(α0 % (2*np.pi) - 3*np.pi/2) <= 1e-8: # going down
            #     return z[0] - (1e-3) # we stop before north/south pole
            # elif np.abs(α0 % (2*np.pi) - 0) <= 1e-8: # going right
            #     return z[1] - np.pi # half turn
            # elif np.abs(α0 % (2*np.pi) - np.pi) <= 1e-8: # going left
            #     return -(z[1] - (-np.pi)) # half turn
            else:
                if t > 0:
                    return z[0] - np.pi/2 
                else:
                    return s
    
        hit_equator.terminal  = True
        hit_equator.direction = s

        z0 = self.problem.initial_cotangent_point(α0)
        sol = sc.integrate.solve_ivp(Hvec, [0, 5.0*np.pi], z0, events=hit_equator, dense_output=True)

        #
        if len(sol.t_events[0]) > 0:
            time = sol.t_events[0][0]
            # improve accuracy of time
            # ti = 0.99*time
            # qi, pi = self.problem.extremal(0, z0[0:2], z0[2:4], ti)
            # def myfun(t):
            #     qf, pf = self.problem.extremal(ti, qi, pi, t)
            #     return hit_equator(t, qf)
            # root = sc.optimize.brentq(myfun, 0.99*time, 1.01*time)
            # time = root
        else:
            time = sol.t[-1]

        return time
     
    # compute the time to return to the equator
    def return_to_equator(self, α0):
        return self.return_to_equator__(α0, first=False)
    
    def first_return_to_equator(self, α0):
        return self.return_to_equator__(α0, first=True)
    
    def plot(self, *, alphas=None, N=None, tf=None, length=1.0, 
             view=geometry2d.plottings.Coords.SPHERE, 
             azimuth=geometry2d.plottings.azimuth__, 
             elevation=geometry2d.plottings.elevation__, 
             color=None, linewidth=None, zorder=None, linestyle='solid',
             figure=None, dpi=None, figsize=None, force=False, show_arrow=True):
        
        # error if alphas is not None and N is not None
        if (not alphas is None) and (not N is None):
            raise geometry2d.errors.ArgumentValueError("alphas and N cannot be both not None")
        
        # if alphas is None and N is None, then N = 0
        if (alphas is None) and (N is None):
            N = 0
        elif (not alphas is None) and (N is None):
            N = len(alphas)
        elif (alphas is None) and (not N is None):
            αs1 = np.linspace(0*np.pi/2, 1*np.pi/2, (N//4)+1)
            αs2 = np.linspace(1*np.pi/2, 2*np.pi/2, (N//4)+1)
            αs3 = np.linspace(2*np.pi/2, 3*np.pi/2, (N//4)+1)
            αs4 = np.linspace(3*np.pi/2, 4*np.pi/2, (N//4)+1)
            alphas = np.concatenate(np.array([αs1[0:-1], αs2[0:-1], αs3[0:-1], αs4[0:-1]]))
        
        # initialize the parameters
        if color is None:
            color = self.color
        if linewidth is None:
            linewidth = self.linewidth
        if zorder is None:
            zorder = self.zorder
        if dpi is None:
            dpi = geometry2d.plottings.dpi__
        if figsize is None:
            if view == geometry2d.plottings.Coords.SPHERE:
                figsize = geometry2d.plottings.figsize_3d__
            elif view == geometry2d.plottings.Coords.PLANE:
                figsize = geometry2d.plottings.figsize_2d__
        
        # tf is either a fixed value or a function of α0
        if tf is None:
            tf_fun = self.return_to_equator
        # check if tf is a function
        elif not callable(tf):
            tf_fun = lambda α0: tf
        else:
            tf_fun = tf
            
        # if N = 0, then return the figure
        if N == 0:
            if view == geometry2d.plottings.Coords.SPHERE:
                return self.plot_3d([], elevation=elevation, azimuth=azimuth,
                    color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle,
                    figure=figure, dpi=dpi, figsize=figsize, show_arrow=show_arrow)
            elif view == geometry2d.plottings.Coords.PLANE:
                return self.plot_2d([], color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle,
                    figure=figure, dpi=dpi, figsize=figsize, show_arrow=show_arrow) 
        
        # computation of the geodesics
        geodesics = {}
        list_state = list([])
        list_time  = list([])
        list_α0    = list([])
        for α0 in alphas:
            if  (np.abs(α0 % (2*np.pi) - 0*np.pi/2) > 1e-8) and     \
                (np.abs(α0 % (2*np.pi) - 1*np.pi/2) > 1e-8) and     \
                (np.abs(α0 % (2*np.pi) - 2*np.pi/2) > 1e-8) and     \
                (np.abs(α0 % (2*np.pi) - 3*np.pi/2) > 1e-8) or force:
                time = length*tf_fun(α0)
                q = self.compute(time, α0)
                list_state.append(q)
                list_time.append(time)
                list_α0.append(time)
                geodesics['state']  = list_state
                geodesics['time']   = list_time
                geodesics['α0']     = list_α0
                
        list_state  = geodesics['state']
        list_time   = geodesics['time']
        list_α0     = geodesics['α0']
        
        #
        list_geodesics_to_plot = list_state

        if view == geometry2d.plottings.Coords.SPHERE:
            return self.plot_3d(list_geodesics_to_plot, elevation=elevation, azimuth=azimuth, 
                         color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle,
                         figure=figure, dpi=dpi, figsize=figsize, show_arrow=show_arrow)
        elif view == geometry2d.plottings.Coords.PLANE:
            return self.plot_2d(list_geodesics_to_plot, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle,
                                figure=figure, dpi=dpi, figsize=figsize, show_arrow=show_arrow)