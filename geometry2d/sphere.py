import geometry2d.plottings
import numpy as np

class Sphere():
    
    def __init__(self, wavefront, t_cut):
        
        self.wavefront     = wavefront
        self.t_cut         = t_cut
        self.problem       = wavefront.problem
        self.data          = self.problem.data
        self.data_name     = 'sphere'
        self.label_default = 'default'
        self.color         = 'blue'
        self.linewidth     = 0.5
        self.zorder        = geometry2d.plottings.z_order_ball
        
    def exists_from_label__(self, tf, label):
        exists = False
        if self.data is not None and self.data.contains(self.data_name):
            spheres = self.data.get(self.data_name) # list of spheres
            for s in spheres: # iterate over the spheres
                # a sphere is a dictionnary with keys: label, locus
                s['locus'] = geometry2d.problem.SphereLocus(s['locus'])
                if s['label'] == label and abs(s['locus'].tf - tf) < 1e-10:
                    exists = True
                    break
        return exists
    
    def exists_from_tf(self, tf):
        # check if there is at least one sphere at tf
        exists = False
        if self.data is not None and self.data.contains(self.data_name):
            spheres = self.data.get(self.data_name)
            for s in spheres:
                s['locus'] = geometry2d.problem.SphereLocus(s['locus'])
                if abs(s['locus'].tf - tf) < 1e-10:
                    exists = True
                    break
        return exists
    
    def get_from_label__(self, tf, label):
        if self.data is not None and self.data.contains(self.data_name):
            spheres = self.data.get(self.data_name)
            for s in spheres:
                s['locus'] = geometry2d.problem.SphereLocus(s['locus'])
                if s['label'] == label and abs(s['locus'].tf - tf) < 1e-10:
                    return s
        return None
    
    def get_from_tf(self, tf):
        # return all the spheres at tf
        spheres_at_tf = {}
        if self.data is not None and self.data.contains(self.data_name):
            spheres = self.data.get(self.data_name)
            for s in spheres:
                s['locus'] = geometry2d.problem.SphereLocus(s['locus'])
                if abs(s['locus'].tf - tf) < 1e-10:
                    # add the sphere to the dictionary
                    # the key is the label and the value is the locus
                    spheres_at_tf[s['label']] = s['locus']
        return spheres_at_tf
    
    def remove_from_label__(self, tf, label):
        if self.data is not None and self.data.contains(self.data_name):
            spheres = self.data.get(self.data_name)
            for s in spheres:
                s['locus'] = geometry2d.problem.SphereLocus(s['locus'])
                if s['label'] == label and abs(s['locus'].tf - tf) < 1e-10:
                    spheres.remove(s)
                    break
            self.data.update({self.data_name: spheres})
            
    def remove_from_tf(self, tf):
        if self.data is not None and self.data.contains(self.data_name):
            spheres = self.data.get(self.data_name)
            for s in spheres:
                s['locus'] = geometry2d.problem.SphereLocus(s['locus'])
                if abs(s['locus'].tf - tf) < 1e-10:
                    spheres.remove(s)
            self.data.update({self.data_name: spheres})
    
    def compute_sphere_locus__(self, wavefront_locus, tf):
        
        states = wavefront_locus.states
        alphas = wavefront_locus.alphas
        
        # compute the sphere
        states_sphere = np.zeros(states.shape)
        for i in range(len(states)):
            state = states[i]
            alpha = alphas[i]
            if tf < self.t_cut(alpha):
                states_sphere[i, :] = state
            else:
                states_sphere[i, :] = np.nan * np.ones(state.shape)
        
        # return the sphere
        return geometry2d.problem.SphereLocus((tf, states_sphere, alphas))
    
    def compute(self, tf, *, label=None, save=True, reset=False, load=True):
        
        if reset:
            if label is None:
                self.remove_from_tf(tf)
            else:
                self.remove_from_label__(tf, label)
        
        # if data is None then force save to False
        if self.data is None:
            save = False
            load = False
            
        if label==None and load and not reset:
            if self.exists_from_tf(tf):
                return self.get_from_tf(tf)
        elif not label==None and load and not reset:
            if self.exists_from_label__(tf, label):
                return self.get_from_label__(tf, label)
            
        # if here, then the spheres are not in the data
        # get the wavefronts from which we will compute the spheres
        if label==None:
            if self.wavefront.exists_from_tf(tf):
                wavefronts = self.wavefront.get_from_tf(tf)
            else:
                raise ValueError('There is no wavefront at tf.')
        else:
            if self.wavefront.exists_from_label__(tf, label):
                wavefront_locus = self.wavefront.get_from_label__(tf, label)
                wavefronts = {label: wavefront_locus}
            else:
                raise ValueError('There is no wavefront with label {} at tf.'.format(label))
            
        # compute the spheres
        spheres = []
        for label, wavefront_locus in wavefronts.items():
            # compute the sphere
            sphere_locus = self.compute_sphere_locus__(wavefront_locus, tf)
            # add the sphere to the dictionary
            spheres.append({'label': label, 'locus': sphere_locus})
            
        # save the spheres
        if save:
         
            # get the data
            if self.data.contains(self.data_name):
                data_spheres = self.data.get(self.data_name)
            else:
                data_spheres = []  
                
            # update the data
            for sphere in spheres:
                # remove the sphere if it already exists
                data_sphere = sphere.copy()
                data_sphere['locus'] = sphere['locus'].get_data()
                # add the sphere
                data_spheres.append(data_sphere)
                
            # save the data
            self.data.update({self.data_name: data_spheres})
            
        # return the spheres: if one sphere then return the sphere else return the array
        if len(spheres) == 1:
            return spheres[0]
        else:
            return spheres
        
    def plot_2d(self, sphere_loci, *, color=None, linewidth=None, zorder=None, figure=None,
                dpi=geometry2d.plottings.dpi__, figsize=geometry2d.plottings.figsize_2d__):
        
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
            
        # check if sphere_loci is a list
        if not isinstance(sphere_loci, list):
            sphere_loci = [sphere_loci]
            
        # iterate over the sphere_loci
        for w in sphere_loci:
            
            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = w['locus'].states[:, 0]
            θ = w['locus'].states[:, 1]
            φ = r - np.pi/2
            
            # plot the sphere_loci
            geometry2d.plottings.plot_2d(figure, θ,         φ, color=color, linewidth=linewidth, zorder=zorder)
            geometry2d.plottings.plot_2d(figure, θ+2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder)
            geometry2d.plottings.plot_2d(figure, θ-2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder)
            
        return figure
 
    def plot_3d(self, sphere_loci, *,
                elevation=geometry2d.plottings.elevation__,
                azimuth=geometry2d.plottings.azimuth__, 
                color=None, linewidth=None, zorder=None, figure=None,
                dpi=geometry2d.plottings.dpi__, figsize=geometry2d.plottings.figsize_3d__):
        
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
            
        # check if sphere_loci is a list
        if not isinstance(sphere_loci, list):
            sphere_loci = [sphere_loci]
            
        # iterate over the sphere_loci
        for w in sphere_loci:
            
            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = w['locus'].states[:, 0]
            θ = w['locus'].states[:, 1]
            φ = r - np.pi/2
            
            # plot the sphere_loci
            x, y, z = geometry2d.plottings.coord3d(θ, φ, self.problem.epsilon)
            geometry2d.plottings.plot_3d(figure, x, y, z, color=color, linewidth=linewidth, zorder=zorder)
            
        return figure
       
    def plot(self, tf, labels=None, *,
            view=geometry2d.plottings.Coords.SPHERE, 
            elevation=geometry2d.plottings.elevation__,
            azimuth=geometry2d.plottings.azimuth__, 
            color=None, linewidth=None, zorder=None, figure=None, dpi=None, figsize=None):
        
        # if there is no sphere at tf then raise an error
        if not self.exists_from_tf(tf):
            raise ValueError('There is no sphere at tf.')
        
        # get the spheres at tf
        spheres_at_tf = self.get_from_tf(tf)
        
        # initialize the parameters
        if labels is None:
            labels = ['left', 'right']
        elif not (isinstance(labels, list) or isinstance(labels, np.ndarray) or isinstance(labels, tuple)):
            labels = [labels]
            
        if dpi is None:
            dpi = geometry2d.plottings.dpi__
        if figsize is None:
            if view == geometry2d.plottings.Coords.SPHERE:
                figsize = geometry2d.plottings.figsize_3d__
            elif view == geometry2d.plottings.Coords.PLANE:
                figsize = geometry2d.plottings.figsize_2d__
                
        # check if there are both 'left' and 'right' in the labels
        left_right_in_labels_and_SPHERE = ('left' in labels) and ('right' in labels) #and \
                                            #(view == geometry2d.plottings.Coords.SPHERE)
        # if left_right_in_labels is True, then we concatenate the spheres before plotting
        if left_right_in_labels_and_SPHERE:
            # get the spheres at tf. Then, concatenate and plot
            wl = spheres_at_tf['left']
            wr = spheres_at_tf['right']
            r = np.concatenate((wr.states[:, 0], wl.states[:, 0], [wr.states[0, 0]]))
            if abs(wr.states[-1, 1] - wl.states[0, 1]) > np.pi:
                θ = np.concatenate((wr.states[:, 1], wl.states[:, 1]+2*np.pi, [wr.states[0, 1]]))
            else:
                θ = np.concatenate((wr.states[:, 1], wl.states[:, 1], [wr.states[0, 1]]))
            α = np.concatenate((wr.alphas, wl.alphas, [wr.alphas[0]]))
            sphere_locus = geometry2d.problem.SphereLocus((tf, np.array([r, θ]).T, α))
            # plot
            if view == geometry2d.plottings.Coords.SPHERE:
                figure = self.plot_3d([{'label':'default', 'locus':sphere_locus}], 
                                    elevation=elevation, azimuth=azimuth, 
                                    color=color, linewidth=linewidth, zorder=zorder, 
                                    figure=figure, dpi=dpi, figsize=figsize)
            elif view == geometry2d.plottings.Coords.PLANE:
                figure = self.plot_2d([{'label':'default', 'locus':sphere_locus}], 
                                    color=color, linewidth=linewidth, zorder=zorder, 
                                    figure=figure, dpi=dpi, figsize=figsize)
            else:
                raise ValueError('The view is not valid.')
                
        # iterates over the labels and create a list of spheres
        spheres_loci = []
        for label in labels:
            # do not append if label is 'left' or 'right' and left_right_in_labels is True
            if (left_right_in_labels_and_SPHERE) and ((label == 'left') or (label == 'right')):
                continue # this skip the rest of the loop and goes to the next iteration
            spheres_loci.append({'label':label, 'locus':spheres_at_tf[label]})
            
        # plot
        if view == geometry2d.plottings.Coords.SPHERE:
            figure = self.plot_3d(spheres_loci, 
                                   elevation=elevation, azimuth=azimuth, 
                                   color=color, linewidth=linewidth, zorder=zorder, 
                                   figure=figure, dpi=dpi, figsize=figsize)
        elif view == geometry2d.plottings.Coords.PLANE:
            figure = self.plot_2d(spheres_loci, 
                                   color=color, linewidth=linewidth, zorder=zorder, 
                                   figure=figure, dpi=dpi, figsize=figsize)
        else:
            raise ValueError('The view is not valid.')
        
        return figure      