import geometry2d.plottings
import geometry2d.utils
import numpy as np

class WaveFront():
    
    def __init__(self, problem):
        
        self.problem = problem
        self.data    = problem.data
        self.data_name = 'wavefront'
        self.label_default = 'default'
        self.color = 'g'
        self.linewidth = 0.5
        self.zorder = geometry2d.plottings.z_order_wavefront
        
    def exists_from_label__(self, tf, label):
        exists = False
        if self.data is not None and self.data.contains(self.data_name):
            wavefronts = self.data.get(self.data_name) # list of wavefronts
            for w in wavefronts: # iterate over the wavefronts
                # a wavefront is a dictionnary with keys: label, locus
                w['locus'] = geometry2d.problem.WaveFrontLocus(w['locus'])
                if w['label'] == label and abs(w['locus'].tf - tf) < 1e-10:
                    exists = True
                    break
        return exists
    
    def exists_from_tf(self, tf):
        # check if there is at least one wavefront at tf
        exists = False
        if self.data is not None and self.data.contains(self.data_name):
            wavefronts = self.data.get(self.data_name)
            for w in wavefronts:
                w['locus'] = geometry2d.problem.WaveFrontLocus(w['locus'])
                if abs(w['locus'].tf - tf) < 1e-10:
                    exists = True
                    break
        return exists
    
    def get_from_label__(self, tf, label):
        if self.data is not None and self.data.contains(self.data_name):
            wavefronts = self.data.get(self.data_name)
            for w in wavefronts:
                w['locus'] = geometry2d.problem.WaveFrontLocus(w['locus'])
                if w['label'] == label and abs(w['locus'].tf - tf) < 1e-10:
                    return w
        return None
    
    def get_from_tf(self, tf):
        # return all the wavefronts at tf
        wavefronts_at_tf = {}
        if self.data is not None and self.data.contains(self.data_name):
            wavefronts = self.data.get(self.data_name)
            for w in wavefronts:
                w['locus'] = geometry2d.problem.WaveFrontLocus(w['locus'])
                if abs(w['locus'].tf - tf) < 1e-10:
                    # add the wavefront to the dictionary
                    # the key is the label and the value is the locus
                    wavefronts_at_tf[w['label']] = w['locus']
        return wavefronts_at_tf
    
    def remove_from_label__(self, tf, label):
        if self.data is not None and self.data.contains(self.data_name):
            wavefronts = self.data.get(self.data_name)
            for w in wavefronts:
                w['locus'] = geometry2d.problem.WaveFrontLocus(w['locus'])
                if w['label'] == label and abs(w['locus'].tf - tf) < 1e-10:
                    wavefronts.remove(w)
                    break
            self.data.update({self.data_name: wavefronts})
            
    def remove_from_tf(self, tf):
        if self.data is not None and self.data.contains(self.data_name):
            wavefronts = self.data.get(self.data_name)
            for w in wavefronts:
                w['locus'] = geometry2d.problem.WaveFrontLocus(w['locus'])
                if abs(w['locus'].tf - tf) < 1e-10:
                    wavefronts.remove(w)
            self.data.update({self.data_name: wavefronts})
     
    def compute(self, tf, αspan=None, *, label=None, save=True, reset=False, load=True, options=None):
        
        if (αspan is None) and (label is not None) and load:
            if (self.data is None):
                raise ValueError('You need a data dictionary to load the conjugate locus.')
            if (not self.data.contains(self.data_name)):
                raise ValueError('You need a data dictionary to load the conjugate locus.')
            if not (self.exists_from_label__(tf, label)):
                # return that the data does not contain the label
                raise ValueError('The data dictionary does not contain the label.')
            # get the wavefront and return it
            return self.get_from_label__(tf, label)
        
        # if reset is True then we remove the wavefronts at tf if they exist
        if reset:
            if label is None:
                self.remove_from_tf(tf)
            else:
                self.remove_from_label__(tf, label)
        
        # if data is None then force save to False
        if self.data is None:
            save = False
            load = False
            
        if αspan is None:
            
            wavefront_left_loaded  = False
            wavefront_right_loaded = False
            wavefront_left  = None
            wavefront_right = None
            if load:
                wavefront_left  = self.get_from_label__(tf, 'left')
                wavefront_right = self.get_from_label__(tf, 'right')
                wavefront_left_loaded = wavefront_left is not None
                wavefront_right_loaded = wavefront_right is not None
                
            # gap between the left and right parts of the conjugate locus
            gap = 1e-3
            
            # right part
            α0  = -np.pi/2+gap
            αf  =  np.pi/2-gap
            if wavefront_right is None:
                w = self.problem.wavefront(tf, α0, αf, options=options)
                wavefront_right = {'label':'right', 'locus':w}
                
            # left part
            α0  = 1*np.pi/2+gap
            αf  = 3*np.pi/2-gap
            if wavefront_left is None:
                w = self.problem.wavefront(tf, α0, αf, options=options)
                wavefront_left = {'label':'left', 'locus':w}
                
            # save the data
            if save:
                
                # get the data
                if self.data.contains(self.data_name):
                    data_wavefronts = self.data.get(self.data_name)
                else:
                    data_wavefronts = []
                    
                # update the data
                if not wavefront_left_loaded:
                    # make a deep copy of the wavefront
                    data_wavefront_left = wavefront_left.copy()
                    data_wavefront_left['locus'] = wavefront_left['locus'].get_data()
                    data_wavefronts.append(data_wavefront_left)
                if not wavefront_right_loaded:
                    data_wavefront_right = wavefront_right.copy()
                    data_wavefront_right['locus'] = wavefront_right['locus'].get_data()
                    data_wavefronts.append(data_wavefront_right)
                    
                # save the data
                self.data.update({self.data_name: data_wavefronts})
                
            # return the wavefront
            return [wavefront_left, wavefront_right]
        
        else:
            
            if label is None:
                label = self.label_default
                
            # load
            wavefront_loaded = False
            wavefront = None
            if load:
                wavefront = self.get_from_label__(tf, label)
                wavefront_loaded = wavefront is not None
                
            # compute
            if wavefront is None:
                w = self.problem.wavefront(tf, αspan[0], αspan[1], options=options)
                wavefront = {'label':label, 'locus':w}
                
            # save
            if save:
                
                # get the data
                if self.data.contains(self.data_name):
                    data_wavefronts = self.data.get(self.data_name)
                else:
                    data_wavefronts = []
                    
                # update the data
                if not wavefront_loaded:
                    
                    data_wavefront = wavefront.copy()
                    data_wavefront['locus'] = wavefront['locus'].get_data()
                    data_wavefronts.append(data_wavefront)
                    
                    # save the data
                    self.data.update({self.data_name: data_wavefronts})
                    
            # return
            return wavefront
        
    def plot_2d(self, wavefront_loci, *, color=None, linewidth=None, zorder=None, figure=None,
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
            
        # check if wavefront_loci is a list
        if not isinstance(wavefront_loci, list):
            wavefront_loci = [wavefront_loci]
            
        # iterate over the wavefront_loci
        for w in wavefront_loci:
            
            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = w['locus'].states[:, 0]
            θ = w['locus'].states[:, 1]
            φ = r - np.pi/2
            
            # plot the wavefront_loci
            geometry2d.plottings.plot_2d(figure, θ,         φ, color=color, linewidth=linewidth, zorder=zorder)
            geometry2d.plottings.plot_2d(figure, θ+2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder)
            geometry2d.plottings.plot_2d(figure, θ-2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder)
            
        return figure
    
    def plot_3d(self, wavefront_loci, *,
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
            
        # check if wavefront_loci is a list
        if not isinstance(wavefront_loci, list):
            wavefront_loci = [wavefront_loci]
            
        # iterate over the wavefront_loci
        for w in wavefront_loci:
            
            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = w['locus'].states[:, 0]
            θ = w['locus'].states[:, 1]
            φ = r - np.pi/2
            
            # plot the wavefront_loci
            x, y, z = geometry2d.plottings.coord3d(θ, φ, self.problem.epsilon)
            geometry2d.plottings.plot_3d(figure, x, y, z, color=color, linewidth=linewidth, zorder=zorder)
            
        return figure
    
    def plot(self, tf, labels=None, *,
            view=geometry2d.plottings.Coords.SPHERE, 
            elevation=geometry2d.plottings.elevation__,
            azimuth=geometry2d.plottings.azimuth__, 
            color=None, linewidth=None, zorder=None, figure=None, dpi=None, figsize=None):
        
        # if there is no wavefront at tf then raise an error
        if not self.exists_from_tf(tf):
            raise ValueError('There is no wavefront at tf.')
        
        # get the wavefronts at tf
        wavefronts_at_tf = self.get_from_tf(tf)
        
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
        left_right_in_labels_and_SPHERE = ('left' in labels) and ('right' in labels) and \
                                            (view == geometry2d.plottings.Coords.SPHERE)
        # if left_right_in_labels is True, then we concatenate the wavefronts before plotting
        if left_right_in_labels_and_SPHERE:
            # get the wavefronts at tf. Then, concatenate and plot
            wl = wavefronts_at_tf['left']
            wr = wavefronts_at_tf['right']
            r = np.concatenate((wl.states[:, 0], wr.states[:, 0]))
            θ = np.concatenate((wl.states[:, 1], wr.states[:, 1]))
            α = np.concatenate((wl.alphas, wr.alphas))
            wavefront_locus = geometry2d.problem.WaveFrontLocus((tf, np.array([r, θ]).T, α))
            # plot
            figure = self.plot_3d([{'label':'default', 'locus':wavefront_locus}], 
                                    elevation=elevation, azimuth=azimuth, 
                                    color=color, linewidth=linewidth, zorder=zorder, 
                                    figure=figure, dpi=dpi, figsize=figsize)
                
        # iterates over the labels and create a list of wavefronts
        wavefronts_loci = []
        for label in labels:
            # do not append if label is 'left' or 'right' and left_right_in_labels is True
            if (left_right_in_labels_and_SPHERE) and ((label == 'left') or (label == 'right')):
                continue # this skip the rest of the loop and goes to the next iteration
            wavefronts_loci.append({'label':label, 'locus':wavefronts_at_tf[label]})
            
        # plot
        if view == geometry2d.plottings.Coords.SPHERE:
            figure = self.plot_3d(wavefronts_loci, 
                                   elevation=elevation, azimuth=azimuth, 
                                   color=color, linewidth=linewidth, zorder=zorder, 
                                   figure=figure, dpi=dpi, figsize=figsize)
        elif view == geometry2d.plottings.Coords.PLANE:
            figure = self.plot_2d(wavefronts_loci, 
                                   color=color, linewidth=linewidth, zorder=zorder, 
                                   figure=figure, dpi=dpi, figsize=figsize)
        else:
            raise ValueError('The view is not valid.')
        
        return figure
        
    #
    def get_data_from_self_intersections(self, tf, curve, αs, *, modulo=None):
        #
        xxs = geometry2d.utils.get_self_intersections(curve, modulo=modulo)
        intersections = []
        for xx in xxs:    
            x     = xx.get('x')
            i     = xx.get('i')
            j     = xx.get('j')
            la    = xx.get('la')
            mu    = xx.get('mu')
            α1    = αs[i]+la*(αs[i+1]-αs[i])
            α2    = αs[j]+mu*(αs[j+1]-αs[j])
            intersections.append( ( tf, α1, α2, x ) )
        return intersections
    
    def self_intersections(self, tf, label=None):
        
        if label is None:
            # check if 'left' and 'right' exist
            if not self.exists_from_label__(tf, 'left') or not self.exists_from_label__(tf, 'right'):
                raise ValueError('The data dictionary does not contain the wavefront.')
        elif not self.exists_from_label__(tf, label):
            raise ValueError('The data dictionary does not contain the label.')
        
        # get the wavefront
        # if label is None, then get the wavefronts 'left' and 'right'
        # and concatenate them, adding 2pi when theta<0
        if label is None:
            intersections = {}
            #
            w = self.get_from_label__(tf, 'left')['locus']
            i = self.get_data_from_self_intersections(tf, w.states, w.alphas)
            intersections['left'] = i[0] # only the first intersection
            #
            w = self.get_from_label__(tf, 'right')['locus']
            i = self.get_data_from_self_intersections(tf, w.states, w.alphas)
            intersections['right'] = i[0] # only the first intersection
        else:
            w = self.get_from_label__(tf, label)['locus']
            intersections = self.get_data_from_self_intersections(tf, w.states, w.alphas)
        
        # return the self-intersections
        return intersections