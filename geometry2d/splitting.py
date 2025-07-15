import geometry2d.plottings
import numpy as np
import scipy

class Splitting():
    
    def __init__(self, problem):
        
        self.problem = problem
        self.data = problem.data
        self.data_name = 'splitting'
        self.label_default = 'default'
        self.color = 'k'
        self.linewidth = 0.5
        self.zorder = geometry2d.plottings.z_order_splitting
        
    def exists_from_label(self, label):
        # check if the data exists
        if self.data is None:
            return False
        if not self.data.contains(self.data_name):
            return False
        if not label in self.data.get(self.data_name).keys():
            return False
        return True
    
    def get_from_label(self, label):
        if self.data is not None and self.data.contains(self.data_name):
            if self.exists_from_label(label):
                return geometry2d.problem.SplittingLocus(self.data.get(self.data_name)[label])
        return None
    
    def remove_from_label(self, label):
        if self.data is not None and self.data.contains(self.data_name):
            if self.exists_from_label(label):
                self.data.get(self.data_name).pop(label)
    
    def compute(self, y, αspan=None, *, label=None, save=True, reset=False, load=True, options=None, gap=1e-3):
        # y is a self-intersection data of the wavefront
        # y[label] = (tf, α1, α2, q)
        
        if (αspan is None) and (label is not None) and load:
            if (self.data is None):
                raise ValueError('You need a data dictionary to load the splitting locus.')
            if (not self.data.contains(self.data_name)):
                raise ValueError('You need a data dictionary to load the splitting locus.')
            if not (self.exists_from_label__(label)):
                # return that the data does not contain the label
                raise ValueError('The data dictionary does not contain the label.')
            # get the wavefront and return it
            return self.get_from_label__(label)
        
        if reset:
            if label is not None:
                self.remove_from_label(label)
            else:
                if αspan is None:
                    self.remove_from_label('left')
                    self.remove_from_label('right')
                else:
                    self.remove_from_label(self.label_default)
                
        # if data is None then force save to False
        if self.data is None:
            save = False
            load = False
            
        if αspan is None:
            
            splitting_left_loaded = False
            splitting_right_loaded = False
            if load:
                splitting_left_loaded  = self.exists_from_label('left')
                splitting_right_loaded = self.exists_from_label('right')
                splitting_left  = self.get_from_label('left')
                splitting_right = self.get_from_label('right')
            
            #gap = 1e-3
            
            # if y is not a dictionary then raise an error
            if not isinstance(y, dict):
                raise ValueError('y must be a dictionary.')
            
            # if y does not contain the keys 'left' and 'right' then raise an error
            if not ('left' in y.keys() and 'right' in y.keys()):
                raise ValueError('y must contain the keys "left" and "right".')
            
            # right part
            if not splitting_right_loaded:
                tf = y['right'][0]
                α1 = y['right'][1]
                α2 = y['right'][2]
                q  = y['right'][3]
                α0  = 0*np.pi/2+gap
                αf  = 1*np.pi/2-gap
                splitting_right = self.problem.splitting_locus(q, α1, tf, α2, α0, αf, options=options)
                
            # left part
            if not splitting_left_loaded:
                tf = y['left'][0]
                α1 = y['left'][1]
                α2 = y['left'][2]
                q  = y['left'][3]
                α0  = 1*np.pi/2+gap
                αf  = 2*np.pi/2-gap
                splitting_left = self.problem.splitting_locus(q, α2, tf, α1, α0, αf, options=options)
                
            # save the data
            if save:
                
                # get the data
                if self.data.contains(self.data_name):
                    data_splitting_locus = self.data.get(self.data_name)
                else:
                    data_splitting_locus = {}
                    
                data_splitting_locus['left']  = splitting_left.get_data()
                data_splitting_locus['right'] = splitting_right.get_data()
                
                # save the data
                self.data.update({self.data_name: data_splitting_locus})
                
            # return the splitting locus
            return [splitting_left, splitting_right]
        
        else:
            
            if label is None:
                label = self.label_default
                
            # load
            if load:
                if self.exists_from_label(label):
                    splitting = self.get_from_label(label)
                else:
                    splitting = None
            else:
                splitting = None
                
            # compute
            if splitting is None:
                tf = y[0]
                α1 = y[1]
                α2 = y[2]
                q  = y[3]
                α0 = αspan[0]
                αf = αspan[1]
                splitting = self.problem.splitting_locus(q, α1, tf, α2, α0, αf, options=options)
            
            # save
            if save:
                
                # get the data
                if self.data.contains(self.data_name):
                    data_splitting_locus = self.data.get(self.data_name)
                else:
                    data_splitting_locus = {}
                    
                data_splitting_locus[label] = splitting.get_data()
                
                # save the data
                self.data.update({self.data_name: data_splitting_locus})
                
            # return the splitting locus
            return splitting

    def splitting_time(self, splitting_loci):
        
        # check if splitting_loci is a list
        if not isinstance(splitting_loci, list):
            splitting_loci = [splitting_loci]
        
        # create a function α0 -> t(α0) that returns the splitting time
        def splitting_time__(α0):
            
            α = α0 % (2*np.pi)
            d = np.inf
            a = None
            t = None
            
            # get the splitting locus that contains α0
            for splitting_locus in splitting_loci:
                
                αs = splitting_locus.alphas[:, 0] % (2*np.pi)
                if min(αs) <= α <= max(αs):
                    return scipy.interpolate.interp1d(αs, splitting_locus.times, kind='cubic', 
                                                      fill_value="extrapolate")(α)
                d_ = min(abs(αs - α))
                if d_ < d:
                    d = d_
                    a = αs
                    t = splitting_locus.times
                    
                αs = splitting_locus.alphas[:, 1] % (2*np.pi)
                if min(αs) <= α <= max(αs):
                    return scipy.interpolate.interp1d(αs, splitting_locus.times, kind='cubic', 
                                                      fill_value="extrapolate")(α)
                
                d_ = min(abs(αs - α))
                if d_ < d:
                    d = d_
                    a = αs
                    t = splitting_locus.times
            
            # interpolate with the closest splitting locus
            if a is not None:
                return scipy.interpolate.interp1d(a, t, kind='cubic', fill_value="extrapolate")(α)
            else:
                # raise en internal error
                raise ValueError('Internal error.')
               
            # if α0 is not in any splitting locus then throw an error
            #raise ValueError('α0 is not in any splitting locus.')
        
        return splitting_time__
            
    def plot_2d(self, splitting_loci, *, color=None, linewidth=None, zorder=None, 
                figure=None, linestyle='solid',
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

        # check if splitting_loci is a list
        if not isinstance(splitting_loci, list):
            splitting_loci = [splitting_loci]

        # iterate over the splitting loci
        for splitting_locus in splitting_loci:

            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = splitting_locus.states[:, 0]
            θ = splitting_locus.states[:, 1]
            φ = r - np.pi/2
            
            # plot the splitting locus            
            geometry2d.plottings.plot_2d(figure, θ,         φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
            geometry2d.plottings.plot_2d(figure, θ+2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
            geometry2d.plottings.plot_2d(figure, θ-2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
        
        return figure
 
    def plot_3d(self, splitting_loci, *,
                elevation=geometry2d.plottings.elevation__,
                azimuth=geometry2d.plottings.azimuth__, 
                color=None, linewidth=None, zorder=None, linestyle='solid',
                figure=None,
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
        
        # check if splitting_loci is a list
        if not isinstance(splitting_loci, list):
            splitting_loci = [splitting_loci]
            
        # iterate over the splitting loci
        for splitting_locus in splitting_loci:
            
            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = splitting_locus.states[:, 0]
            θ = splitting_locus.states[:, 1]
            φ = r - np.pi/2
            
            # plot the splitting locus
            x, y, z = geometry2d.plottings.coord3d(θ, φ, self.problem.epsilon)
            geometry2d.plottings.plot_3d(figure, x, y, z, color=color, linewidth=linewidth, 
                                         zorder=zorder, linestyle=linestyle)
            
        return figure    
    
    def plot(self, labels=None, *,
            view=geometry2d.plottings.Coords.SPHERE, 
            elevation=geometry2d.plottings.elevation__,
            azimuth=geometry2d.plottings.azimuth__, 
            color=None, linewidth=None, zorder=None, linestyle='solid',
            figure=None, dpi=None, figsize=None):
        
        # if label is None, we plot all the saved splitting loci
        if labels is None:
            labels = self.data.get(self.data_name).keys()
        # check if labels is a list
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
        if left_right_in_labels_and_SPHERE:
            # get splitting loci for 'left' and 'right'
            splitting_loci_left  = self.get_from_label('left')
            splitting_loci_right = self.get_from_label('right')
            # if one of the splitting loci is None, then raise an error
            if (splitting_loci_left is None) or (splitting_loci_right is None):
                raise ValueError('The splitting loci for "left" and "right" must exist.')
            # concatenate the splitting loci: modulo (2*np.pi) for index 1 and 2
            # splitting locus being a np.array of size (n, 3)
            t  = np.concatenate((splitting_loci_right.times, splitting_loci_left.times))
            r  = np.concatenate((splitting_loci_right.states[:, 0], splitting_loci_left.states[:, 0])) % (2*np.pi)
            θ  = np.concatenate((splitting_loci_right.states[:, 1], splitting_loci_left.states[:, 1])) % (2*np.pi)
            α1 = np.concatenate((splitting_loci_right.alphas[:, 0], splitting_loci_left.alphas[:, 0])) % (2*np.pi)
            α2 = np.concatenate((splitting_loci_right.alphas[:, 1], splitting_loci_left.alphas[:, 1])) % (2*np.pi)
            # aggregate the splitting loci
            splitting_locus = geometry2d.problem.SplittingLocus(( t, np.array([r, θ]).T, np.array([α1, α2]).T ))
            # plot the splitting locus
            figure = self.plot_3d(splitting_locus, 
                                    elevation=elevation, azimuth=azimuth, 
                                    color=color, linewidth=linewidth, zorder=zorder,
                                    figure=figure, dpi=dpi, figsize=figsize, linestyle=linestyle)

        # iterate over the labels and create a list of the splitting loci
        splitting_loci = []
        for label in labels:
            # do not append if label is 'left' or 'right' and left_right_in_labels is True
            if (left_right_in_labels_and_SPHERE) and ((label == 'left') or (label == 'right')):
                continue # this skip the rest of the loop and goes to the next iteration
            if self.exists_from_label(label):
                splitting_loci.append(self.get_from_label(label))
            
        # plot the splitting loci
        if view == geometry2d.plottings.Coords.SPHERE:
            figure = self.plot_3d(splitting_loci, 
                                  elevation=elevation, azimuth=azimuth, 
                                  color=color, linewidth=linewidth, zorder=zorder,
                                  figure=figure, dpi=dpi, figsize=figsize, linestyle=linestyle)
        elif view == geometry2d.plottings.Coords.PLANE:
            figure = self.plot_2d(splitting_loci, 
                                  color=color, linewidth=linewidth, zorder=zorder, 
                                  figure=figure, dpi=dpi, figsize=figsize, linestyle=linestyle)
        else:
            raise ValueError('The view must be either SPHERE or PLANE.')

        return figure
    