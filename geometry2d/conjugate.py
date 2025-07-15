import geometry2d.plottings
import numpy as np
import scipy

class Conjugate():
    
    def __init__(self, problem):
        
        self.problem = problem
        self.data_name = 'conjugate_locus'
        self.label_default = 'default'
        self.color = 'r'
        self.linewidth = 1
        self.zorder = geometry2d.plottings.z_order_conjugate
        self.data = problem.data
        self.color_surface = 'r'
        self.alpha_surface = 0.4
        self.zorder_surface = geometry2d.plottings.z_order_conj_surf

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
                return geometry2d.problem.ConjugateLocus(self.data.get(self.data_name)[label])
        return None
    
    def remove_from_label(self, label):
        if self.data is not None and self.data.contains(self.data_name):
            if self.exists_from_label(label):
                self.data.get(self.data_name).pop(label)
                
    def compute(self, αspan=None, *, label=None, save=True, reset=False, load=True, gap=1e-2):
        
        # # if αspan is None then label must be None otherwise raise an error
        # if (αspan is None) and (label is not None):
        #         raise ValueError('If αspan is None then label must be None.')
            
        # if αspan is None, label is not None, load is True and data is None or does not contain the label 
        # then raise an error
        if (αspan is None) and (label is not None) and load:
            if (self.data is None):
                raise ValueError('You need a data dictionary to load the conjugate locus.')
            if (not self.data.contains(self.data_name)):
                raise ValueError('You need a data dictionary to load the conjugate locus.')
            if not (label in self.data.get(self.data_name)):
                # return that the data does not contain the label
                raise ValueError('The data dictionary does not contain the label.')
            # get the conjugate locus and return it
            return self.get_from_label(label)

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

        # if αspan is None then compute the left and right parts of the conjugate locus
        # then save the data in the data dictionary if save is True
        # and then return a list of the computed conjugate loci
        if αspan is None:
            
            conjugate_left_loaded = False
            conjugate_right_loaded = False
            if load:
                conjugate_left_loaded  = self.exists_from_label('left')
                conjugate_right_loaded = self.exists_from_label('right')
                conjugate_left  = self.get_from_label('left')
                conjugate_right = self.get_from_label('right')
            
            # gap between the left and right parts of the conjugate locus
            #gap = 1e-2
            
            # right part
            if not conjugate_right_loaded:
                α0  = -np.pi/2+gap
                αf  =  np.pi/2-gap
                conjugate_right = self.problem.conjugate_locus(α0, αf)
            
            # left part
            if not conjugate_left_loaded:
                α0  = 1*np.pi/2+gap
                αf  = 3*np.pi/2-gap
                conjugate_left = self.problem.conjugate_locus(α0, αf)
        
            # save the data
            if save:
                
                # get the data
                if self.data.contains(self.data_name):
                    data_conjugate_locus = self.data.get(self.data_name)
                else:
                    data_conjugate_locus = {}
                    
                # update the data
                data_conjugate_locus['left']  = conjugate_left.get_data()
                data_conjugate_locus['right'] = conjugate_right.get_data()
                self.data.update({self.data_name: data_conjugate_locus})
                
            # return the list of the computed conjugate loci
            return [conjugate_left, conjugate_right]
        
        # if αspan is not None then compute the conjugate locus for the given αspan
        # then save the data in the data dictionary if save is True
        # and then return the computed conjugate locus
        else:
            
            if label is None:
                label = self.label_default
            
            # load
            if load:
                if self.exists_from_label(label):
                    conjugate = self.get_from_label(label)
                else:
                    conjugate = None
            else:
                conjugate = None
            
            # compute the conjugate locus
            if conjugate is None:
                conjugate = self.problem.conjugate_locus(float(αspan[0]), float(αspan[1]))
            
            # save the data
            if save:
                
                # get the data
                if self.data.contains(self.data_name):
                    data_conjugate_locus = self.data.get(self.data_name)
                else:
                    data_conjugate_locus = {}
                    
                # save the data
                data_conjugate_locus[label] = conjugate.get_data()
                self.data.update({self.data_name: data_conjugate_locus})
                
            # return the computed conjugate locus
            return conjugate
        
    def conjugate_time(self, conjugate_loci=None):
        
        # check if conjugate_loci is a list
        if not isinstance(conjugate_loci, list):
            conjugate_loci = [conjugate_loci]
            
        # create a function α0 -> t(α0) that returns the conjugate time
        def conjugate_time__(α0):
            
            # get the conjugate locus which contains α0
            for conjugate_locus in conjugate_loci:
                alphas = conjugate_locus.alphas % (2*np.pi)
                a      = α0 % (2*np.pi)
                if (a >= min(alphas)) and (a <= max(alphas)):
                    return scipy.interpolate.interp1d(alphas, conjugate_locus.times, kind='cubic')(a)
                
            # throw an error if α0 is not in the conjugate locus
            raise ValueError('α0 is not in the conjugate locus.')
        
        return conjugate_time__
            
        
    # function to plot one or more conjugate loci
    def plot_2d(self, conjugate_loci, *, color=None, linewidth=None, zorder=None, linestyle='solid',
                figure=None, dpi=geometry2d.plottings.dpi__, figsize=geometry2d.plottings.figsize_2d__):
        
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

        # check if conjugate_loci is a list of geometry2d.problem.ConjugateLocus
        if not isinstance(conjugate_loci, list):
            conjugate_loci = [conjugate_loci]

        # iterate over the conjugate loci
        for conjugate_locus in conjugate_loci:

            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = conjugate_locus.states[:, 0]
            θ = conjugate_locus.states[:, 1]
            φ = r - np.pi/2
            
            # plot the conjugate locus
            geometry2d.plottings.plot_2d(figure, θ,         φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
            geometry2d.plottings.plot_2d(figure, θ+2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
            geometry2d.plottings.plot_2d(figure, θ-2*np.pi, φ, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
        
        return figure
    
    def plot_3d(self, conjugate_loci, *,
                elevation=geometry2d.plottings.elevation__,
                azimuth=geometry2d.plottings.azimuth__, 
                color=None, linewidth=None, zorder=None, linestyle='solid',
                plot_surface=False, color_surface=None, alpha_surface=None, zorder_surface=None, 
                figure=None, dpi=geometry2d.plottings.dpi__, figsize=geometry2d.plottings.figsize_3d__):
        
        # initialize the parameters
        if color is None:
            color = self.color
        if linewidth is None:
            linewidth = self.linewidth
        if zorder is None:
            zorder = self.zorder
        if color_surface is None:
            color_surface = self.color_surface
        if alpha_surface is None:
            alpha_surface = self.alpha_surface
        if zorder_surface is None:
            zorder_surface = self.zorder_surface
            
        # initialize the figure
        if figure is None:
            figure = geometry2d.plottings.init_figure_3d(self.problem.epsilon, 
                                                    self.problem.initial_point,
                                                    elevation,
                                                    azimuth, dpi=dpi, figsize=figsize)
        
        # check if conjugate_loci is a list
        if not isinstance(conjugate_loci, list):
            conjugate_loci = [conjugate_loci]
            
        # iterate over the conjugate loci
        for conjugate_locus in conjugate_loci:
            
            # change of coordinates
            # from (r, θ) to (θ, φ), with φ = r - π / 2
            # r = π / 2 is the equator
            r = conjugate_locus.states[:, 0]
            θ = conjugate_locus.states[:, 1]
            φ = r - np.pi/2
            
            # plot the conjugate locus
            x, y, z = geometry2d.plottings.coord3d(θ, φ, self.problem.epsilon)
            geometry2d.plottings.plot_3d(figure, x, y, z, color=color, linewidth=linewidth, zorder=zorder, linestyle=linestyle)
            
            # if plot_surface is True then plot the surface
            if plot_surface:
                X, Y, Z = geometry2d.plottings.surface_from_spherical_curve(θ, φ, self.problem.epsilon)
                geometry2d.plottings.plot_surface(figure, X, Y, Z, 
                                                  color=color_surface, alpha=alpha_surface, zorder=zorder_surface)

        return figure
    
    def plot(self, labels=None, *,
            view=geometry2d.plottings.Coords.SPHERE, 
            elevation=geometry2d.plottings.elevation__,
            azimuth=geometry2d.plottings.azimuth__, 
            color=None, linewidth=None, zorder=None, linestyle='solid',
            plot_surface=None, color_surface=None, alpha_surface=None, zorder_surface=None, figure=None,
            dpi=None, figsize=None):
        
        # if label is None, we plot all the saved conjugate loci
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
            #
            if plot_surface is None:
                plot_surface_left_right = True
            else:
                plot_surface_left_right = plot_surface
            #
            # get conjugate loci for 'left' and 'right'
            conjugate_loci_left  = self.compute(label='left', load=True)
            conjugate_loci_right = self.compute(label='right', load=True)
            # concatenate the conjugate loci: modulo (2*np.pi) for index 1 and 2
            # conjugate locus being a np.array of size (n, 3)
            t = np.concatenate((conjugate_loci_left.times, conjugate_loci_right.times))
            r = np.concatenate((conjugate_loci_left.states[:, 0], conjugate_loci_right.states[:, 0])) % (2*np.pi)
            θ = np.concatenate((conjugate_loci_left.states[:, 1], conjugate_loci_right.states[:, 1])) % (2*np.pi)
            α = np.concatenate((conjugate_loci_left.alphas, conjugate_loci_right.alphas)) % (2*np.pi)
            # aggregate the conjugate loci
            conjugate_locus = geometry2d.problem.ConjugateLocus(( t, np.array([r, θ]).T, α ))
            # plot the conjugate locus
            figure = self.plot_3d(conjugate_locus, 
                                    elevation=elevation, azimuth=azimuth, 
                                    color=color, linewidth=linewidth, zorder=zorder,
                                    plot_surface=plot_surface_left_right, color_surface=color_surface, 
                                    alpha_surface=alpha_surface, zorder_surface=zorder_surface, 
                                    figure=figure, dpi=dpi, figsize=figsize, linestyle=linestyle)

        # iterate over the labels and create a list of the conjugate loci
        conjugate_loci = []
        for label in labels:
            # do not append if label is 'left' or 'right' and left_right_in_labels is True
            if (left_right_in_labels_and_SPHERE) and ((label == 'left') or (label == 'right')):
                continue # this skip the rest of the loop and goes to the next iteration
            conjugate_loci.append(self.compute(label=label, load=True))
            
        # plot the conjugate loci
        if view == geometry2d.plottings.Coords.SPHERE:
            figure = self.plot_3d(conjugate_loci, 
                                  elevation=elevation, azimuth=azimuth, 
                                  color=color, linewidth=linewidth, zorder=zorder,
                                  plot_surface=plot_surface, color_surface=color_surface, 
                                  alpha_surface=alpha_surface, zorder_surface=zorder_surface, 
                                  figure=figure, dpi=dpi, figsize=figsize, linestyle=linestyle)
        elif view == geometry2d.plottings.Coords.PLANE:
            figure = self.plot_2d(conjugate_loci, 
                                  color=color, linewidth=linewidth, zorder=zorder, 
                                  figure=figure, dpi=dpi, figsize=figsize, linestyle=linestyle)
        else:
            raise ValueError('The view must be either SPHERE or PLANE.')

        return figure
    