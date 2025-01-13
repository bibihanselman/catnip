'''
CATNIP: Comparative Analysis Tool for Normalized Imagery and Profiles. (Working name)
Processes multiwavelength disk images using FIGG, and computes radial
and azimuthal profiles according to user input.
Original written by Cat Sarosi, 2022
Modified by Bibi Hanselman
Created 30 July 2024
Updated 30 July 2024
'''

import numpy as np
import cv2
import math
import seaborn as sb
import pandas as pd
import astropy.units as u
import photutils.aperture as phot
from matplotlib.patches import Circle, Ellipse
from copy import copy, deepcopy
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

class AstroObject():
    '''
    Class to represent a disk to analyze. Contains attributes for individual 
    image information (AstroImage instances), disk properties, and multiline 
    plot display settings.
    
    
    Attributes
    ----------
    name : str
        Disk name.
    scattered_light: AstroImage
        AstroImage instance for scattered light data.
    continuum: AstroImage
        AstroImage instance for mm continuum data.
    line: AstroImage
        AstroImage instance for line emission data.
    imlist: list
        List containing all applicable AstroImages.
    img_data_dict: dict
        Contains unprocessed image arrays for each AstroImage object.
    prc_ims: dict
        Contains FIGG-processed image arrays for each AstroImage object.
    prc_scales: dict
        Contains ImageNormalize instances for each AstroImage object, determining
        the colorbar stretch.
    plot_types: list
        List of strings denoting the types of panels to plot.
    rad_lines: bool
        Whether to overlay ray annotations on disk images denoting the azimuthal
        location of radial profiles.
    az_rings: bool
        Whether to overlay ring annotations on disk images denoting the radial
        location of azimuthal profiles.
    line_cmap: str
        Name of matplotlib.colormap.Colormap with which to color ray annotations.
    ring_cmap: str
        Name of matplotlib.colormap.Colormap with which to color ring annotations.
    column_width: float
        Width, in inches, of each column in the multiline plot.
    row_ht: float
        Height, in inches, of each row in the multiline plot.
    img_mult: float
        Height multiplier for imagery panels. The height of these panels is
        row_ht multiplied by this value.
    force_simple: bool
        Whether to always produce "simple" (single-column) plots regardless of
        the number of panels. By default, make_plot_multiline() automatically creates
        a "complex" (two-column) layout when there are more than 3 panels. Setting
        this value to True forgoes this check.
    au_scale: float
        Width of au scalebar in au. If set to 0, a scalebar is not plotted.
    distance: float
        Distance of the disk in pc.
    PAlist : list
        PAs for which to plot radial profiles for each image in the multiline
        plot.
     
        
    Methods
    -------
    update(display_dict):
        Sets values for display-related attributes.
    plot_im(im, im_ax, subfig):
        Helper function to plot an image.
    check_plot_types():
        Check if the inputted plot types (elements of plot_types) are compatible
        with the panel options in make_plot_multline().
    partition_list(lst):
        Returns all possible ways to partition the elements in a list into 2 
        separate lists. Used to determine all possible panel configurations
        in the complex layout.
    optimize_cols(lst):
        In the complex layout, chooses the column pairing that minimizes the
        difference in heights between the two columns.
    make_plot_multiline():
        Generates a multipanel plot displaying disk imagery and/or intensity
        profiles for the disk. Types of panels, their order, and the choice of
        column layout are all user-controlled.
    
    History
    -------
    Original author: Kate Follette in IDL.
    Revised by Cat Sarosi (2022).
    Revised further by Bibi Hanselman (2024). New display-related attributes and
    an overhaul of the axis-plotting routine to extend flexibility/ 
    user control and improve the aesthetic quality of the final graphic. New
    plotting options (ex. projected imagery) to enhance the utility of the tool,
    with support for custom inputs for each disk in the user interface to cover
    many possible disk image inputs. Also incorporated processing code from
    the figg module during initialization to process the images according to
    user-defined parameters.
    '''
    def __init__(self, name, image_list, display_dict):
        '''
        Initialize required disk attributes and prepare plots (deprojected images,
        intensity profiles) for each disk image.
        
        Parameters
        ----------
        name : str
            Disk name.
        image_dict : dict
            Dict pulled from sheets interface that contains values for
            each of the instantiated image (AstroImage) attributes.
        display_dict : dict
            Dict pulled from sheets interface that contains values for
            each of the disk (AstroObject) attributes.
        '''
        self.name = name

        # Instantiate AstroImages
        self.imlist = [AstroImage(im) for im in image_list]

        # Set display info
        self.update(display_dict)

        # Assign certain disk-wide parameters to the images
        for im in self.imlist:
            im.incl = display_dict['Inclination']
            im.PA = display_dict['PA']
            
            im.PAs = np.arange(0,360,display_dict['PA Step']).tolist()
            im.PAs.append(im.PA)
            im.PAs.append(im.PA + 90)
            for PA in self.PAlist: im.PAs.append(PA)

            im.radlist = display_dict['Radius List']
        
        # Create the profiles
        for im in self.imlist: im.profiles()

#############################################################################

    def update(self, display_dict):
        '''
        Updates display-related values in the disk dictionary.
        To update image processing parameters (PA, inclination, PAstep, PA list, radius list),
        you must reinstantiate the AstroObject to reprocess the imagery and profiles.
        '''
        self.plot_types = display_dict['Plot Types']
        self.annotate = display_dict['Annotations']
        ### The following are hard-coded for now, but will be customizable display parameters in a future update ###
        self.line_cmap = 'summer'
        self.ring_cmap = 'autumn'
        self.column_width = 10
        self.row_ht = 3
        self.img_mult = 2
        self.force_simple = False
        ############################################################################################################
        self.au_scale = display_dict['AU Scale']
        self.distance = display_dict['Distance']
        self.PAlist = display_dict['PA List']

#############################################################################

    def plot_im(self, im, im_ax, subfig):
        '''
        Helper function to plot image data.

        Parameters
        ----------
        im : AstroImage
            Image instance containing the data to be displayed, along with
            relevant cropping parameters.
        im_ax : matplotlib.axes.Axes
            Axis on which to plot the image (corresponds to imagery type).
        subfig : matplotlib.figure.SubFigure
            Subfigure containing the desired axis (deprojected or on sky).

        Returns
        -------
        None.

        '''
        bounds = {}
        boundsize = int(round(self.boundsize/im.pixscale))
        bounds['x'] = [im.xcen - boundsize, im.xcen - boundsize/2, im.xcen + boundsize/2, im.xcen + boundsize]
        bounds['y'] = [im.ycen - boundsize, im.ycen - boundsize/2, im.ycen + boundsize/2, im.ycen + boundsize]

        if im.r2:
            im.cropped_data = im.data[int(round(im.bounds['y'][0])):int(round(im.bounds['y'][3])),int(round(im.bounds['x'][0])):int(round(im.bounds['x'][3]))]
            im.norm_data = im.data/np.nanmax(im.cropped_data)
        else:
            im.norm_data = im.data/np.nanmax(im.data) # Unnecessary? 7/23/24

        im_show = im_ax.imshow(im.norm_data,
                          cmap = im.cmap, origin = 'lower',
                          norm = im.scale)

        self.fig.colorbar(im_show,label = 'Normalized Intensity',ax=im_ax,location = 'bottom')

        im_ax.set_xlim(bounds['x'][0],bounds['x'][3])
        im_ax.set_ylim(bounds['y'][0],bounds['y'][3])
        im_ax.set_xticks([bounds['x'][0],bounds['x'][1],im.xcen,bounds['x'][2],bounds['x'][3]])
        im_ax.set_xticklabels([str(round((val - im.xcen)*im.pixscale/1000,2)) + '"' for val in im_ax.get_xticks()])
        im_ax.set_yticks([bounds['y'][0],bounds['y'][1],im.ycen,bounds['y'][2],bounds['y'][3]])
        im_ax.set_yticklabels([str(round((val - im.ycen)*im.pixscale/1000,2)) + '"' for val in im_ax.get_yticks()])
        im_ax.tick_params(axis='both', labelrotation=30, bottom=False, left=False)
        im_ax.set_xlabel('Relative RA')
        im_ax.set_ylabel('Relative Dec')
        # if im.imagery_type == 'line':
        #     im_type = "Line, Peak Intensity (Moment 8)"
        # else:
        #     im_type = im.imagery_type
        im_type = im.tag

        # ANNOTATIONS
        if self.annotate:
            # Plot circles at az profile radii
            for i, rad in enumerate(im.radlist):
                radius = int(rad)/im.pixscale
                if subfig == 'deprj_ims':
                    im_ax.add_patch(Circle((im.xcen, im.ycen), radius, color=self.az_colors[i], lw=2*self.img_mult, ls='--', alpha=0.5, fill=False))
                else:
                    im_ax.add_patch(Ellipse((im.xcen, im.ycen), 2 * radius, 2 * radius * math.cos(math.radians(im.incl)), angle=-im.PA, color=self.az_colors[i], lw=2*self.img_mult, ls='--', alpha=0.5, fill=False))

            # Plot rays at rad profile PAs
            for i, PA in enumerate(self.PAlist):
                PA = int(PA)
                im_ax.plot([im.xcen, im.xcen + math.sqrt(2) * boundsize * math.cos(math.radians(PA))], [im.ycen, im.ycen + math.sqrt(2) * boundsize * math.sin(math.radians(PA))],
                        color=self.rad_colors[i], lw=3, ls='--', alpha=0.5)

        # Create axis text
        if im.imagery_type == 'scattered light':
            stokes_text = {'Q_phi': '$\mathcal{Q}_\phi$', 'U_phi': '$\mathcal{U}_\phi$', 'Q': 'Q', 'U': 'U', 'I': 'I', 'LP_I': 'LP_I'}
            im_ax.text(0.05,0.9,f'{im.band} | {stokes_text[im.stokes_component]}', color='white', fontsize = 8*self.img_mult, transform=im_ax.transAxes)
        elif im.imagery_type == 'continuum':
            im_ax.text(0.05,0.9,f'{im.band}', color='white', fontsize = 8*self.img_mult, transform=im_ax.transAxes)

        # Make a scalebar
        if self.au_scale:
            pix_dist = im.pixscale / 1000 * self.distance # arcsec / pixel * parsec ~= au / pixel
            bar_length = (self.au_scale / pix_dist) / (2 * boundsize) # bar pixel width / total pixel width
            im_ax.plot([0.05,0.05+bar_length],[0.07,0.07],
                      lw=2*self.img_mult,color='white', transform=im_ax.transAxes)
            im_ax.text(0.04,0.1,f'{self.au_scale} au',color='white',fontsize=6*self.img_mult,ha='left',transform=im_ax.transAxes)

        # Set title
        im_ax.set_title(im.imagery_source + " " + im_type + "\n" + im.date)

#############################################################################

    def partition_list(self, lst):
        '''
        Takes a list of elements and returns all possible ways to partition the
        elements into 2 separate lists.
        (Thanks Gemini! <3 Even though your ads suck)
        
        Parameters
        ----------
        lst: list
            List to partition. (In this context, a list of subfigure tuples.)
        
        Returns
        -------
        result: list
            List containing all possible partitions of lst. Each element (tuple)
            is a unique partition.
        '''
        result = []
        n = len(lst)

        # Iterate over all possible bitmasks representing partitions
        for i in range(1, 2**(n-1)):
            left = []
            right = []
            for j in range(n):
                if (i >> j) & 1:
                  left.append(lst[j])
                else:
                  right.append(lst[j])
            result.append((left, right))

        return result

#############################################################################

    def optimize_cols(self, lst):
        '''
        Chooses the column pairing that minimizes the difference in heights
        between the two columns. (This was NOT worth the pain of planning...)

        Parameters
        ----------
        lst : list
            List of tuples. Each tuple represents a subfigure, storing the plot
            type (str) and subfigure height (float).

        Returns
        -------
        best_pair : list
            The partition of unzipped tuples for which the difference in
            the sums of the height tuples is the minimum for all partitions of lst.

        '''
        pairs = self.partition_list(lst)
        mindiff = len(self.hts)
        for pair in pairs:
            # Unzip the cols in the list
            cols = [list(zip(*pair[i])) for i in range(len(pair))]

            # Calculate difference of height sums
            diff = abs(sum(cols[0][1])-sum(cols[1][1]))

            # If this is the lowest diff so far:
            if diff < mindiff:
                mindiff = diff
                # Save as "best" pair
                best_pair = cols

        return best_pair

#############################################################################

    def make_plot_multiline(self):
        '''
        Generates a multipanel plot displaying disk imagery and/or intensity
        profiles for the disk. Types of panels, their order, and the choice of
        column layout are all user-controlled.
        
        The plot_types attribute determines the types of panels to display, and
        in what order (for two-column plots, order is preserved only within columns
        due to the optimization code). Refer to the `check_plot_types` docstring
        for supported types.
        '''

        # Determine number of subfigures
        self.n_rows = len(self.plot_types)

        # Get number of heatmaps
        heatmap_bools = [im.plot_heatmap for im in self.imlist]
        self.n_heatmaps = heatmap_bools.count(True)

        # Get beautiful colors for rings and lines :)
        ring_cmap = mpl.cm.get_cmap(self.ring_cmap)
        self.az_colors = ring_cmap(np.linspace(0,0.5,self.max_radlist_len))
        line_cmap = mpl.cm.get_cmap(self.line_cmap)
        self.rad_colors = line_cmap(np.linspace(0,0.5,self.max_PAlist_len))

        # Get list of height ratios
        self.hts = []
        for plot in self.plot_types:
            if 'ims' in plot:
                ht = self.img_mult * self.row_ht
            elif plot == 'az':
                # Calculate appropriate scaling factor
                self.az_n_cols = 1 if self.max_radlist_len < 4 else 2
                self.az_n_rows = math.ceil(self.max_radlist_len / self.az_n_cols)
                ht = self.az_n_rows * self.row_ht
            elif plot == 'rad':
                # Calculate appropriate scaling factor
                self.rad_n_cols = 1 if self.max_PAlist_len < 4 else 2
                self.rad_n_rows = math.ceil(self.max_PAlist_len / self.rad_n_cols)
                ht = self.rad_n_rows * self.row_ht
            elif plot == 'rad_az':
                ht = self.n_heatmaps * self.row_ht
            else: ht = self.row_ht
            self.hts.append(ht)

        # Set fig title string (there's probably a nicer way to do this...)
        if any(plot in self.plot_types for plot in ['rad', 'az']) and any(plot in self.plot_types for plot in ['rad_az', 'deprj_ims', 'or_ims']):
            self.fig_title = f'Multiwavelength Intensity Profiles and High-Resolution Imagery for {self.name}'
        else:
            if any(plot in self.plot_types for plot in ['rad', 'az']):
                self.fig_title = f'Multiwavelength Intensity Profiles for {self.name}'
            elif any(plot in self.plot_types for plot in ['rad_az', 'deprj_ims', 'or_ims']):
                self.fig_title = f'High-Resolution Imagery for {self.name}'

        # Set figure layout...
        from matplotlib.gridspec import GridSpec
        sb.set_theme(style = 'white', rc={"font.family" : "serif", "xtick.bottom" : True, "ytick.left" : True, "figure.titlesize":18}) #add ticks

        # Create the figure and GridSpec grid(s)
        from matplotlib.figure import Figure
        # Complex (2-column)
        if self.n_rows > 3 and not self.force_simple:
            self.complex_plot = True
            # Find the best column pairing!
            self.best_pair = self.optimize_cols(list(zip(self.plot_types,self.hts)))

            # Get the correct figsize
            sums = [sum(col[1]) for col in self.best_pair]
            mpl.rcParams.update({"figure.figsize":(2 * self.column_width, min(sums))})
            self.fig = Figure(layout='constrained')
            self.fig.suptitle(self.fig_title)

            # Make a gridspec
            self.gs = GridSpec(1, 2, figure=self.fig)

            # Make subfigs in each col
            self.subfigs = [self.fig.add_subfigure(self.gs[0, i]) for i in range(2)]

            # Make gss for each subfig
            self.subgss = [GridSpec(len(self.best_pair[i][1]), 1, figure=self.subfigs[i], height_ratios = self.best_pair[i][1]) for i in range(2)]

            # Create a dict to store plot location information because I love dicts
            self.plot_locs = {}
            arranged_plot_types = [self.best_pair[i][0] for i in range(len(self.best_pair))]
            for col in arranged_plot_types:
                for plot in col:
                    idx1 = arranged_plot_types.index(col)
                    idx2 = col.index(plot)
                    facecolor = 'whitesmoke' if (idx1 + idx2) % 2 == 0 else 'white'
                    self.plot_locs[plot] = {'subfig': self.subfigs[idx1], 'subgsrow': self.subgss[idx1][idx2], 'facecolor': facecolor}

        # Simple (1-column)
        else:
            self.complex_plot = False
            mpl.rcParams.update({"figure.figsize":(self.column_width, sum(self.hts))})
            self.fig = Figure(layout='constrained')
            self.fig.suptitle(self.fig_title)
            self.gs = GridSpec(self.n_rows, 1, figure=self.fig, height_ratios=self.hts)

        # Create subfigs according to specified plot types
        axs = {}
        for index, plot in enumerate(self.plot_types):
            if self.complex_plot:
                plot_loc = self.plot_locs[plot]
                axs[plot] = plot_loc['subfig'].add_subfigure(plot_loc['subgsrow'], facecolor=plot_loc['facecolor'])
            else: axs[plot] = self.fig.add_subfigure(self.gs[index], facecolor='whitesmoke' if index % 2 == 0 else 'white')

            # Make suptitles
            suptitles = {'avgrad': 'Azimuthally Averaged Radial Profiles',
                         'rad': 'Radial Profiles at Given PAs',
                         'az': 'Azimuthal Profiles',
                         'rad_az': 'Polar Projections',
                         'deprj_ims': 'Deprojected Imagery',
                         'or_ims': 'On Sky'}
            axs[plot].suptitle(suptitles[plot])

            # Make subplots on a case-by-case basis
            if 'ims' in plot:
                # Make subplots
                axs[plot] = axs[plot].subplots(1, len(self.imlist))
            elif plot == 'az':
                self.az_subfig = axs[plot]
                # Make appropriate gridspec
                self.az_gs = GridSpec(self.az_n_rows, self.az_n_cols, figure=self.az_subfig)
                # Make subplots
                axs[plot] = list(range(self.max_radlist_len))
                for i in range(self.max_radlist_len):
                    axs[plot][i] = self.az_subfig.add_subplot(self.az_n_rows, self.az_n_cols, i + 1)
                # If length of radlist is odd and >4
                if self.max_radlist_len > 4 and self.max_radlist_len % 2 != 0:
                    # Stretch out the final subplot
                    axs[plot][self.max_radlist_len - 1].remove()
                    axs[plot][self.max_radlist_len - 1] = self.az_subfig.add_subplot(self.az_gs[self.az_n_rows - 1,:])
            elif plot == 'rad': # Probably inelegant to just paste the az code 7/29/24
                self.rad_subfig = axs[plot]
                # Make appropriate gridspec
                self.rad_gs = GridSpec(self.rad_n_rows, self.rad_n_cols, figure=self.rad_subfig)
                # Make subplots
                axs[plot] = list(range(self.max_PAlist_len))
                for i in range(self.max_PAlist_len):
                    axs[plot][i] = self.rad_subfig.add_subplot(self.rad_n_rows, self.rad_n_cols, i + 1)
                # If length of PAlist is odd and >4
                if self.max_PAlist_len > 4 and self.PA_radlist_len % 2 != 0:
                    # Stretch out the final subplot
                    axs[plot][self.max_PAlist_len - 1].remove()
                    axs[plot][self.max_PAlist_len - 1] = self.rad_subfig.add_subplot(self.rad_gs[self.rad_n_rows - 1,:])
            elif plot == 'rad_az':
                # Make subplots
                axs[plot] = axs[plot].subplots(self.n_heatmaps, 1)
            else:
                # Make subplots
                axs[plot] = axs[plot].subplots()

        # Calculate boundsize. This is the largest width of the images
        # so that all images can be put on this scale.
        self.boundsize = 0
        for index, im in enumerate(self.imlist):
            im_boundsize = im.auto_bound * im.pixscale
            if im_boundsize > self.boundsize: self.boundsize = im_boundsize
        # Bounds for radial profiles - the min mininum and the max maximum radii
        self.min_rad = min([min(im.rad_plot_data['Radius (mas)'].tolist()) for im in self.imlist])
        self.max_rad = max([max(im.rad_plot_data['Radius (mas)'].tolist()) for im in self.imlist])

        # Plot the images!
        for index, im in enumerate(self.imlist):

            # IMAGE PLOT CODE
            for subfig in [plot for plot in self.plot_types if 'ims' in plot]:
                # Assign the correct image instance to plot. Copy of instance uses projected data.
                data_im = im if subfig == 'deprj_ims' else copy(im)
                self.plot_im(data_im, axs[subfig][index], subfig)

            # AVGRAD PLOT CODE
            if 'avgrad' in self.plot_types:
                sb.lineplot(data = im.rad_plot_data.loc[im.rad_plot_data['PA'] == 'mean', :],
                            x = 'Radius (mas)', y='flux',  marker = 'o', color = im.color,
                            label = im.tag, ax=axs['avgrad'])
                axs['avgrad'].xaxis.set_minor_locator(MultipleLocator(20))
                axs['avgrad'].yaxis.set_minor_locator(MultipleLocator(0.1))
                axs['avgrad'].grid(True)
                axs['avgrad'].set(xlabel='Radius (mas)', ylabel= 'Normalized Intensity', xlim=(self.min_rad - 0.1, self.max_rad + 0.1), ylim = (0,1))
                axs['avgrad'].legend(loc='upper right', title='Tracer')

            # RAD PLOT CODE
            if 'rad' in self.plot_types:
                for i in range(self.max_PAlist_len):
                    rad_ax = axs['rad'][i]
                    rad_data = im.rad_plot_data[im.rad_plot_data['PA'] == int(self.PAlist[i])]
                    sb.lineplot(data = rad_data,
                                x = 'Radius (mas)', y='flux',  marker = 'o', color = im.color,
                                label = im.tag, ax=rad_ax)
                    rad_ax.set(xlabel='Radius (mas)', ylabel= 'Normalized Intensity', xlim=(self.min_rad - 0.1, self.max_rad + 0.1), ylim = (0,1))
                    rad_ax.xaxis.set_minor_locator(MultipleLocator(20))
                    rad_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                    rad_ax.grid(True)
                    if self.rad_lines:
                        rad_ax.text(0.5,1.1,f'PA = {self.PAlist[i]} deg',
                                   horizontalalignment = 'center', verticalalignment='top',
                                   color = self.rad_colors[i], transform=rad_ax.transAxes)
                        #was originally just text, but I centered it, so it's basically a suptitle... might change to that. 7/29/24
                    else: rad_ax.set_title(f'PA = {self.PAlist[i]} deg')
                    rad_ax.get_legend().remove()

            #AZ PLOT CODE
            if 'az' in self.plot_types:
                for i in range(self.max_radlist_len):
                    az_ax = axs['az'][i]
                    az_data = im.az_plot_data[im.az_plot_data['Radius (mas)'] == im.radlist[i]]
                    az_data.drop(az_data.tail(2).index,inplace=True) # drop the extra PA and PA + 90
                    sb.lineplot(data = az_data,
                                x = 'PA', y='flux',  marker = 'o', color = im.color,
                                label = im.tag, ax=az_ax)
                    #locator_params isn't working for some reason, so setting the ticks manually... 7/19/24
                    az_ax.set_xticks(np.arange(0,360,45))
                    az_ax.xaxis.set_minor_locator(MultipleLocator(15))
                    az_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                    az_ax.grid(True)
                    # set title, axs
                    az_ax.set(xlabel='PA (deg)', ylabel= 'Normalized Intensity', xlim=(-0.1, 354.9), ylim = (0,1))
                    if self.az_rings:
                        az_ax.text(0.5,1.1,f'Radius = {im.radlist[i]} mas',
                                   horizontalalignment = 'center', verticalalignment='top',
                                   color = self.az_colors[i], transform=az_ax.transAxes)
                        #was originally just text, but I centered it, so it's basically a suptitle... might change to that. 7/29/24
                    else: az_ax.set_title(f'Radius = {im.radlist[i]} mas')
                    az_ax.get_legend().remove()

            # HEATMAPS
            if 'rad_az' in self.plot_types and im.plot_heatmap:
                rad_az_ax = axs['rad_az'] if isinstance(axs['rad_az'], mpl.axes.Axes) else axs['rad_az'][index]
                im.rad_az_plot_data = im.radial_profiles.drop([str(im.PA),str(im.PA+90), 'mean'], axis = 1).astype('float')
                im.rad_az_plot_data['Radius (mas)'] = ([str(round(float(val))) for val in im.rad_az_plot_data['Radius (mas)'].values])
                im.rad_az_plot_data = im.rad_az_plot_data.set_index('Radius (mas)')
                sb.heatmap(im.rad_az_plot_data/np.nanmax(im.rad_az_plot_data),
                          cmap = im.cmap, cbar_kws= {'pad':0.001, 'label':'Normalized Intensity'},
                          norm = im.scale, ax = rad_az_ax)

                rad_az_ax.locator_params(axis='both', nbins=8)
                nPAs = len(im.PAs) - 2 - len(self.PAlist)
                rad_az_ax.set_xticks(np.arange(0,nPAs,nPAs/8))
                rad_az_ax.set_xticklabels(np.arange(0,360,45))
                rad_az_ax.xaxis.set_minor_locator(MultipleLocator(3))
                rad_az_ax.tick_params(axis='both', labelrotation=0)

                rad_az_ax.set(title = im.imagery_source + " " + im.tag, xlabel='PA (deg)', ylabel= 'Radius (mas)')

        for plot in self.plot_types:
            if plot in ('rad', 'az'):
                # put the legend in the first plot (for now)
                axs[plot][0].legend(title='Tag', loc='upper right')


#############################################################

class AstroImage():
    '''
    Class to represent a disk image. Contains required parameters to process,
    deproject, construct profiles, and eventually produce plots for the image.
    
    Attributes
    ----------
    obj : str
        Disk name.
    imagery_type : str
        Image tracer (scattered light, mm continuum, line emission).
    imagery_source : str
        Instrument with which the image was taken.
    path : str
        Path of fits file for the image.
    trim : float or None
        Amount by which to trim line emission data. Irrelevant for now (v1.0).
    r2: bool
        Whether to r2-scale the images. Recommended for scattered light imagery.
    stokes_component: bool
        Stokes component of the fits file to use for polarimetric data.
    override_center: list or None
        Center coordinates to manually set for an image if the center information
        from the header is incorrect. Usually needed for ALMA data.
    plot_heatmap : bool
        Whether to plot a polar projection of the imagery in the multiline plot.
    interval : list
        Colorbar bounds.
    cmap : str
        Name of cmap to use for image and polar map plotting for this AstroImage
        in the multipanel plot.
    color : str
        Name of color to use for profile lineplots for this AstroImage in the
        multipanel plot.
    fits_table : fits.hdu.hdulist.HDUList
        Fits file data, opened from the fits file path.
    header : fits.header.Header
        Fits file header information.
    PA : int
        Position angle of the disk.
    incl : int
        Inclination of the disk.
    PAs : list
        List of PAs to loop through during radial and azimuthal profile generation.
    radlist : list
        Radii for which to generate and plot azimuthal profiles in the multiline plot.
    
    
    Methods
    -------
    strip_header():
        Extracts relevant information about the disk image from the fits file
        header.
    deproject(r2=False):
        Deprojects the image data using PA and incl.
    pa_rotate():
        Rotates image by each PA in PAs and returns a list of all rotated images
        for radial and azimuthal profile generation.
    make_profiles():
        Constructs radial and azimuthal profiles for the image.
    plot_prep():
        Maps radial and azimuthal profile data into a format suitable for 
        plotting in AstroObject.make_plot_multiline() by melting the dataframes
        produced in make_profiles().
    profiles():
        Wrapper function that calls all the above functions (except for
        strip_header()) in order. Called by AstroObject.__init__().
    
    History
    -------
    Original author: Kate Follette in IDL.
    Revised by Cat Sarosi (2022).
    Revised further by Bibi Hanselman (2024). Added descriptive log messages,
    cleaned up unnecessary code, further wrangled the data in light of changes
    to AstroObject.make_plot_multiline, and removed individual image plotting
    functionality as it's now effectively covered among the possible multiline
    plot display options.
    '''
    def __init__(self,image_dict=None):
        '''
        Initializes required image attributes for deprojection and profile
        creation.

        Parameters
        ----------
        image_dict : dict, optional
            Dict pulled from sheets interface that contains values for
            each of the image attributes. The default is None.

        '''
        if image_dict:
            # Essential image information
            self.obj = image_dict['Name']
            self.imagery_type = image_dict['Tracer']
            self.tag = image_dict['Tag']
            self.imagery_source = image_dict['Source']
            self.pixscale = image_dict['Pixel Scale']
            self.px_bin = image_dict['Pixel Bin']
            self.band = image_dict['Band']
            self.data = image_dict['Data']
            
            # Tracer-specific information
            self.r2 = self.imagery_type == 'scattered light'
            # once again, r2 will be hard-coded until I implement an r2 correction toggle (and figure out how to r2 correct an on sky image...) 1/2/25
            self.stokes_component = image_dict['Stokes Component']
            
            # Center coordinates
            self.ycen, self.xcen = [image_dict['Crop'], image_dict['Crop']]
            self.crop = int(image_dict['Crop']/2)
            
            # Option to plot heatmap (polar map)
            self.plot_heatmap = True # Will be true for all images, for now... 1/2/25
            
            # Scaling considerations
            self.interval = image_dict['Colorbar']
            self.scale = image_dict['Scale']
            
            # Color settings
            self.cmap = image_dict['Color']
            self.color = image_dict['Line Color']

##############################################################################

    def __copy__(self):
        '''
        Copies this AstroImage while reassigning the data and ycen attributes
        to their values before deprojection. Used to plot on sky (projected)
        imagery.

        Returns
        -------
        or_im : AstroImage
            Identical image object, but containing projected data.

        '''
        or_im = AstroImage()
        or_im.__dict__.update(self.__dict__)
        or_im.data = self.or_im
        or_im.ycen = self.or_ycen
        or_im.bounds = deepcopy(self.bounds)
        or_im.bounds['y'] = [or_im.ycen - self.auto_bound, or_im.ycen - self.auto_bound/2, or_im.ycen + self.auto_bound/2, or_im.ycen + self.auto_bound]
        return or_im

##############################################################################
    def deproject(self):
        '''
        Deprojects the image data using disk PA and incl.
        
        Parameters
        ----------
        r2 : bool, optional
            Whether to r2-scale the data. The default is False.
        
        '''
        ndimx,ndimy = self.data.shape[1],self.data.shape[0]

        M = cv2.getRotationMatrix2D((self.xcen,self.ycen), self.PA - 90, 1.0)

        imrot = cv2.warpAffine(self.data, M,
                               (self.data.shape[0], self.data.shape[1]),
                               flags = cv2.INTER_CUBIC)

        #stretch image in y by cos(incl) to deproject and divide by that factor to preserve flux
        im_rebin = cv2.resize(imrot,
                              (ndimx,int(np.round(ndimy*(1/math.cos(math.radians(self.incl)))))),
                              interpolation = cv2.INTER_CUBIC)
        im_rebin = im_rebin/(1./math.cos(math.radians(self.incl)))

        ndimy2 = im_rebin.shape[0]
        ycen2 = int(round((ndimy2/ndimy)*self.ycen))

        if self.r2:
            print('Applying r2 correction...')
            r = np.zeros((ndimy2,ndimx))
            for x in np.arange(0, ndimx-1):
                for y in np.arange(0, ndimy2-1):
                    r[y,x]=np.sqrt((x-self.xcen)**2+(y-ycen2)**2)
                    im_rebin[y,x]=im_rebin[y,x]*(r[y,x])**2
            print('Applied r2 correction')

        #rotate back to original orientation
        M = cv2.getRotationMatrix2D((self.xcen, ycen2), -1*(self.PA - 90), 1.0)

        im_rebin_rot = cv2.warpAffine(im_rebin, M,
                                      (ndimx, ndimy2),
                                      flags = cv2.INTER_CUBIC)

        self.data = im_rebin_rot
        self.ycen = ycen2

##############################################################################
    def pa_rotate(self):
        '''
        Rotates image by each PA in PAs and returns a list of all rotated images
        (file_rot) for radial and azimuthal profile construction. (Not using for now
        to save memory, but will keep here just in case.)
        '''
        print('Rotating over PAs...')
        shape = (self.data.shape[0],self.data.shape[1])
        nPAs = len(self.PAs)
        file_rot = np.zeros((nPAs,shape[0],shape[1]))

        for i in np.arange(0, nPAs):
            pa = -self.PAs[i] + 90
            if pa == 0:
                file_rot[i] = self.data
            else:
                M = cv2.getRotationMatrix2D((self.xcen, self.ycen), pa, 1.0)
                file_rot[i] = cv2.warpAffine(self.data,
                                             M,
                                             (shape[1], shape[0]),
                                             flags =    cv2.INTER_CUBIC)
        self.file_rot = file_rot
##############################################################################
    def make_profiles(self):
        '''
        Constructs radial and azimuthal profiles for the image.
        '''
        radial_profiles = pd.DataFrame(columns=[str(PA) for PA in self.PAs])
        azimuthal_profiles = pd.DataFrame(columns=[str(PA) for PA in self.PAs])
        shape = self.data.shape
        print(f'Shape: {shape}')
        halfx = int(shape[1]/2)
        halfy = int(shape[0]/2)
        half_px_bin = int(self.px_bin//2)

        PA_mask = np.ones(shape, dtype=bool)
        PA_mask[halfy:shape[0],halfx-half_px_bin:halfx+half_px_bin+1] = False

        print('Generating radial profiles...')
        # Radial profile loop
        for radcut in np.arange(0,int(shape[0]/2/self.px_bin)):
            r_in,r_out = (radcut * self.px_bin, (radcut+1) * self.px_bin)
            ap = (phot.CircularAnnulus([self.xcen,self.ycen], r_in, r_out)) if radcut > 0 else(phot.CircularAperture([self.xcen,self.ycen],r_out))

            for pa in self.PAs:
                PA = 90 - pa
                if PA == 0:
                    file_rot = self.data
                else:
                    M = cv2.getRotationMatrix2D((self.xcen, self.ycen), PA, 1.0)
                    file_rot = cv2.warpAffine(self.data,
                                                M,
                                                (shape[1], shape[0]),
                                                flags =    cv2.INTER_CUBIC)
                radial_profiles.loc[radcut, str(pa)] = phot.ApertureStats(file_rot,
                                                                          ap, sum_method = 'center',
                                                                          mask = PA_mask).median #mask = PA_mask + np.where(self.file_rot[panum]<0, True,False)).mean

                #probably WRONG??!?!?! Should be looping over more angles at larger radii
                #plot masked image

        print('Generating azimuthal profiles...')
        # Az profile loop
        for pa in self.PAs:
            PA = 90 - pa
            if PA == 0:
                file_rot = self.data
            else:
                M = cv2.getRotationMatrix2D((self.xcen, self.ycen), PA, 1.0)
                file_rot = cv2.warpAffine(self.data,
                                            M,
                                            (shape[1], shape[0]),
                                            flags =    cv2.INTER_CUBIC)
            for rad in self.radlist:
                rad = int(rad)
                r_in,r_out = (int(rad/self.pixscale - half_px_bin),int(rad/self.pixscale + half_px_bin + 1))
                ap = phot.CircularAnnulus([self.xcen,self.ycen],r_in,r_out) if r_in > 0 else phot.CircularAperture([self.xcen,self.ycen],r_out)
                azimuthal_profiles.loc[str(rad), str(pa)] = phot.ApertureStats(file_rot,
                                                                    ap, sum_method = 'center',
                                                                    mask = PA_mask).median  #mask = PA_mask + np.where(self.file_rot[panum]<0, True,False)).sum

        self.radial_profiles = radial_profiles
        self.azimuthal_profiles = azimuthal_profiles
        self.n_cuts = radcut + 1
##############################################################################
    def plot_prep(self):
        '''
        Maps radial and azimuthal profile data into a format suitable for 
        plotting in AstroObject.make_plot_multiline() by melting the dataframes
        produced in make_profiles().
        '''
        self.auto_bound = self.n_cuts * self.px_bin
        self.bounds = {}
        self.bounds['x'] = [self.xcen - self.auto_bound, self.xcen - self.auto_bound/2, self.xcen + self.auto_bound/2, self.xcen + self.auto_bound]
        self.bounds['y'] = [self.ycen - self.auto_bound, self.ycen - self.auto_bound/2, self.ycen + self.auto_bound/2, self.ycen + self.auto_bound]

        #make x axis in arcsec
        self.radial_profiles['Radius (mas)'] = np.arange(0,self.n_cuts) * self.pixscale * self.px_bin + self.pixscale * self.px_bin/2
        self.radial_profiles['mean'] = self.radial_profiles.loc[:, ~self.radial_profiles.columns.isin([str(self.PA), str(self.PA+90),'Radius (mas)'])].mean(axis= 1,skipna=True)
        rad_plot_data = pd.melt(self.radial_profiles, id_vars=['Radius (mas)'], var_name='PA', value_name = 'flux')
        rad_plot_data['flux'] = rad_plot_data['flux']/np.nanmax(rad_plot_data['flux'])
        
        # Adjust the PAs so that they're oriented correctly (CCW)
        PAs = rad_plot_data['PA'].drop(rad_plot_data[rad_plot_data['PA'] == 'mean'].index).astype(int)
        PAs = 180 - PAs
        PAs[PAs < 0] += 360
        rad_plot_data['PA'].loc[rad_plot_data['PA'] != 'mean'] = PAs

        self.azimuthal_profiles.loc[-1] = self.radial_profiles.loc[:, ~self.radial_profiles.columns.isin([str(self.PA), str(self.PA+90),'Radius (mas)'])].median(axis= 0,skipna=True)
        self.azimuthal_profiles.rename(index={-1:'median'},inplace=True)
        self.azimuthal_profiles['Radius (mas)'] = self.azimuthal_profiles.index
        az_plot_data = pd.melt(self.azimuthal_profiles, id_vars=['Radius (mas)'], var_name='PA', value_name = 'flux')
        az_plot_data['flux'] = az_plot_data['flux']/np.nanmax(az_plot_data['flux'])

        # Adjust the PAs so that they're oriented correctly (CCW)
        PAs = az_plot_data['PA'].astype(int)
        PAs = 180 - PAs
        PAs[PAs < 0] += 360
        az_plot_data['PA'] = PAs

        self.rad_plot_data = rad_plot_data
        self.az_plot_data = az_plot_data
##############################################################################
    def profiles(self):
        '''
        Calls all of the above functions in order.
        '''
        print(f'Current image: {self.obj} {self.imagery_type}')

        self.shape = (self.data.shape[1],self.data.shape[0])
        self.nPAs = len(self.PAs)

        # Save original data plus centers before deprojection
        self.or_im, self.or_ycen = self.data, self.ycen
        self.deproject()
        
        # Crop the new image
        # self.data = self.data[self.ycen - self.crop:self.ycen + self.crop,self.xcen - self.crop:self.xcen + self.crop]

        # Profile construction
        #self.pa_rotate()
        self.make_profiles()
        self.plot_prep()

        print(f'{self.obj} {self.imagery_type} done!')