import bettermoments as bm
from astropy.io import fits
import numpy as np
import skimage
import skimage.transform
import cv2
import imutils
import matplotlib.pyplot as plt
import math
import seaborn as sb
import os
import glob as glob
import pandas as pd
import astropy.units as u
import photutils.aperture as phot
from matplotlib.patches import Circle, Ellipse
from copy import copy, deepcopy
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import figg

class AstroObject():
    def __init__(self, name, image_dict, display_dict):

        self.name = name

        # Instantiate AstroImages
        self.scattered_light = AstroImage(image_dict['scattered light']) if 'scattered light' in image_dict else None
        self.sum_mm = AstroImage(image_dict['continuum']) if 'continuum' in image_dict else None
        self.line = AstroImage(image_dict['line']) if 'line' in image_dict else None

        self.imlist = [im for im in [self.scattered_light,self.sum_mm,self.line] if im is not None]

        # Assign certain disk-wide parameters to the images
        for im in self.imlist:
            im.incl = display_dict['Inclination']
            im.PA = display_dict['PA']

            im.PAs = np.arange(0,360,display_dict['PA Step']).tolist()
            im.PAs.append(im.PA)
            im.PAs.append(im.PA + 90)

            im.radlist = [str(display_dict['Radius List'])] if isinstance(display_dict['Radius List'],int) else display_dict['Radius List']
            if im.radlist is None: im.radlist = []
            im.PAlist = [str(display_dict['PA List'])] if isinstance(display_dict['PA List'],int) else display_dict['PA List']
            if im.PAlist is None: im.PAlist = []

        # Set display info
        self.update(display_dict)

        # Store image data in img_data_dict
        self.img_data_dict = {}
        for im in self.imlist: self.img_data_dict[im.imagery_type] = im.data

        # Process the data for the disk (also saves image parameters in figg module for future executions)
        self.prc_ims, self.prc_scales = figg.process(image_dict, name = self.name, custom_ims=self.img_data_dict)

        # Assign processed ims to im.data and create the profiles
        for im in self.imlist:
          im.data = self.prc_ims[im.imagery_type]
          im.profiles()

#############################################################################

    def update(self, display_dict):
        '''
        Updates display-related values in the disk dictionary.
        To update image processing parameters (PA, inclination, PAstep, PA list, radius list),
        you must reinstantiate the AstroObject to reprocess the imagery and profiles.
        '''
        self.plot_types = display_dict['Plot Types']
        self.rad_lines = display_dict['Rad Lines']
        self.az_rings = display_dict['Az Rings']
        self.line_cmap = display_dict['Line CMap']
        self.ring_cmap = display_dict['Ring CMap']
        self.column_width = display_dict['Column Width']
        self.row_ht = display_dict['Row Height']
        self.img_mult = display_dict['Image Multiplier']
        self.force_simple = display_dict['Force Simple']
        self.au_scale = display_dict['AU Scale']
        self.distance = display_dict['Distance']

#############################################################################

    def plot_im(self, im, im_ax, subfig):
        '''
        Lil' helper function to plot an image.
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
                          norm = self.prc_scales[im.imagery_type])

        plt.colorbar(im_show,label = 'Normalized Intensity',ax=im_ax,location = 'bottom')

        im_ax.set_xlim(bounds['x'][0],bounds['x'][3])
        im_ax.set_ylim(bounds['y'][0],bounds['y'][3])
        im_ax.set_xticks([bounds['x'][0],bounds['x'][1],im.xcen,bounds['x'][2],bounds['x'][3]])
        im_ax.set_xticklabels([str(round((val - im.xcen)*im.pixscale/1000,2)) + '"' for val in im_ax.get_xticks()])
        im_ax.set_yticks([bounds['y'][0],bounds['y'][1],im.ycen,bounds['y'][2],bounds['y'][3]])
        im_ax.set_yticklabels([str(round((val - im.ycen)*im.pixscale/1000,2)) + '"' for val in im_ax.get_yticks()])
        im_ax.tick_params(axis='both', labelrotation=30, bottom=False, left=False)
        im_ax.set_xlabel('Relative RA')
        im_ax.set_ylabel('Relative Dec')
        if im.imagery_type == 'line':
            im_type = "Line, Peak Intensity (Moment 8)"
        else:
            im_type = im.imagery_type

        # Plot circles at az profile radii
        if self.az_rings:
            for i, rad in enumerate(im.radlist):
                radius = int(rad)/im.pixscale
                if subfig == 'deprj_ims':
                    im_ax.add_patch(Circle((im.xcen, im.ycen), radius, color=self.az_colors[i], lw=2*self.img_mult, ls='--', alpha=0.5, fill=False))
                else:
                    im_ax.add_patch(Ellipse((im.xcen, im.ycen), 2 * radius, 2 * radius * math.cos(math.radians(im.incl)), -im.PA, color=self.az_colors[i], lw=2*self.img_mult, ls='--', alpha=0.5, fill=False))

        # Plot rays at rad profile PAs
        if self.rad_lines:
            for i, PA in enumerate(im.PAlist):
                PA = int(PA)
                im_ax.plot([im.xcen, im.xcen + math.sqrt(2) * boundsize * -math.cos(math.radians(PA))], [im.ycen, im.ycen + math.sqrt(2) * boundsize * math.sin(math.radians(PA))],
                          color=self.rad_colors[i], lw=3, ls='--', alpha=0.5)

        # Create axis text
        if im.imagery_type == 'scattered light':
            stokes_text = {'Q_phi': '$\mathcal{Q}_\phi$', 'U_phi': '$\mathcal{U}_\phi$', 'Q': 'Q', 'U': 'U'}
            im_ax.text(0.05,0.9,f'{im.band} | {stokes_text[im.stokes_component]}', color='white', fontsize = 8*self.img_mult, transform=im_ax.transAxes)
        elif im.imagery_type == 'continuum':
            im_ax.text(0.05,0.9,f'{im.band}', color='white', fontsize = 8*self.img_mult, transform=im_ax.transAxes)

        # Make a scalebar
        pix_dist = im.pixscale / 1000 * self.distance # arcsec / pixel * parsec ~= au / pixel
        bar_length = (self.au_scale / pix_dist) / (2 * boundsize) # bar pixel width / total pixel width
        im_ax.plot([0.05,0.05+bar_length],[0.07,0.07],
                  lw=2*self.img_mult,color='white', transform=im_ax.transAxes)
        im_ax.text(0.04,0.1,f'{self.au_scale} AU',color='white',fontsize=6*self.img_mult,ha='left',transform=im_ax.transAxes)

        # Set title
        im_ax.set_title(im.imagery_source + " " + im_type + "\n" + im.date)

#############################################################################

    def check_plot_types(self):
        '''
        Check if the inputted plot types are valid. The supported types and corresponding strings are:
            'avgrad': Azimuthally averaged radial profiles.
            'rad': Radial profiles.
            'az': Azimuthal profiles.
            'rad_az': Polar projections.
            'deprj_ims': Deprojected imagery.
            'or_ims': Original (projected) imagery.
        '''
        valid_plot_types = ['avgrad','rad','az','rad_az','deprj_ims','or_ims']
        for plot in self.plot_types:
            if plot not in valid_plot_types:
                del self.plot_types[self.plot_types.index(plot)]
                print(f"'{plot}' is not a supported subplot type. Removing from astro_objects['{self.name}'].plot_types.")

        # Check if 'az' should be removed
        # Get longest rad list and length
        radlists = [im.radlist for im in self.imlist]
        self.max_radlist_len = len(max(radlists, key=len))
        # If the longest length is 0...
        if 'az' in self.plot_types and self.max_radlist_len == 0:
            del self.plot_types[self.plot_types.index('az')]
            print(f"No radlists were given for any {self.name} imagery. Removing 'az' from astro_objects['{self.name}'].plot_types.")

        # Check if 'rad' should be removed
        # Get longest PA list and length
        PAlists = [im.PAlist for im in self.imlist]
        self.max_PAlist_len = len(max(PAlists, key=len))
        # If the longest length is 0...
        if 'rad' in self.plot_types and self.max_PAlist_len == 0:
            del self.plot_types[self.plot_types.index('rad')]
            print(f"No PAlists were given for any {self.name} imagery. Removing 'rad' from astro_objects['{self.name}'].plot_types.")

        if len(self.plot_types) == 0: # If there were no valid types
            self.plot_types = ['rad'] # Default setting. At least plot something...

#############################################################################

    def partition_list(self, lst):
        '''
        Takes a list of elements and returns all possible ways to partition the elements into 2 separate lists. (Thanks Gemini! <3)
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
        profiles

        PURPOSE:
            #Calculate radial profile for a  disk at given PAs

        INPUTS:
           #filename:[str] description

        AUTHOR:
            Original in IDL: Kate Follette
            Revised in Python: Catherine Sarosi Nov 30, 2022
        '''
        # First, check if self.plot_types contains any invalid types
        self.check_plot_types()

        # Determine number of subfigures
        self.n_rows = len(self.plot_types)

        # Get number of heatmaps
        heatmap_bools = [im.plotheatmap for im in self.imlist]
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
        sb.set(style = 'white', rc={"xtick.bottom" : True, "ytick.left" : True, "figure.titlesize":18}) #add ticks

        # Create the figure and GridSpec grid(s)
        # Complex (2-column)
        if (self.max_radlist_len > 1 or self.n_heatmaps > 1) and not self.force_simple:
            self.complex_plot = True
            # Find the best column pairing!
            self.best_pair = self.optimize_cols(list(zip(self.plot_types,self.hts)))

            # Get the correct figsize
            sums = [sum(col[1]) for col in self.best_pair]
            plt.rcParams.update({"figure.figsize":(2 * self.column_width, min(sums))})
            self.fig = plt.figure(constrained_layout=True)
            plt.suptitle(self.fig_title)

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
            plt.rcParams.update({"figure.figsize":(self.column_width, sum(self.hts))})
            self.fig = plt.figure(constrained_layout=True)
            plt.suptitle(self.fig_title)
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
                         'or_ims': 'Original Imagery'}
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
        self.min_rad = min([min(im.rad_plot_data['xaxis'].tolist()) for im in self.imlist])
        self.max_rad = max([max(im.rad_plot_data['xaxis'].tolist()) for im in self.imlist])

        # Plot the images!
        for index, im in enumerate(self.imlist):
            nPAs = len(im.PAs) - 2

            # IMAGE PLOT CODE
            for subfig in [plot for plot in self.plot_types if 'ims' in plot]:
                # Assign the correct image instance to plot. Copy of instance uses projected data.
                data_im = im if subfig == 'deprj_ims' else copy(im)
                self.plot_im(data_im, axs[subfig][index], subfig)

            # AVGRAD PLOT CODE
            if 'avgrad' in self.plot_types:
                sb.lineplot(data = im.rad_plot_data.loc[im.rad_plot_data['PA'] == 'mean', :],
                            x = 'xaxis', y='flux',  marker = 'o', color = im.color,
                            label = im.imagery_type, ax=axs['avgrad'])
                axs['avgrad'].xaxis.set_minor_locator(MultipleLocator(20))
                axs['avgrad'].yaxis.set_minor_locator(MultipleLocator(0.1))
                axs['avgrad'].grid(True)
                axs['avgrad'].set(xlabel='Radius (mas)', ylabel= 'Normalized Intensity', xlim=(self.min_rad - 0.1, self.max_rad + 0.1), ylim = (0,1))
                axs['avgrad'].legend(loc='upper right', title='Tracer')

            # RAD PLOT CODE
            if 'rad' in self.plot_types:
                for i in range(self.max_PAlist_len):
                    rad_ax = axs['rad'][i]
                    rad_data = im.rad_plot_data[im.rad_plot_data['PA'] == im.PAlist[i]]
                    sb.lineplot(data = rad_data,
                                x = 'xaxis', y='flux',  marker = 'o', color = im.color,
                                label = im.imagery_type, ax=rad_ax)
                    rad_ax.set(xlabel='Radius (mas)', ylabel= 'Normalized Intensity', xlim=(self.min_rad - 0.1, self.max_rad + 0.1), ylim = (0,1))
                    rad_ax.xaxis.set_minor_locator(MultipleLocator(20))
                    rad_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                    rad_ax.grid(True)
                    if self.rad_lines:
                        rad_ax.text(0.5,1.1,f'PA = {im.PAlist[i]} deg',
                                   horizontalalignment = 'center', verticalalignment='top',
                                   color = self.rad_colors[i], transform=rad_ax.transAxes)
                        #was originally just text, but I centered it, so it's basically a suptitle... might change to that. 7/29/24
                    else: rad_ax.set_title(f'PA = {im.PAlist[i]} deg')
                    rad_ax.get_legend().remove()

            #AZ PLOT CODE
            if 'az' in self.plot_types:
                for i in range(self.max_radlist_len):
                    az_ax = axs['az'][i]
                    az_data = im.az_plot_data[im.az_plot_data['Radius (mas)'] == im.radlist[i]]
                    az_data.drop(az_data.tail(2).index,inplace=True) # drop the extra PA and PA + 90
                    sb.lineplot(data = az_data,
                                x = 'PA', y='flux',  marker = 'o', color = im.color,
                                label = im.imagery_type, ax=az_ax)
                    #locator_params isn't working for some reason, so setting the ticks manually... 7/19/24
                    az_ax.set_xticks(np.arange(0,nPAs,nPAs/8))
                    az_ax.set_xticklabels(np.arange(0,360,45))
                    az_ax.xaxis.set_minor_locator(MultipleLocator(3))
                    az_ax.yaxis.set_minor_locator(MultipleLocator(0.1))
                    az_ax.grid(True)
                    # set title, axs
                    az_ax.set(xlabel='PA (deg)', ylabel= 'Normalized Intensity', xlim=(-0.1, nPAs - 0.9), ylim = (0,1))
                    if self.az_rings:
                        az_ax.text(0.5,1.1,f'Radius = {im.radlist[i]} mas',
                                   horizontalalignment = 'center', verticalalignment='top',
                                   color = self.az_colors[i], transform=az_ax.transAxes)
                        #was originally just text, but I centered it, so it's basically a suptitle... might change to that. 7/29/24
                    else: az_ax.set_title(f'Radius = {im.radlist[i]} mas')
                    az_ax.get_legend().remove()

            # HEATMAPS
            if 'rad_az' in self.plot_types and im.plotheatmap:
                rad_az_ax = axs['rad_az'] if isinstance(axs['rad_az'], mpl.axes.Axes) else axs['rad_az'][index]
                im.radial_profiles['xaxis'] = im.radial_profiles['xaxis'] # <- huh? 7/16/24
                im.rad_az_plot_data = im.radial_profiles.drop([str(im.PA),str(im.PA+90), 'mean'], axis = 1).astype('float')
                im.rad_az_plot_data['xaxis'] = ([str(round(float(val))) for val in im.rad_az_plot_data['xaxis'].values])
                im.rad_az_plot_data = im.rad_az_plot_data.set_index('xaxis')
                sb.heatmap(im.rad_az_plot_data/np.nanmax(im.rad_az_plot_data),
                          cmap = im.cmap, cbar_kws= {'pad':0.001, 'label':'Normalized Intensity'},
                          norm = self.prc_scales[im.imagery_type], ax = rad_az_ax)
                rad_az_ax.locator_params(axis='both', nbins=8)
                rad_az_ax.set_xticks(np.arange(0,nPAs,nPAs/8))
                rad_az_ax.set_xticklabels(np.arange(0,360,45))
                rad_az_ax.xaxis.set_minor_locator(MultipleLocator(3))
                rad_az_ax.tick_params(axis='both', labelrotation=0)
                rad_az_ax.set(title = im.imagery_source + " " + im.imagery_type, xlabel='PA (deg)', ylabel= 'Radius (mas)')

        for plot in self.plot_types:
            if plot in ('rad', 'az'):
                # put the legend in the first plot (for now)
                axs[plot][0].legend(title='Tracer', loc='upper right')

#############################################################

class AstroImage():
    def __init__(self,image_dict=None):
        if image_dict:
            #Set a path to the data directory
            self.obj = image_dict['Object']
            self.imagery_type = image_dict['Tracer']
            self.imagery_source = image_dict['Source']
            self.path = image_dict['Path']

            self.trim = np.nan if image_dict['trim'] == '' else image_dict['trim']
            self.r2 = image_dict['r2']
            self.stokes_component = image_dict['stokes_component']

            self.deproject_data = image_dict['deproject']

            self.override_center = [int(i) for i in image_dict['override_center']] if image_dict['override_center'] is not None else None

            self.plotheatmap = image_dict['Plot Heatmap']

            self.interval = image_dict["CB-Range"] #image_dict['perc_interval']
            self.asinh_a = image_dict['Scale Parameter'] #image_dict['asinh_a']

            self.n_start = image_dict['n_start'] if image_dict['n_start'] is not None else image_dict['n_deltas']
            self.n_deltas = image_dict['n_deltas']
            self.delta_lim = image_dict['delta_lim']
            self.show_deltas = image_dict['show_deltas']

            self.cmap = image_dict['cmap']
            self.color = image_dict['color']

            self.fits_table = fits.open(image_dict['Path'])
            self.header = self.fits_table[0].header

            # Open file and get required information from the header. Used primarily to get value for im.data, which is passed to figg.process.
            self.strip_header()

            self.or_im = self.data
            self.or_ycen = self.ycen
##############################################################################

    def __copy__(self):
        or_im = AstroImage()
        or_im.__dict__.update(self.__dict__)
        or_im.data = self.or_im
        or_im.ycen = self.or_ycen
        or_im.bounds = deepcopy(self.bounds)
        or_im.bounds['y'] = [or_im.ycen - self.auto_bound, or_im.ycen - self.auto_bound/2, or_im.ycen + self.auto_bound/2, or_im.ycen + self.auto_bound]
        return or_im

##############################################################################
    def strip_header(self):
        header = self.fits_table[0].header

        # If the source was ALMA...
        if self.imagery_source == 'ALMA':
            self.date = header['DATE-OBS'].split('T')[0]
            self.target = header['OBJECT']
            self.pixscale = None
            for row in header['HISTORY']:
                if 'cell' in row:
                    #https://casaguides.nrao.edu/index.php?title=Image_Continuum#Image_Parameters
                    self.pixscale = (float(row.split("cell")[1].split("=")[1].split("arcsec")[0].replace("'","").replace("[","").replace("]","")) * u.arcsec).to(u.si.mas).value
                    break
            if self.pixscale is None:
               self.pixscale = (abs(float(header['CDELT1'])) * u.deg).to(u.si.mas).value
            self.xcen,self.ycen = (header['CRPIX1'],header['CRPIX2'])
            self.resolution = (header['BMAJ'] + header['BMIN'])/2 * u.deg
            self.px_bin = int(np.floor(self.resolution.to(u.arcsec)/((self.pixscale * u.si.mas).to(u.arcsec))).value)
            self.units = header['BUNIT']

            # Get the frequency band
            self.band = ''
            # https://www.eso.org/public/teles-instr/alma/receiver-bands/
            bands = {'Band 1': [35, 50],
                     'Band 2': [67, 84],
                     'Band 3': [84, 116],
                     'Band 4': [125, 163],
                     'Band 5': [163, 211],
                     'Band 6': [211, 275],
                     'Band 7': [275, 373],
                     'Band 8': [385, 500],
                     'Band 9': [602, 720],
                     'Band 10': [787, 950]}
            for band in bands:
                if bands[band][0] <= (self.header['RESTFRQ'] * 1e-9) <= bands[band][1]: self.band = band

            if self.imagery_type == 'continuum':
                #conversion to float 64 required for image rotation later
                #https://github.com/quatrope/astroalign/issues/70
                self.data = skimage.util.img_as_float64(self.fits_table[0].data[0,0,:,:])

            if self.imagery_type == 'line':
                cube_data = self.fits_table[0].data
                if not math.isnan(self.trim):
                    trim = self.trim
                    data_trimmed = cube_data[0,trim[0][0]:trim[0][1],trim[1][0]:trim[1][1],trim[1][0]:trim[1][1]]
                    trimmed_path = self.path.split('.fits')[0] + '_trimmed.fits'
                    fits.writeto(trimmed_path, data_trimmed, header = file[0].header, overwrite = True)
                    newpath = trimmed_path
                else:
                    newpath = self.path
                if not os.path.exists(newpath.split('.fits')[0] + '_M8.fits'):
                    moments = bm.collapse_zeroth(velax=velax, data=masked_data, rms=rms)
                new_file = fits.open(newpath.split('.fits')[0] + '_M8.fits')
                #conversion to float 64 required for image rotation later
                #https://github.com/quatrope/astroalign/issues/70
                self.data = skimage.util.img_as_float64(new_file[0].data[::]).byteswap().newbyteorder()

        # If the source was SPHERE...
        if self.imagery_source == 'SPHERE':
            self.date = header['DATE']
            self.target = header["TARGET"]
            self.pixscale = header['SCALE']
            self.zeropt = header['ZEROPT']
            self.xcen,self.ycen = (header['STAR_X'],header['STAR_Y'])
            #Assuming North up, east left until told otherwise)
            #self.rot = 0

            self.band = header['FILTER']

            stokes_index = header['STOKES'].split(',').index(self.stokes_component)
            #conversion to float 64 required for image rotation later
            #https://github.com/quatrope/astroalign/issues/70
            self.data = skimage.util.img_as_float64(self.fits_table[0].data[stokes_index])

            #https://courses.lumenlearning.com/suny-physics/chapter/27-6-limits-of-resolution-the-rayleigh-criterion/
            rayleigh_crit = 1.22 * (header['WAVELE'] * u.si.micron).to(u.meter)/(8.2 * u.meter) * u.si.rad
            self.resolution = rayleigh_crit
            self.px_bin = int(np.floor(rayleigh_crit.to(u.si.mas)/(self.pixscale * u.si.mas)))
            self.units = header['FLUXUNIT']

        if self.override_center is not None:
            self.xcen,self.ycen = self.override_center[0], self.override_center[1]
##############################################################################
    def deproject(self,r2=False):
        ndimx,ndimy = self.data.shape[1],self.data.shape[0]
        halfx = int(round(self.xcen))
        halfy = int(round(self.ycen))
        half_px_bin = 1 #int(self.px_bin//2)

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
        if self.override_center is not None:
            ycen2 = int(round((ndimy2/ndimy)*self.ycen))
            print('Overrode center')
        else:
            ycen2 = ndimy2/2

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
        pa_rotate

        PURPOSE:
            #loop through PAs and rotate image so that that PA value is directly L/R

        INPUTS:
           #filename:[str] description

        AUTHOR:
            Original in IDL: Kate Follette
            Revised in Python: Catherine Sarosi Nov 30, 2022
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
        make_profiles

        PURPOSE:
            #Calculate radial profile for a disk at given PAs

        INPUTS:
           #filename:[str] description

        AUTHOR:
            Original in IDL: Kate Follette
            Revised in Python: Catherine Sarosi Nov 30, 2022
        '''
        radial_profiles = pd.DataFrame(columns=[str(PA) for PA in self.PAs])
        azimuthal_angles = np.arange(0,360)
        azimuthal_profiles = pd.DataFrame(columns=[str(PA) for PA in self.PAs])
        shape = self.file_rot.shape[1:]
        halfx = int(shape[1]/2)
        halfy = int(shape[0]/2)
        half_px_bin = int(self.px_bin//2)

        PA_mask = np.ones(shape, dtype=bool)
        PA_mask[halfy:shape[0],halfx-half_px_bin:halfx+half_px_bin+1] = False

        print('Generating radial profile...')
        for radcut in np.arange(0,(shape[0]/2/self.n_deltas)):
            r_in,r_out = (radcut * self.px_bin, (radcut+1) * self.px_bin)
            ap = (phot.CircularAnnulus([self.xcen,self.ycen], r_in, r_out)) if radcut > 0 else(phot.CircularAperture([self.xcen,self.ycen],r_out))

            for panum,pa in np.ndenumerate(self.PAs):
                radial_profiles.loc[radcut, str(pa)] = phot.ApertureStats(self.file_rot[panum],
                                                                          ap, sum_method = 'center',
                                                                          mask = PA_mask).median #mask = PA_mask + np.where(self.file_rot[panum]<0, True,False)).mean

                #probably WRONG??!?!?! Should be looping over more angles at higher radii
                #plot masked image
            deltas = []
            delta_lim = self.delta_lim
            if len(radial_profiles) > self.n_start:
                if self.r2:
                    delta_lim = self.delta_lim * (radcut+1 * self.px_bin)**2
                for i in np.arange(0,self.n_deltas):
                    deltas.append(np.max(abs(radial_profiles.loc[radcut-(i+1),:] - radial_profiles.loc[radcut-i,:])))
                clip_condition = all(delta < delta_lim for delta in deltas)
                if self.show_deltas:
                    print(radial_profiles.loc[radcut,:])
                    print(deltas)
                    print(delta_lim)
                if self.r2:
                    clip_condition = bool(clip_condition + (np.where(radial_profiles.loc[radcut-self.n_deltas:radcut+1,:] < 0, True,False).all()))
                if clip_condition:
                    break

        print('Generating azimuthal profiles...')
        for panum,pa in np.ndenumerate(self.PAs):
            # for rad in np.arange(self.radstep,radcut*self.pixscale*self.px_bin,self.radstep, dtype=int).tolist():
            for rad in self.radlist:
                rad = int(rad)
                r_in,r_out = (int(rad/self.pixscale - half_px_bin),int(rad/self.pixscale + half_px_bin + 1))
                ap = phot.CircularAnnulus([self.xcen,self.ycen],r_in,r_out) if r_in > 0 else phot.CircularAperture([self.xcen,self.ycen],r_out)
                azimuthal_profiles.loc[str(rad), str(pa)] = phot.ApertureStats(self.file_rot[panum],
                                                                    ap, sum_method = 'center',
                                                                    mask = PA_mask).median  #mask = PA_mask + np.where(self.file_rot[panum]<0, True,False)).sum

        self.radial_profiles = radial_profiles
        self.azimuthal_profiles = azimuthal_profiles
        self.n_cuts = radcut + 1
##############################################################################
    def plot_prep(self):
        '''
        profiles

        PURPOSE:
            #Calculate radial profile for a disk at given PAs

        INPUTS:
           #filename:[str] description

        AUTHOR:
            Original in IDL: Kate Follette
            Revised in Python: Catherine Sarosi Nov 30, 2022
        '''
        self.auto_bound = self.n_cuts * self.px_bin
        self.bounds = {}
        self.bounds['x'] = [self.xcen - self.auto_bound, self.xcen - self.auto_bound/2, self.xcen + self.auto_bound/2, self.xcen + self.auto_bound]
        self.bounds['y'] = [self.ycen - self.auto_bound, self.ycen - self.auto_bound/2, self.ycen + self.auto_bound/2, self.ycen + self.auto_bound]

        #make x axis in arcsec
        self.radial_profiles['xaxis'] = np.arange(0,self.n_cuts) * self.pixscale * self.px_bin + self.pixscale * self.px_bin/2
        self.radial_profiles['mean'] = self.radial_profiles.loc[:, ~self.radial_profiles.columns.isin([str(self.PA), str(self.PA+90),'xaxis'])].mean(axis= 1,skipna=True)
        rad_plot_data = pd.melt(self.radial_profiles, id_vars=['xaxis'], var_name='PA', value_name = 'flux')
        rad_plot_data['flux'] = rad_plot_data['flux']/np.nanmax(rad_plot_data['flux'])

        self.azimuthal_profiles.loc[-1] = self.radial_profiles.loc[:, ~self.radial_profiles.columns.isin([str(self.PA), str(self.PA+90),'xaxis'])].median(axis= 0,skipna=True)
        self.azimuthal_profiles.rename(index={-1:'median'},inplace=True)
        self.azimuthal_profiles['Radius (mas)'] = self.azimuthal_profiles.index
        az_plot_data = pd.melt(self.azimuthal_profiles, id_vars=['Radius (mas)'], var_name='PA', value_name = 'flux')
        az_plot_data['flux'] = az_plot_data['flux']/np.nanmax(az_plot_data['flux'])


        self.rad_plot_data = rad_plot_data
        self.az_plot_data = az_plot_data
##############################################################################
    def profiles(self):
        '''
        profiles

        PURPOSE:
            #Calculate radial and azimuthal profiles for a disk

        INPUTS:
           #filename:[str] description

        AUTHOR:
            Original in IDL: Kate Follette
            Revised in Python: Catherine Sarosi Nov 30, 2022
        '''
        print(f'Current image: {self.obj} {self.imagery_type}')

        self.shape = (self.data.shape[1],self.data.shape[0])
        self.nPAs = len(self.PAs)

        if self.override_center is not None:
            self.xcen,self.ycen = self.override_center[0], self.override_center[1]

        # Shitty way to save original data plus centers before deprojection
        self.or_im, self.or_ycen = self.data, self.ycen
        if self.deproject_data:
            self.deproject()

        self.pa_rotate()
        self.make_profiles()
        self.plot_prep()

        print(f'{self.obj} {self.imagery_type} done!')
