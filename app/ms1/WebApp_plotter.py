'''
Class for producing plots for web app
2023/02/23
E. Tyler Carr
'''
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from .functions_Universal_v3 import parse_headers
import io

class WebApp_plotter:
    '''
    Class that takes in a path to .xlsx or .csv file, stores data as dataframes and provides methods for plotting
    '''        
    def make_seq_line(self, data_path, seq_csv, ionization, y_scale='log', share_y=False, y_fixed=False, \
                      y_step=4, same_frame=False, legend=True, chemical_names=None, save_image=True, \
                      image_title=None, dark_mode=False):
        '''
        Method to make line plot of abundance vs. sequence faceted by chemical names.
        ------------------------------------------------------------------------------
        data_path (str):
            The path to xlsx or csv file filled with plotting data,
            xlsx files are expected to have sheets named 'tracer_pos' AND 'tracer_neg'
        seq_csv (str): 
            The path to a csv file whose first column yields the location labels,
            and whose second column yields the associated sequence number to that location
        ionization (str): 
            Determines which type of ESI to use for plots
            "pos": ESI+
            "neg": ESI-
        y_fixed (bool):
            Determines whether or not the y-ticks will be fixed for all plots
            True: ensures that the y-ticks are fixed for every plot
            False: lets matplotlib decide the y-ticks... this can cause y-ticks to differ between figures
        y_step (int):
            gives the number of ticks that will appear on a fixed y-axis
            only works when y_fixed=True
        same_frame (bool):
            Determines if every figure should have the same shape of (nrows=4, ncols=4)
            True: will ensure that all figures have a shape=(4, 4)
            False: will generate as many shape=(4, 4) figures it can fill, 
                then it may make a final figure with a smaller shape depending 
                on how many chemicals are left to plot
        legend (bool):
            Determines whether or not to build a legend for differentiating MB and Pooled locations
            True: will place a legend in the top right of figure for MB and Pooled locations
            False: no legend is generated
        chemical_names (list[str]):
            A list chemical names that you want plotted.
            If chemical_names=None, then all chemicals in data_path will be plotted
        save_image (bool):
            Determines whether or not to save the plots as .png
            True: saves all figures
            False: saves no figures
        image_title (str):
            Path to where you want your images saves, along with the naming scaffolding to be used
            If image_title=None, plots will be saved in your working directory with title 
                f'{plot_type}_ESI_{ionization}_{x}.png' where x is a 3 digit number
        dark_mode (bool):
            Determines if the plots will be made in dark mode or not
        ------------------------------------------------------------------------------
        By default there is no output unless save_image=True, in which case .png files will 
        be saved to disk. 
        '''
        ##########################################################
        ###        Check file types and import data            ###
        ##########################################################
        
        # Find the right sheet_name based on ionization value
        if ionization == 'pos':
            sn = 'tracer_pos'
            ion_token = '+' # used for plot title
        elif ionization == 'neg':
            sn = 'tracer_neg'
            ion_token = '-' # used for plot title
        else:
            raise Exception('ionization parameter must be ["pos", "neg"]')
            
        # logic to determine file type, raise exception if not .xlsx or .csv
        if data_path[-5:] == '.xlsx':
            # create the function for importing data
            import_func = lambda file_path: pd.read_excel(file_path, sheet_name=sn)
            
        elif data_path[-4:] == '.csv':
            # create the function for importing data
            import_func = lambda file_path: pd.read_csv(file_path)
        
        else:
            raise Exception('File type must be .xlsx or .csv')
        
        # import data using the lambda functions defined above
        df_tracer = import_func(data_path)
        
        # now we should read in the csv file with the location/sequence information
        if seq_csv[-4:] == '.csv':
            df_loc_seq = pd.read_csv(seq_csv)
        else:
            raise Exception('seq_csv must be the path to a CSV file')
            
            
        ############################################################
        ###   Setting colors for plotting before cleaning data   ###
        ############################################################
        
        # colors for [Sample, MB, Pooled]
        c_aes = ['teal', 'Orange', 'magenta']    # for scatter points
        c_leg_text = 'white'                     # for legend text
        c_leg_bg = "#f2f2f2"                     # for legend background
        c_leg_ec = '#000'                        # for legend edgecolor
        
        # if dark_mode=True, turn teal into cyan; and fix legend colors
        if dark_mode == True:
            c_aes[0] = 'cyan'
            c_leg_text = 'black'
            c_leg_bg = '#333'
            c_leg_ec = '#fff'
        
        # get indices of MD and Pooled locations and set conditional colors for plotting later
        try:
            locs = df_loc_seq.iloc[:,0]
            mb_indices = locs.index[locs.str.startswith('MB')]
            pool_indices = locs.index[locs.str.startswith('Pooled')]
            # set marker colors for plot
            mark_colors = [c_aes[1] if i in mb_indices else c_aes[2] if i in pool_indices else c_aes[0] \
                                                              for i in range(0, len(locs))]
        # in case there is an error above... this can probably be removed
        except:
            mark_colors = [c_aes[0] for i in range(0, len(locs))]
        
        
        ################################################
        ###             Clean the data               ###
        ################################################
        
        # start by getting df with chemical names and abundance at each location in sequential order
        col_names = [x for x in df_loc_seq.iloc[:, 0]]
        col_names.insert(0, 'Chemical_Name')
        df = df_tracer[col_names].copy()
        # need to make a column for lower cased names for sorting alpha-numerically (ignoring case)
        df['chem_name'] = df.loc[:, 'Chemical_Name'].str.lower().copy()
        df = df.sort_values('chem_name').copy()
        df = df.drop(['chem_name'], axis=1)
        
        # create a list of chemical names to work with
        if chemical_names is not None:
            # if user specified a list of chemicals, only keep relavent chems in our df
            df = df[df['Chemical_Name'].isin(chemical_names)]
        
        # capitalize the first letter of each chemical
        df['Chemical_Name'] = df['Chemical_Name'].apply(capitalize_chems)
        
        # our list of final chemical names with appropriate capitalization
        chemical_names = df['Chemical_Name']
        
        # split chem names into a nested list, one list of chem_names per plot
        # since each plot can only comfortably fit 16 chemicals
        chem_names = [[]]
        og_index = 0
        for c in chemical_names:
            if len(chem_names[og_index]) < 16:
                chem_names[og_index].append(c)
            else:
                chem_names.append([c])
                og_index += 1
            
                
        ################################################
        ###        Set up figures and axes           ###
        ###      And set up global aesthetics        ###
        ################################################
        
        # start by getting fig and axes objects by calling make_subplots()
        if y_fixed == True:
            # we need to get the minimum and maximum abundencies to auto generate our plot y-ranges
            y_max = df.max(numeric_only=True).max()
            y_min = df.min(numeric_only=True).min()
            
            # get a tupple of tupples that hold figs and axes objects for our faceted plots
            figs_axes = make_subplots(chem_names, y_max, y_min, y_scale=y_scale, y_steps=y_step, \
                                      share_y=share_y, same_frame=same_frame, dark_mode=dark_mode)         
        else:
            figs_axes = make_subplots(chem_names, share_y=share_y, y_scale=y_scale, \
                                      same_frame=same_frame, dark_mode=dark_mode)
    
    
        ####################################################
        ###      Split up by plot_type and plot          ###
        ####################################################        

        x_label = "Sequence"     # label for plot
        n_seq = len(df_loc_seq)  # number of sequences
        x_values = range(1, n_seq+1)

        # plot each chemical in its respective subplot
        # start by iterating through your sublists of chemicals (in groups of 16 or less)
        sublist_index = 0
        chem_index = 0    # index for pulling information from primary df
        while sublist_index < len(chem_names):
            chem_sublist = chem_names[sublist_index]
            fig, ax, shape, subtitle = figs_axes[sublist_index][0], figs_axes[sublist_index][1], \
                                       figs_axes[sublist_index][2], figs_axes[sublist_index][3]
            # set x and y labels, and plot title
            fig.text(0.5, -0.035, x_label, ha='center', va='center', fontsize=28)
            fig.text(-0.015, 0.5, 'Abundance', ha='center', va='center', rotation='vertical', fontsize=28)
            title = "Abundance vs. Sequence\n"
            sub = f"{subtitle}, ESI{ion_token}"
            fig.text(0.05, 1.045, title, fontsize=32)
            fig.text(0.05, 1.045, sub, fontsize=26)

            # add legend if needed
            if legend == True:
                # background box
                bg_patch = FancyBboxPatch(xy=(0.536, 1.044), width=0.416, height=0.092, \
                                          boxstyle='round,pad=0.008', fc=c_leg_bg, ec=c_leg_ec, lw=2, \
                                          transform=fig.transFigure, figure=fig)
                fig.patches.extend([bg_patch])
                # legend innards
                fig.text(0.55, 1.08, "Pooled", backgroundcolor=c_aes[2], \
                         c=c_leg_text, fontsize=23, fontfamily='serif', fontweight=500)
                fig.text(0.658, 1.08, "Method Blank", backgroundcolor=c_aes[1], \
                         c=c_leg_text, fontsize=23, fontfamily='serif', fontweight=500)
                fig.text(0.85, 1.08, "Sample", backgroundcolor=c_aes[0], \
                         c=c_leg_text, fontsize=23, fontfamily='serif', fontweight=500) 


            nrows, ncols = shape[0], shape[1]   # shape of subpot axis
            row_index, col_index = 0, 0         # indices for which subplot to put a chemical in

            # iterate through each chemical in the chemical sublist
            for chem in chem_sublist:
                # get abundance values to plot on y-axis
                y_values = [y for y in df.iloc[chem_index, 1:]]

                # plot the chemical abundance vs sequence in appropriate subplot
                # first deal with a single chemical
                if (nrows == 1) and (ncols == 1):
                    # plot once to get lines with no markers
                    ax.plot(x_values, y_values, color=c_aes[0])
                    # need to iterate through each color
                    for x, y, col in zip(x_values, y_values, mark_colors):
                        ax.plot(x, y, marker='o', color=c_aes[0], markersize=8, \
                                markeredgecolor=col, markerfacecolor=col)
                    ax.set_title(chem, fontsize=24, fontweight=600)

                # now deal with 2 and 3 chemicals
                elif (nrows != 1) and (ncols == 1):
                    # plot once to get lines with no markers
                    ax[row_index].plot(x_values, y_values, color=c_aes[0])
                    # need to iterate through each color
                    for x, y, col in zip(x_values, y_values, mark_colors):
                        ax[row_index].plot(x, y, marker='o', color=c_aes[0], \
                                            markersize=8, markeredgecolor=col, markerfacecolor=col)
                    ax[row_index].set_title(chem, fontsize=22, fontweight=600)
                    row_index += 1

                # finally deal with any more than 3 chemicals
                else:
                    try:
                        # plot once to get lines with no markers
                        ax[row_index, col_index].plot(x_values, y_values, color=c_aes[0])
                        # need to iterate through each color
                        for x, y, col in zip(x_values, y_values, mark_colors):
                            ax[row_index, col_index].plot(x, y, marker='o', color=c_aes[0], \
                                                          markersize=4, markeredgecolor=col, markerfacecolor=col)
                        ax[row_index, col_index].set_title(chem, fontsize=18, fontweight=600)
                        col_index += 1
                    except:
                        row_index += 1
                        # plot once to get lines with no markers
                        ax[row_index, 0].plot(x_values, y_values, color=c_aes[0])
                        # need to iterate through each color
                        for x, y, col in zip(x_values, y_values, mark_colors):
                            ax[row_index, 0].plot(x, y, marker='o', color=c_aes[0], \
                                                          markersize=4, markeredgecolor=col, markerfacecolor=col)
                        ax[row_index, 0].set_title(chem, fontsize=18, fontweight=600)
                        col_index = 1
                            
                # iterate to the next chemical
                chem_index += 1
                
                
            ########################################################
            ###               Setup shared x-axis                ###
            ########################################################
    
            # set x_ticks -- need to make sure the axis is so labels go to the highest row of plots
            # first for a single chemical (one plot)
            x0, x1 = 0, n_seq+1       # limits for xticks
            if (nrows == 1) and (ncols == 1):
                ax.set_xlim(x0, x1)
            # 2 or 3 chemicals
            elif nrows != 1 and ncols == 1:
                xticks = [t for t in ax[0].get_xticks() if t>=x0 and t<=x1]
                for r in range(nrows):
                    # make sure all plots have the same xlims
                    ax[r].set_xlim(x0, x1)
                    # if the plot below has axes on, then remove xticks
                    try:
                        if ax[r+1].axison:
                            ax[r].set_xticks(ticks=xticks, labels=['' for x in xticks], rotation=60, \
                                            fontsize=12, ha='right')
                        else:
                            pass
                    except:
                        pass
            # 4 or more chemicals
            else:
                xticks = [t for t in ax[0,0].get_xticks() if t>=x0 and t<=x1]
                for r in range(nrows):
                    for c in range(ncols):
                        # make sure all plots have the same xlims
                        ax[r,c].set_xlim(x0, x1)
                        # if the plot below has axes on, then remove xticks
                        try:
                            if ax[r+1, c].axison:
                                ax[r, c].set_xticks(ticks=xticks, labels=['' for x in xticks], rotation=60, \
                                                    fontsize=12, ha='right')
                            else:
                                pass
                        except:
                            pass

            # save image and iterate to the next sublist of 16 or fewer chemicals
            if save_image:
                if image_title is not None:
                    png_title = f"{image_title}_"
                else:
                    png_title = f"ab_vs_seq_ESI_{ion_token}_"
                png_title += f"{sublist_index}".zfill(3)
                fig.savefig(png_title, bbox_inches='tight')
            
            # iterate to the next figure
            sublist_index += 1
        return
            
    
    def make_seq_scatter(self, df_in, seq_csv, ionization, y_scale='log', fit=True, share_y=True, \
                      y_fixed=False, y_step=4, same_frame=False, legend=True, chemical_names=None, \
                      dark_mode=False):
        '''
        Method to make line plot of abundance vs. sequence faceted by chemical names.
        ------------------------------------------------------------------------------
        df_in (str):
            The data frame to be used for plotting.
        seq_csv (str): 
            The path to a csv file whose first column yields the location labels,
            and whose second column yields the associated sequence number to that location
        ionization (str): 
            Determines which type of ESI to use for plots
            "pos": ESI+
            "neg": ESI-
        y_fixed (bool):
            Determines whether or not the y-ticks will be fixed for all plots
            True: ensures that the y-ticks are fixed for every plot
            False: lets matplotlib decide the y-ticks... this can cause y-ticks to differ between figures
        y_step (int):
            gives the number of ticks that will appear on a fixed y-axis
            only works when y_fixed=True
        same_frame (bool):
            Determines if every figure should have the same shape of (nrows=4, ncols=4)
            True: will ensure that all figures have a shape=(4, 4)
            False: will generate as many shape=(4, 4) figures it can fill, 
                then it may make a final figure with a smaller shape depending 
                on how many chemicals are left to plot
        legend (bool):
            Determines whether or not to build a legend for differentiating MB and Pooled locations
            True: will place a legend in the top right of figure for MB and Pooled locations
            False: no legend is generated
        chemical_names (list[str]):
            A list chemical names that you want plotted.
            If chemical_names=None, then all chemicals in data_path will be plotted
        dark_mode (bool):
            Determines if the plots will be made in dark mode or not
        ------------------------------------------------------------------------------
        By default there is no output unless save_image=True, in which case .png files will 
        be saved to disk. 
        '''

        listOfPNGs=[]

        ##########################################################
        ###     set df_tracer and check for sequence data      ###
        ##########################################################
                            
        # set df_tracer to the input dataframe
        df_tracer = df_in
        
        # check if there is a sequence csv file
        if seq_csv is None:
            order_samples = False
        else:
            df_loc_seq = seq_csv
            order_samples = True
        if ionization == 'pos':
            ion_token = '+' 
        elif ionization == 'neg':
            ion_token = '-'   
            
        ############################################################
        ###   Setting colors for plotting before cleaning data   ###
        ############################################################
        
        # colors for [Sample, MB, Pooled]
        c_aes = ['teal', 'Orange', 'magenta']    # for scatter points
        c_leg_text = 'white'                     # for legend text
        c_leg_bg = "#f2f2f2"                     # for legend background
        c_leg_ec = '#000'                        # for legend edgecolor
        
        # if dark_mode=True, turn teal into cyan; and fix legend colors
        if dark_mode == True:
            c_aes[0] = 'cyan'
            c_leg_text = 'black'
            c_leg_bg = '#333'
            c_leg_ec = '#fff'
        
        # get indices of MD and Pooled locations and set conditional colors for plotting later
        # try:
        #     locs = df_loc_seq.iloc[:,0]
        #     mb_indices = [i for i in locs.index[locs.str.startswith('MB')]]
        #     pool_indices = [i for i in locs.index[locs.str.startswith('Pooled')]]
        #     # set marker colors for plot
        #     mark_colors = [c_aes[1] if i in mb_indices else c_aes[2] if i in pool_indices else c_aes[0] \
        #                                                       for i in range(0, len(locs))]
        # # in case there is an error above... this can probably be removed
        # except:
        #     mark_colors = [c_aes[0] for i in range(0, len(locs))]
        
        
        ################################################
        ###             Clean the data               ###
        ################################################
        
        # start by getting df with chemical names and abundance at each location in sequential order
        if order_samples:
            col_names = [x for x in df_loc_seq.iloc[:, 0]]
            # col_names.insert(0, 'Chemical_Name')
            df = df_tracer[col_names].copy()
        else:
            headers = parse_headers(df_in, 0)
            abundance = [item for sublist in headers for item in sublist if len(sublist) > 1]
            abundance.remove('Detection_Count(all_samples)')
            abundance.remove('Detection_Count(all_samples)(%)')
            df = df_in[abundance].copy()
        
        # our list of final chemical names with appropriate capitalization
        chemical_names = df_in['Chemical_Name']
    
        # # need to make a column for lower cased names for sorting alpha-numerically (ignoring case)
        # df['chem_name'] = df.loc[:, 'Chemical_Name'].str.lower().copy()
        # #df.loc[:,'Chemical_Name'] = df.loc[:,'Chemical_Name'].str.lower().copy()
        # df = df.sort_values('chem_name').copy()
        # df = df.drop(['chem_name'], axis=1)
        
        # # create a list of chemical names to work with
        # if chemical_names is not None:
        #     # if user specified a list of chemicals, only keep relavent chems in our df
        #     df = df[df['Chemical_Name'].isin(chemical_names)]
        
        # # capitalize the first letter of each chemical
        # df['Chemical_Name'] = df['Chemical_Name'].apply(capitalize_chems)
        
        
        # split chem names into a nested list, one list of chem_names per plot
        # since each plot can only comfortably fit 16 chemicals
        chem_names = [[]]
        og_index = 0
        for c in chemical_names:
            if len(chem_names[og_index]) < 16:
                chem_names[og_index].append(c)
            else:
                chem_names.append([c])
                og_index += 1
            
                
        ################################################
        ###        Set up figures and axes           ###
        ###      And set up global aesthetics        ###
        ################################################
        
        # start by getting fig and axes objects by calling make_subplots()
        if y_fixed == True:
            # we need to get the minimum and maximum abundencies to auto generate our plot y-ranges
            y_max = df.max(numeric_only=True).max()
            y_min = df.min(numeric_only=True).min()
            
            # get a tupple of tupples that hold figs and axes objects for our faceted plots
            figs_axes = make_subplots(chem_names, y_max, y_min, y_scale=y_scale, share_y=share_y, \
                                      y_steps=y_step, same_frame=same_frame, dark_mode=dark_mode)         
        else:
            figs_axes = make_subplots(chem_names, share_y=share_y, y_scale=y_scale, \
                                      same_frame=same_frame, dark_mode=dark_mode)
    
    
        ####################################################
        ###      Split up by plot_type and plot          ###
        ####################################################        

        x_label = "Samples"     # label for plot
        if order_samples:
            n_seq = len(df_loc_seq)  # number of sequences
        else:
            n_seq = len(abundance)  # number of samples
    
        # plot each chemical in its respective subplot
        # start by iterating through your sublists of chemicals (in groups of 16 or less)
        sublist_index = 0
        chem_index = 0    # index for pulling information from primary df
        while sublist_index < len(chem_names):
            chem_sublist = chem_names[sublist_index]
            fig, ax, shape, subtitle = figs_axes[sublist_index][0], figs_axes[sublist_index][1], \
                                       figs_axes[sublist_index][2], figs_axes[sublist_index][3]
            # set x and y labels, and plot title
            fig.text(0.5, -0.035, x_label, ha='center', va='center', fontsize=28)
            fig.text(-0.042, 0.5, 'Abundance', ha='center', va='center', rotation='vertical', fontsize=28)
            # title = "Abundance vs. Sequence\n"
            title = "Abundance across Samples\n"
            sub = f"{subtitle}, ESI{ion_token}"
            fig.text(0.05, 1.045, title, fontsize=32)
            fig.text(0.05, 1.045, sub, fontsize=26)

            # add legend if needed
            # if legend == True:
            if False:
                # background box
                bg_patch = FancyBboxPatch(xy=(0.536, 1.044), width=0.416, height=0.092, \
                                          boxstyle='round,pad=0.008', fc=c_leg_bg, ec=c_leg_ec, lw=2, \
                                          transform=fig.transFigure, figure=fig)
                fig.patches.extend([bg_patch])
                # legend innards
                fig.text(0.55, 1.08, "Pooled", backgroundcolor=c_aes[2], \
                         c=c_leg_text, fontsize=23, fontfamily='serif', fontweight=500)
                fig.text(0.658, 1.08, "Method Blank", backgroundcolor=c_aes[1], \
                         c=c_leg_text, fontsize=23, fontfamily='serif', fontweight=500)
                fig.text(0.85, 1.08, "Sample", backgroundcolor=c_aes[0], \
                         c=c_leg_text, fontsize=23, fontfamily='serif', fontweight=500) 


            nrows, ncols = shape[0], shape[1]   # shape of subpot axis
            row_index, col_index = 0, 0         # indices for which subplot to put a chemical in
            
            # pick a different markersize and linewidths for different figure shapes
            if ncols == 1:
                m_size = 140
                lin_w = 2.4
            elif ncols == 2:
                m_size = 90
                lin_w = 2.3
            elif ncols == 3:
                m_size = 50
                lin_w = 2.2
            else:
                m_size = 20
                lin_w = 1.9
            # iterate through each chemical in the chemical sublist
            for chem in chem_sublist:
                
                ##############################################################
                ###      Get x and y values from each location to plot     ###
                ##############################################################
                
                # set up x_values for the primary three location types
                # domain = range(1, n_seq+1)
                # x_values_mb = [x+1 for x in mb_indices[:]]          
                # x_values_pooled = [x+1 for x in pool_indices[:]]    
                # x_values_sample = [x-1 for x in domain if ((x not in x_values_mb) and (x not in x_values_pooled))]
                
                x_values_mb=[]
                x_values_pooled=[]
                x_values_sample = range(0, n_seq)
                
                # get abundance values to plot on y-axis
                y_values_sample = [y for y in df.iloc[chem_index, x_values_sample]]
                # get rid of nan values and set x_values
                y_keep_index = []
                for i, y in enumerate(y_values_sample):
                    if math.isnan(y) == False:
                        y_keep_index.append(i)
                y_values_sample = [y_values_sample[i] for i in y_keep_index]
                x_values_sample = [x_values_sample[i] for i in y_keep_index]

                        
                y_values_mb = [y for y in df.iloc[chem_index, x_values_mb]]
                # get rid of nan values and set x_values
                y_keep_index = []
                for i, y in enumerate(y_values_mb):
                    if math.isnan(y) == False:
                        y_keep_index.append(i)
                y_values_mb = [y_values_mb[i] for i in y_keep_index]
                x_values_mb = [x_values_mb[i] for i in y_keep_index]
                        
                y_values_pooled = [y for y in df.iloc[chem_index, x_values_pooled]]
                # get rid of nan values and set x_values
                y_keep_index = []
                for i, y in enumerate(y_values_pooled):
                    if math.isnan(y) == False:
                        y_keep_index.append(i)
                y_values_pooled = [y_values_pooled[i] for i in y_keep_index]
                x_values_pooled = [x_values_pooled[i] for i in y_keep_index]
                
                
                ###############################################################
                ###                 Actually start plotting                 ###
                ###############################################################
                
                # plot the chemical abundance vs sequence in appropriate subplot
                # first deal with a single chemical
                if (nrows == 1) and (ncols == 1):
                    # plot once to get lines with no markers
                    ax.scatter(x_values_sample, y_values_sample, color=c_aes[0], s=m_size, zorder=100)
                    ax.scatter(x_values_mb, y_values_mb, color=c_aes[1], s=m_size, zorder=100)
                    ax.scatter(x_values_pooled, y_values_pooled, color=c_aes[2], s=m_size, zorder=100)
                    ax.set_title(chem, fontsize=24, fontweight=600)
                    
                    # add a quadratic fits to plot
                    if fit == True:
                        if len(x_values_sample) > 2:
                            x_fit = np.linspace(min(x_values_sample), max(x_values_sample), len(x_values_sample))
                            coefs = np.polyfit(x_fit, y_values_sample, 2)
                            y_fit = np.polyval(coefs, x_fit)
                            # QA check, someimtes fit gives a negative value at the edge, looks horrible
                            if y_fit[-1] < 0:
                                y_fit = x_fit[:-1]
                                x_fit = x_fit[:-1]
                            if y_fit[0] < 0:
                                y_fit = y_fit[1:]
                                x_fit = x_fit[1:]
                            ax.plot(x_fit, y_fit, color=c_aes[0], lw=3, zorder=100)
                        if len(x_values_mb) > 2:
                            x_fit = np.linspace(min(x_values_mb), max(x_values_mb), len(x_values_mb))
                            coefs = np.polyfit(x_fit, y_values_mb, 2)
                            y_fit = np.polyval(coefs, x_fit)
                            # QA check, someimtes fit gives a negative value at the edge, looks horrible
                            if y_fit[-1] < 0:
                                y_fit = x_fit[:-1]
                                x_fit = x_fit[:-1]
                            if y_fit[0] < 0:
                                y_fit = y_fit[1:]
                                x_fit = x_fit[1:]
                            ax.plot(x_fit, y_fit, color=c_aes[1], lw=3, zorder=100)
                        if len(x_values_pooled) > 2:
                            x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), len(x_values_pooled))
                            coefs = np.polyfit(x_fit, y_values_pooled, 2)
                            y_fit = np.polyval(coefs, x_fit)
                            # QA check, someimtes fit gives a negative value at the edge, looks horrible
                            if y_fit[-1] < 0:
                                y_fit = x_fit[:-1]
                                x_fit = x_fit[:-1]
                            if y_fit[0] < 0:
                                y_fit = y_fit[1:]
                                x_fit = x_fit[1:]
                            ax.plot(x_fit, y_fit, color=c_aes[2], lw=3, zorder=100)

                    

                # now deal with 2 and 3 chemicals
                elif (nrows != 1) and (ncols == 1):
                    # plot once to get lines with no markers
                    ax[row_index].scatter(x_values_sample, y_values_sample, color=c_aes[0], s=m_size, zorder=100)
                    ax[row_index].scatter(x_values_mb, y_values_mb, color=c_aes[1], s=m_size, zorder=100)
                    ax[row_index].scatter(x_values_pooled, y_values_pooled, color=c_aes[2], s=m_size, zorder=100)
                    ax[row_index].set_title(chem, fontsize=18, fontweight=600)
                    
                    # add a quadratic fits to plot
                    if fit == True:
                        if len(x_values_sample) > 2:
                            x_fit = np.linspace(min(x_values_sample), max(x_values_sample), len(x_values_sample))
                            coefs = np.polyfit(x_fit, y_values_sample, 2)
                            y_fit = np.polyval(coefs, x_fit)
                            # QA check, someimtes fit gives a negative value at the edge, looks horrible
                            if y_fit[-1] < 0:
                                y_fit = x_fit[:-1]
                                x_fit = x_fit[:-1]
                            if y_fit[0] < 0:
                                y_fit = y_fit[1:]
                                x_fit = x_fit[1:]
                            ax[row_index].plot(x_fit, y_fit, color=c_aes[0], lw=3, zorder=100)
                        if len(x_values_mb) > 2:
                            x_fit = np.linspace(min(x_values_mb), max(x_values_mb), len(x_values_mb))
                            coefs = np.polyfit(x_fit, y_values_mb, 2)
                            y_fit = np.polyval(coefs, x_fit)
                            # QA check, someimtes fit gives a negative value at the edge, looks horrible
                            if y_fit[-1] < 0:
                                y_fit = x_fit[:-1]
                                x_fit = x_fit[:-1]
                            if y_fit[0] < 0:
                                y_fit = y_fit[1:]
                                x_fit = x_fit[1:]
                            ax[row_index].plot(x_fit, y_fit, color=c_aes[1], lw=3, zorder=100)
                        if len(x_values_pooled) > 2:
                            x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), len(x_values_pooled))
                            coefs = np.polyfit(x_fit, y_values_pooled, 2)
                            y_fit = np.polyval(coefs, x_fit)
                            # QA check, someimtes fit gives a negative value at the edge, looks horrible
                            if y_fit[-1] < 0:
                                y_fit = x_fit[:-1]
                                x_fit = x_fit[:-1]
                            if y_fit[0] < 0:
                                y_fit = y_fit[1:]
                                x_fit = x_fit[1:]
                            ax[row_index].plot(x_fit, y_fit, color=c_aes[2], lw=3, zorder=100)

                    # iterate to next plot
                    row_index += 1

                # finally deal with any more than 3 chemicals
                else:
                    try:
                        # plot once to get lines with no markers
                        
                        ax[row_index, col_index].scatter(x_values_sample, y_values_sample, color=c_aes[0], \
                                                         s=m_size, zorder=100)
                        ax[row_index, col_index].scatter(x_values_mb, y_values_mb, color=c_aes[1], \
                                                         s=m_size, zorder=100)
                        ax[row_index, col_index].scatter(x_values_pooled, y_values_pooled, color=c_aes[2], \
                                                         s=m_size, zorder=100)
                        ax[row_index, col_index].set_title(chem, fontsize=18, fontweight=600)
                        
                        # add a quadratic fits to plot
                        if fit == True:
                            if len(x_values_sample) > 2:
                                x_fit = np.linspace(min(x_values_sample), max(x_values_sample), len(x_values_sample))
                                coefs = np.polyfit(x_fit, y_values_sample, 2)
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), 80)
                                y_fit = np.polyval(coefs, x_fit)
                                # QA check, someimtes fit gives a negative value at the edge, looks horrible
                                if y_fit[-1] < 0:
                                    y_fit = x_fit[:-1]
                                    x_fit = x_fit[:-1]
                                if y_fit[0] < 0:
                                    y_fit = y_fit[1:]
                                    x_fit = x_fit[1:]
                                ax[row_index, col_index].plot(x_fit, y_fit, color=c_aes[0], lw=3, zorder=100)
                            if len(x_values_mb) > 2:
                                x_fit = np.linspace(min(x_values_mb), max(x_values_mb), len(x_values_mb))
                                coefs = np.polyfit(x_fit, y_values_mb, 2)
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), 80)
                                y_fit = np.polyval(coefs, x_fit)
                                # QA check, someimtes fit gives a negative value at the edge, looks horrible
                                if y_fit[-1] < 0:
                                    y_fit = x_fit[:-1]
                                    x_fit = x_fit[:-1]
                                if y_fit[0] < 0:
                                    y_fit = y_fit[1:]
                                    x_fit = x_fit[1:]
                                ax[row_index, col_index].plot(x_fit, y_fit, color=c_aes[1], lw=3, zorder=100)
                            if len(x_values_pooled) > 2:
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), len(x_values_pooled))
                                coefs = np.polyfit(x_fit, y_values_pooled, 2)
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), 80)
                                y_fit = np.polyval(coefs, x_fit)
                                # QA check, someimtes fit gives a negative value at the edge, looks horrible
                                if y_fit[-1] < 0:
                                    y_fit = x_fit[:-1]
                                    x_fit = x_fit[:-1]
                                if y_fit[0] < 0:
                                    y_fit = y_fit[1:]
                                    x_fit = x_fit[1:]
                                ax[row_index, col_index].plot(x_fit, y_fit, color=c_aes[2], lw=3, zorder=100)

                        # iterate to next plot
                        col_index += 1
                    
                    # try block fails after hitting the last column, so you jump to the next row
                    except:
                        row_index += 1
                        column_index = 0
                        # plot once to get lines with no markers
                        ax[row_index, column_index].scatter(x_values_sample, y_values_sample, color=c_aes[0], \
                                                            s=m_size, zorder=100)
                        ax[row_index, column_index].scatter(x_values_mb, y_values_mb, color=c_aes[1], \
                                                            s=m_size, zorder=100)
                        ax[row_index, column_index].scatter(x_values_pooled, y_values_pooled, color=c_aes[2], \
                                                            s=m_size, zorder=100)
                        ax[row_index, column_index].set_title(chem, fontsize=18, fontweight=600)
                        
                        # add a quadratic fits to plot
                        if fit == True:
                            if len(x_values_sample) > 2:
                                x_fit = np.linspace(min(x_values_sample), max(x_values_sample), len(x_values_sample))
                                coefs = np.polyfit(x_fit, y_values_sample, 2)
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), 80)
                                y_fit = np.polyval(coefs, x_fit)
                                # QA check, someimtes fit gives a negative value at the edge, looks horrible
                                if y_fit[-1] < 0:
                                    y_fit = x_fit[:-1]
                                    x_fit = x_fit[:-1]
                                if y_fit[0] < 0:
                                    y_fit = y_fit[1:]
                                    x_fit = x_fit[1:]
                                ax[row_index, column_index].plot(x_fit, y_fit, color=c_aes[0], lw=3, zorder=100)
                            if len(x_values_mb) > 2:
                                x_fit = np.linspace(min(x_values_mb), max(x_values_mb), len(x_values_mb))
                                coefs = np.polyfit(x_fit, y_values_mb, 2)
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), 80)
                                y_fit = np.polyval(coefs, x_fit)
                                # QA check, someimtes fit gives a negative value at the edge, looks horrible
                                if y_fit[-1] < 0:
                                    y_fit = x_fit[:-1]
                                    x_fit = x_fit[:-1]
                                if y_fit[0] < 0:
                                    y_fit = y_fit[1:]
                                    x_fit = x_fit[1:]
                                ax[row_index, column_index].plot(x_fit, y_fit, color=c_aes[1], lw=3, zorder=100)
                            if len(x_values_pooled) > 2:
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), len(x_values_pooled))
                                coefs = np.polyfit(x_fit, y_values_pooled, 2)
                                x_fit = np.linspace(min(x_values_pooled), max(x_values_pooled), 80)
                                y_fit = np.polyval(coefs, x_fit)
                                # QA check, someimtes fit gives a negative value at the edge, looks horrible
                                if y_fit[-1] < 0:
                                    y_fit = x_fit[:-1]
                                    x_fit = x_fit[:-1]
                                if y_fit[0] < 0:
                                    y_fit = y_fit[1:]
                                    x_fit = x_fit[1:]
                                ax[row_index, column_index].plot(x_fit, y_fit, color=c_aes[2], lw=3, zorder=100)
                        
                        # iterate to next plot
                        col_index = 1
                            
                # iterate to the next chemical
                chem_index += 1
                
                
            ########################################################
            ###               Setup shared x-axis                ###
            ########################################################
    
            # set x_ticks -- need to make sure the axis is so labels go to the highest row of plots
            # first for a single chemical (one plot)
            x0, x1 = 0, n_seq+1       # limits for xticks
            if (nrows == 1) and (ncols == 1):
                ax.set_xlim(x0, x1)
            # 2 or 3 chemicals
            elif nrows != 1 and ncols == 1:
                xticks = [t for t in ax[0].get_xticks() if t>=x0 and t<=x1]
                for r in range(nrows):
                    # make sure all plots have the same xlims
                    ax[r].set_xlim(x0, x1)
                    # if the plot below has axes on, then remove xticks
                    try:
                        if ax[r+1].axison:
                            ax[r].set_xticks(ticks=xticks, labels=['' for x in xticks], rotation=60, \
                                            fontsize=12, ha='right')
                        else:
                            pass
                    except:
                        pass
            # 4 or more chemicals
            else:
                xticks = [t for t in ax[0,0].get_xticks() if t>=x0 and t<=x1]
                for r in range(nrows):
                    for c in range(ncols):
                        # make sure all plots have the same xlims
                        ax[r,c].set_xlim(x0, x1)
                        # if the plot below has axes on, then remove xticks
                        try:
                            if ax[r+1, c].axison:
                                ax[r, c].set_xticks(ticks=xticks, labels=['' for x in xticks], rotation=60, \
                                                    fontsize=12, ha='right')
                            else:
                                pass
                        except:
                            pass
            # store matplot figure as a PNG
            buffer = io.BytesIO()
            plt.savefig(buffer)#, format='png')

            # append to list of PNGs
            listOfPNGs.append(buffer.getvalue())

            # close plt to prevent resource leaks
            plt.close()

            # iterate to the next figure
            sublist_index += 1

        return listOfPNGs, df
    
        
    def make_loc_plot(self, data_path, seq_csv, ionization, y_fixed=False, y_step=4, \
                       same_frame=False, chemical_names=None, save_image=True, \
                       image_title=None, dark_mode=False):
        '''
        Method to make line plot of abundance vs. location faceted by chemical names.
        Locations are expected to be in the form of f'Pooled{x}', f'MB{x}', and f'{y}_{x}' 
        where x is some value to distinguish between observations at the same location,
        and y is some sample location
            e.g., Pooled1, Pooled2, MB1, MB2, D1S2_1, D1S2_2, D2S2_1, D2S2_2
        ------------------------------------------------------------------------------
        data_path (str):
            The path to xlsx or csv file filled with plotting data,
            xlsx files are expected to have sheets named 'tracer_pos' AND 'tracer_neg'
        seq_csv (str): 
            The path to a csv file whose first column yields the location labels,
            and whose second column yields the associated sequence number to that location
        ionization (str): 
            Determines which type of ESI to use for plots
            "pos": ESI+
            "neg": ESI-
        y_fixed (bool):
            Determines whether or not the y-ticks will be fixed for all plots
            True: ensures that the y-ticks are fixed for every plot
            False: lets matplotlib decide the y-ticks... this can cause y-ticks to differ between figures
        y_step (int):
            gives the number of ticks that will appear on a fixed y-axis
            only works when y_fixed=True
        same_frame (bool):
            Determines if every figure should have the same shape of (nrows=4, ncols=4)
            True: will ensure that all figures have a shape=(4, 4)
            False: will generate as many shape=(4, 4) figures it can fill, 
                then it may make a final figure with a smaller shape depending 
                on how many chemicals are left to plot
        chemical_names (list[str]):
            A list chemical names that you want plotted.
            If chemical_names=None, then all chemicals in data_path will be plotted
        save_image (bool):
            Determines whether or not to save the plots as .png
            True: saves all figures
            False: saves no figures
        image_title (str):
            Path to where you want your images saves, along with the naming scaffolding to be used
            If image_title=None, plots will be saved in your working directory with title 
                f'{plot_type}_ESI_{ionization}_{x}.png' where x is a 3 digit number
        dark_mode (bool):
            Determines if the plots will be made in dark mode or not
        ------------------------------------------------------------------------------
        By default there is no output unless save_image=True, in which case .png files will 
        be saved to disk. 
        '''
        ##########################################################
        ###        Check file types and import data            ###
        ##########################################################
        
        # Find the right sheet_name based on ionization value
        if ionization == 'pos':
            sn = 'tracer_pos'
            ion_token = '+' # used for plot title
        elif ionization == 'neg':
            sn = 'tracer_neg'
            ion_token = '-' # used for plot title
        else:
            raise Exception('ionization parameter must be ["pos", "neg"]')
            
        # logic to determine file type, raise exception if not .xlsx or .csv
        if data_path[-5:] == '.xlsx':
            # create the function for importing data
            import_func = lambda file_path: pd.read_excel(file_path, sheet_name=sn)
            
        elif data_path[-4:] == '.csv':
            # create the function for importing data
            import_func = lambda file_path: pd.read_csv(file_path)
        
        else:
            raise Exception('File type must be .xlsx or .csv')
        
        # import data using the lambda functions defined above
        df_tracer = import_func(data_path)
        
        # now we should read in the csv file with the location/sequence information
        if seq_csv[-4:] == '.csv':
            df_loc_seq = pd.read_csv(seq_csv)
        else:
            raise Exception('seq_csv must be the path to a CSV file')
            
            
        ############################################################
        ###   Setting colors for plotting before cleaning data   ###
        ############################################################
        
        # colors for [linecolor, markercolor]
        c_aes = ['k', 'k']               # for scatter points
        c_leg_text = 'white'                     # for legend text
        c_leg_bg = "#f2f2f2"                     # for legend background
        c_leg_ec = '#000'                        # for legend edgecolor
        
        # if dark_mode=True, turn teal into cyan; and fix legend colors
        if dark_mode == True:
            c_aes[0] = 'lime'
            c_aes[1] = 'lime'
            c_leg_text = 'black'
            c_leg_bg = '#333'
            c_leg_ec = '#fff'
        
        
        ################################################
        ###             Clean the data               ###
        ################################################
        
        # start by getting df with chemical names and abundance at each location in sequential order
        col_names = [x for x in df_loc_seq.iloc[:, 0]]
        col_names.insert(0, 'Chemical_Name')
        df = df_tracer[col_names].copy()
        # need to make a column for lower cased names for sorting alpha-numerically (ignoring case)
        df['chem_name'] = df.loc[:, 'Chemical_Name'].str.lower().copy()
        #df.loc[:,'Chemical_Name'] = df.loc[:,'Chemical_Name'].str.lower().copy()
        df = df.sort_values('chem_name').copy()
        df = df.drop(['chem_name'], axis=1)
        
        # create a list of chemical names to work with
        if chemical_names is not None:
            # if user specified a list of chemicals, only keep relavent chems in our df
            df = df[df['Chemical_Name'].isin(chemical_names)]
        
        # capitalize the first letter of each chemical
        df['Chemical_Name'] = df['Chemical_Name'].apply(capitalize_chems)
        
        # our list of final chemical names with appropriate capitalization
        chemical_names = df['Chemical_Name']
        
        # split chem names into a nested list, one list of chem_names per plot
        # since each plot can only comfortably fit 16 chemicals
        chem_names = [[]]
        og_index = 0
        for c in chemical_names:
            if len(chem_names[og_index]) < 16:
                chem_names[og_index].append(c)
            else:
                chem_names.append([c])
                og_index += 1
                
        # there should be a single column for each location, whose records are a list of
        # the abundance observations for each chemical
        loc_tup = ('MB', 'Pooled')
        loc_types = []

        for col in df.columns[1:]:
            # if we don't have the location name yet, add it to our location list
            if not col.startswith(loc_tup):
                # get the sample location name without the _* ending
                sample_temp = col.split('_', 1)[0]
                loc_types.append(sample_temp)

        # now we have all of the loc types, set MB and Pooled to be the last values for plotting order
        for l in loc_tup:
            loc_types.append(l)

        # make new generalized location df
        df_loc = pd.DataFrame({'Chemical_Name': df['Chemical_Name']})
        df_loc[loc_types] = ''

        # update our dataframe to have new generalized location columns, whose record for a given chemical
        # is a tuple containing the abundances found at that location type
        # start by iterating through chemicals
        for index, row in df.iterrows():
            chem_loc_dic = {x: [] for x in loc_types}
            # now iterate throught locations for this chemical
            for i, location in enumerate(row[1:].index):
                i += 1 # iterate i so that its index corresponds to the correct column in df
                # now if location is found, append the abundance to appropraite key in our dictionary
                for loc_pre in loc_types:
                    if location.startswith(loc_pre):
                        chem_loc_dic[loc_pre].append(row[i])
            # now we have our dict of lists of abundances for this chemical, update this in df_loc
            for key, val in chem_loc_dic.items():
                df_loc[key][index] = val 
                
                
        ################################################
        ###        Set up figures and axes           ###
        ###      And set up global aesthetics        ###
        ################################################
        
        # start by getting fig and axes objects by calling make_subplots()
        if y_fixed == True:
            # we need to get the minimum and maximum abundencies to auto generate our plot y-ranges
            y_max = df.max(numeric_only=True).max()
            y_min = df.min(numeric_only=True).min()
            
            # get a tupple of tupples that hold figs and axes objects for our faceted plots
            figs_axes = make_subplots(chem_names, y_max, y_min, y_steps=y_step, \
                                      same_frame=same_frame, dark_mode=dark_mode)         
        else:
            figs_axes = make_subplots(chem_names, same_frame=same_frame, dark_mode=dark_mode)
    
        # if plot_type='loc', we should set our df to df_loc... this wasn't done earlier because of 
        # the y_max and y_min variables for y_fixed since df_loc has lists of entries
        df = df_loc
        del df_loc
    
    
        ####################################################
        ###      Split up by plot_type and plot          ###
        ####################################################
        
        # set variables that are dependent on plot_type
        n_chems = len(df)
        x_label = "Location"
        tick_labels = df.columns[1:]
        tick_pos = [x for x in range(1, len(tick_labels)+1)]
        # plot each chemical in its respective subplot
        # start by iterating through your sublists of chemicals (in groups of 16 or less)
        sublist_index = 0
        chem_index = 0    # index for pulling information from primary df
        while sublist_index < len(chem_names):
            chem_sublist = chem_names[sublist_index]
            fig, ax, shape, subtitle = figs_axes[sublist_index][0], figs_axes[sublist_index][1], \
                                       figs_axes[sublist_index][2], figs_axes[sublist_index][3]
            # set x and y labels, and plot title
            fig.tight_layout(h_pad=1)
            fig.text(0.5, -0.07, x_label, ha='center', va='center', fontsize=28)
            fig.text(-0.015, 0.5, 'Abundance', ha='center', va='center', rotation='vertical', fontsize=28)
            title = "Abundance vs. Location\n"
            sub = f"{subtitle}, ESI{ion_token}"
            fig.text(0.05, 1.045, title, fontsize=32)
            fig.text(0.05, 1.045, sub, fontsize=26)

            nrows, ncols = shape[0], shape[1]   # shape of subpot axis

            # figure out how many plots are in our figure
            # the try and except flow is to handle 4 or less chemicals... it is slow and could 
            # probably be improved. However, it is only slow in the case that it has to handle
            # 4 or less chemicals, which should be fairly infrequently, so it is probably good enough
            ax_count = 0
            try:
                for a in ax:
                    try:
                        for b in a:
                            if b.axison:
                                ax_count += 1
                    except:
                        if a.axison:
                            ax_count += 1
            except:
                ax_count = 1

            # get the y_values and x_values in a form to plot 
            # while loop it iterate through all plots in a figure
            row_index, col_index = 0, 0         # indices for which subplot to put a chemical in
            while (chem_index - 16*sublist_index) < ax_count: #and (chem_index < n_chems):
                # for loop to set up x and y values for each plot in a figure
                x_values, y_values = [], []
                for i, v in enumerate(df.iloc[chem_index, 1:]):
                    y_values.append(v)
                    x_app = [i+1 for x in range(len(v))]
                    x_values.append(x_app)

                # first deal with a single chemical
                if (nrows == 1) and (ncols == 1):
                    # while loop to actually plot all of the x and yvalues on a plot
                    i = 0
                    while i < len(x_values):
                        # plot, set ticks to be blank to immitate share_x option, set chemical name as plot title
                        ax.plot(x_values[i], y_values[i], color=c_aes[0], marker='o', \
                                    markersize=10, markeredgecolor=c_aes[1], markerfacecolor=c_aes[1])
                        ax.set_xticks(ticks=tick_pos, labels=tick_labels, \
                                                        rotation=50, fontsize=18, ha='right')
                        ax.set_xlim(0, tick_pos[-1]+0.75)
                        ax.set_title(chemical_names.iloc[chem_index], fontsize=18, fontweight=600)
                        i += 1      # iterate to next location for plotting this chemical

                # now deal with 2 and 3 chemicals
                elif (nrows != 1) and (ncols == 1):
                    i = 0
                    while i < len(x_values):
                        # plot, set ticks to be blank to immitate share_x option, set chemical name as plot title
                        ax[row_index].plot(x_values[i], y_values[i], color=c_aes[0], marker='o', \
                                    markersize=10, markeredgecolor=c_aes[1], markerfacecolor=c_aes[1], lw=2)
                        ax[row_index].set_xticks(ticks=tick_pos, labels=tick_labels, \
                                                        rotation=50, fontsize=18, ha='right')
                        ax[row_index].set_xlim(0, tick_pos[-1]+0.75)
                        ax[row_index].set_title(chemical_names.iloc[chem_index], fontsize=18, fontweight=600)
                        # set up the shared x-axis
                        if row_index == nrows-1:
                            ax[row_index].set_xticks(ticks=tick_pos, labels=tick_labels, \
                                                        rotation=50, fontsize=18, ha='right')
                        else:
                            ax[row_index].set_xticks(ticks=tick_pos, labels=['' for x in tick_labels], \
                                                        rotation=0, fontsize=1, ha='right')
                        i += 1      # iterate to next location for plotting this chemical
                    row_index += 1

                # now deal with 4 or more chemicals
                else:
                    # while loop to actually plot all of the x and yvalues on a plot
                    i = 0
                    while i < len(x_values):
                        # check if we need to move to the next row of plots
                        if col_index >= ncols:
                            col_index = 0
                            row_index += 1
                        # plot, set ticks to be blank to immitate share_x option, set chemical name as plot title
                        ax[row_index, col_index].plot(x_values[i], y_values[i], color=c_aes[0], marker='o', \
                                                      markersize=4, markeredgecolor=c_aes[1], \
                                                      markerfacecolor=c_aes[1], lw=2)
                        ax[row_index, col_index].set_xticks(ticks=tick_pos, labels=['' for x in tick_labels], \
                                                        rotation=0, fontsize=1, ha='right')
                        ax[row_index, col_index].set_xlim(0, tick_pos[-1]+0.75)
                        ax[row_index, col_index].set_title(chemical_names.iloc[chem_index], fontsize=18, fontweight=600)                        
                        i += 1        # iterate to next location for plotting this chemical

                    # set x_ticks -- need to make sure the axis is so labels go to the highest row of plots
                    for c in range(ncols):
                        if ax[-1, c].axison:
                            ax[-1, c].set_xticks(ticks=tick_pos, labels=tick_labels, rotation=60, \
                                                fontsize=12, ha='right')
                        elif ax[-2, c].axison:
                            ax[-2, c].set_xticks(ticks=tick_pos, labels=tick_labels, rotation=60, \
                                                fontsize=12, ha='right')
                        elif ax[-3, c].axison:
                            ax[-3, c].set_xticks(ticks=tick_pos, labels=tick_labels, rotation=60, \
                                                fontsize=12, ha='right')
                        else:
                            ax[-4, c].set_xticks(ticks=tick_pos, labels=tick_labels, rotation=60, \
                                                fontsize=12, ha='right')

                chem_index += 1   # iterate to the next chemical for plotting
                col_index += 1    # iterate to the next plot in this figure
            
            # save image and iterate to the next sublist of 16 or fewer chemicals
            if save_image:
                if image_title is not None:
                    png_title = f"{image_title}_"
                else:
                    png_title = f"ab_vs_seq_ESI_{ion_token}_"
                png_title += f"{sublist_index}".zfill(3)
                fig.savefig(png_title, bbox_inches='tight')
               
            # iterate to the next figure
            sublist_index += 1    
        return
                



def capitalize_chems(s):
    '''
    Function that takes in a string, and will capitalize the first letter of the compound
    after any hyphenated values
    Will ignore certain fully alphabetical prefixes ['alpha-', 'beta-', 'ortho-', 'cis-', '(R)-', etc]
    e.g., if s = 13C,d2-hydrochlorothiazide; this function returns 13C,d2-Hydrochlorothiazide
          if s = 2-hydroxy-ibuprofen; this function returns 2-Hydroxy-ibuprofen
          if s = alpha-glucose; tjis function returns alpha-Glucose
    ----------------------------------------------------------------------------------------
    s (str):
        The string that will be capitalized
    '''
    # create a list of prefixes to ignore
    ignore_list = ['ortho', 'meta', 'para', 'N', 'O', 'alpha', 'beta', 'gamma', \
                   'sec', 'tert', 'cis', 'trans', '(E)', '(Z)', '(R)', '(S)', 'D', 'L']
    
    # split the chemical name by hyphens
    s_parts = s.split('-')
    
    # iterate through our substrings the find and capitalize the first compound name to capitalize
    # This is done by finding the first substring that does not contain a number and does not show 
    # up in our ignore list
    for i, s_sub in enumerate(s_parts):
        if not ((s_sub in ignore_list) or any(chr.isdigit() for chr in s_sub)):
            s_parts[i] = s_sub[0].upper() + s_sub[1:]
            return '-'.join(s_parts)

def make_subplots(chem_names, y_max=None, y_min=None, y_steps=4, share_y=True, y_scale='log', same_frame=False, \
                  dark_mode=False):
    '''
    Function to take in a nested list of chemical names and output the appropriate figures, 
    axes, axes shapes, and subtitles for plt naming in a list of tupples for plotting. 
    ----------------------------------------------------------------------------------------
    chem_names (list[list[str]]):
        A list of lists of chemical names to be plotted
        Each figure/axis pair can handle up to 16 chemicals so, each list within chem_names 
            must have a length less than or equal to 16
            e.g., input: [["c0", "c1", "c2",...,"c16"], ["c17", "c18", "c19"]]
                  output: [(fig1, ax1, (nrows1, ncols1), subtitle1), (fig2, ax2, (nrows2, ncols2), subtitle2)]
    y_max (float):
        The order of magnitude of 10*y_max is used as the max tick value on the y-axis
    y_min (float):
        The order of magnitude of y_min/10 is used as the min tick value on the y-axis
    y_steps (int):
        Determines the number of ticks that will appear on the y-axis
        Tries to make evenly spaced ticks, but due to forcing an integer power of 10 some y_steps
            are incompatible with some combinations of y_min and y_max... just check your y-axis carefully
            and try using a different value for y_steps if the ticks are not evenly spaced
        e.g., y_min=0.00002; y_max=45120; y_steps=6 will produce ticks at
              y_ticks=[10^-5, 10^-3, 10^-1, 10^1, 10^3, 10^5]
    ---> if y_max = y_min = None; then matplotlib will automatically produce y-ticks for each figure
         independently from one another. This may cause the y-ticks to differ from plot to plot
    same_frame (bool):
        Determines if every figure should have the same shape of (nrows=4, ncols=4)
        True: will ensure that all figures have a shape=(4, 4)
        False: will generate as many shape=(4, 4) figures it can fill, 
            then it may make a final figure with a smaller shape depending 
            on how many chemicals are left to plot
    dark_mode (bool):
        Determines if the plots will be made in dark mode or not
    sh_x (bool):
        Determines if the plots will share an x-axis
    ----------------------------------------------------------------------------------------
    Output takes the form of a list of tupples which contain a plt.Figure object, a plt.Axes object,
    a tupple that contains the shape of the subplots in the form of (nrows, ncols), and a subtitle
    associted with the figure that numerically lists which chemicals are within the figure
    '''
    
    #########################################################################
    ###          QA and setting and global aesthetic parameters           ###
    #########################################################################
    
    # ensure each list within chem_names is the right length
    for c_list in chem_names:
        if len(c_list) > 16:
            raise Exception("You have a list within chem_names whose length is longer than 16")
    
    # rcParams must be set before any plt objects are created!
    # Now set parameters that are needed for dark_mode=True
    if dark_mode == True:
        # es params
        plt.rcParams.update({'axes.facecolor': '#0d0d0d', 
                             'axes.edgecolor': '#fff',
                             'axes.titlecolor': '#fff'})
        # tick params
        plt.rcParams.update({'xtick.color': '#fff',
                             'ytick.color': '#fff'})
        # figure params
        plt.rcParams.update({'figure.facecolor': "#000"})
        # text params
        plt.rcParams.update({'text.color': '#fff'})

    # now parameters for light_mode (dark_mode=False)
    else:
        # axes params
        plt.rcParams.update({'axes.facecolor': '#fff', 
                             'axes.edgecolor': '#000',
                             'axes.titlecolor': '#000'})
        # tick params
        plt.rcParams.update({'xtick.color': '#000',
                             'ytick.color': '#000'})        
        # figure params
        plt.rcParams.update({'figure.facecolor': '#e6e6e6'})
        # text params
        plt.rcParams.update({'text.color': '#000'})

    # now set params that are needed for dark and light mode
    # font params
    plt.rcParams.update({'font.family': 'serif'})
    # tick params
    plt.rcParams.update({'xtick.labelsize': 13,
                         'ytick.labelsize': 13,
                         'xtick.major.size': 5,
                         'ytick.major.size': 5,
                         'xtick.major.width': 0.9,
                         'ytick.major.width': 0.9 })
    
    
    ####################################################
    ###       Create the subplots and figures        ###
    ####################################################
    
    # create figs and axes for each list within chem_names
    n_plots = len(chem_names)
    i = 0
    subtitle_i = 0
    figs_axes = []
    while i < n_plots:
        c_list = chem_names[i]
        n_chems = len(c_list)
        
        # now we need a gnarly if else block to handle differnt chemical list lengths...
        # if same_frame, then all plots get a 4x4
        if (n_chems in [13, 14, 15, 16]) or (same_frame == True):
            fig, ax = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems == 1:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems == 2:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems == 3:
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems == 4:
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems in [5, 6]:
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems in [7, 8, 9]:
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems in [7, 8, 9]:
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        elif n_chems in [10, 11, 12]:
            fig, ax = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=share_y, subplot_kw={'yscale':y_scale})
        
        
        ######################################################
        ###        Setting figure/axes aesthetics          ###
        ######################################################
        
        # need to handle 1x1 plot differently since axes object isnt an array
        if (n_chems == 1) and (same_frame == False):
            # generate the subtitle for the plot to state which chemicals are being plotted
            subtitle = f'Chemical {subtitle_i+1}'
            # turn on grid and get shape
            ax.grid()
            shape = (1, 1)
            # if y_fixed == True then set the ticks
            if (y_max is not None) and (y_min is not None):
                # Need to find the floor-nearest-power-of-ten for ymax and ymin
                y_max_pow = int(np.floor(np.log10(y_max)))
                y_min_pow = int(np.floor(np.log10(y_min)))
                n = y_steps # number of ticks on y-axis
                # set y value ranges 
                ax.set_ylim([10**(y_min_pow-1), 10**(y_max_pow+1)])
                # pick the tick locations and labels
                ax.set_yticks(ticks=[10**x for x in np.linspace(y_min_pow, y_max_pow, n)], \
                                  labels=[f'$10^{{{int(x)}}}$' for x in np.linspace(y_min_pow, y_max_pow, n)])
        else:
            # generate the subtitle for the plot to state which chemicals are being plotted
            subtitle_f = subtitle_i + n_chems
            subtitle = f'Chemicals {subtitle_i+1}-{subtitle_f}'
            subtitle_i = subtitle_f
            # get shape of axis object
            axe = ax.ravel()              # have to unpack gridspec object (from subplots() function)
            gs = axe[0].get_gridspec()
            shape = (gs.nrows, gs.ncols)

            # remove unused subplot axes
            for j in range(n_chems, gs.ncols*gs.nrows):
                axe[j].set_axis_off()
                
                

            # setting axes grids and ticks
            for j in range(0,n_chems):
                # turn on the grids
                axe[j].grid()
                
                # if y_fixed == True then set the ticks
                if (y_max is not None) and (y_min is not None):
                    # Need to find the floor-nearest-power-of-ten for ymax and ymin
                    y_max_pow = int(np.floor(np.log10(y_max)))
                    y_min_pow = int(np.floor(np.log10(y_min)))
                    n = y_steps # number of ticks on y-axis
                    # set y value ranges 
                    axe[j].set_ylim([10**(y_min_pow-1), 10**(y_max_pow+1)])
                    # pick the tick locations and labels
                    axe[j].set_yticks(ticks=[10**x for x in np.linspace(y_min_pow, y_max_pow, n)], \
                                      labels=[f'$10^{{{int(x)}}}$' for x in np.linspace(y_min_pow, y_max_pow, n)])
        
        # a few more fig aesthetics, append our list of tupples to return, then iterate to next figure
        fig.set_size_inches(14, 8)
        # if share y is not on, we need extra space between plots horizontally (width w_pad)
        if share_y == False and y_scale == 'linear':
            fig.tight_layout(h_pad=1, w_pad=3.2)
        else:
            fig.tight_layout(h_pad=1)
        figs_axes.append((fig, ax, shape, subtitle))
        i += 1
    
    return figs_axes
