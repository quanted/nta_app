"""
Class for producing plots for web app
2025/01/17
E. Tyler Carr
"""
import logging
from typing import Any, Literal, Union

from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
import numpy as np
import pandas as pd

from .task_functions import determine_string_width, parse_headers

## set axes return type annotation alias for the WAPlotter._make_subplots() method
AxesType = Union[np.ndarray[np.ndarray[Axes]], np.ndarray[Axes], Axes]

# set up logging
logger = logging.getLogger("nta_app.ms1")


class WebApp_plotter:
    """
    Class that takes in a path to .xlsx or .csv file, stores data as dataframes and provides methods for plotting
    """

    def __init__(self):
        self.color_mode = "light"
        self._set_color_dict(self.color_mode)  ## creates self.color_dict
        self._set_rcParams_dict()  ## creates self.rcParams_dict using self.color_dict
        self._set_rcParams()  ## by default sets rcParams with "light" mode enabled

    def _set_color_dict(self, color_mode: Literal["light", "dark"] = "light") -> None:
        """Updates self.color_dict from light to dark or vice versa. Updates self.color_mode.

        Parameters
        ----------
        color_mode : Literal["light", "dark"], default "light"
            Which mode to set the colors to.
        """
        if color_mode == "light":
            self.color_dict = {
                "fig_face_c": "#e6e6e6",
                "axes_face_c": "white",
                "axes_edge_c": "black",
                "axes_title_c": "black",
                "text_c": "black",
                "xtick_c": "black",
                "ytick_c": "black",
                "legend_font_c": "white",
                "legend_bg_c": "#f2f2f2",
                "legend_edge_c": "black",
                "scatter_colors": [
                    "teal",
                    "orange",
                    "magenta",
                    "blue",
                    "green",
                    "red",
                    "cyan",
                    "yellow",
                    "white",
                    "black",
                ],
            }
        elif color_mode == "dark":
            self.color_dict = {
                "fig_face_c": "black",
                "axes_face_c": "#0d0d0d",
                "axes_edge_c": "white",
                "axes_title_c": "white",
                "text_c": "white",
                "xtick_c": "white",
                "ytick_c": "white",
                "legend_font_c": "black",
                "legend_bg_c": "#a2a2a2",
                "legend_edge_c": "black",
                "scatter_colors": [
                    "cyan",
                    "orange",
                    "magenta",
                    "blue",
                    "green",
                    "red",
                    "teal",
                    "yellow",
                    "white",
                    "black",
                ],
            }
        else:
            raise ValueError(f"The `color_mode` parameter must be 'light' or 'dark', not {color_mode}")

        self.color_mode = color_mode

    def _set_rcParams_dict(self):
        """Sets the self.rcParams_dict for styling plots."""
        self.rcParam_dict = {
            "font.family": "serif",
            "figure.facecolor": self.color_dict["fig_face_c"],
            "figure.subplot.hspace": 0.95,
            "axes.facecolor": self.color_dict["axes_face_c"],
            "axes.edgecolor": self.color_dict["axes_edge_c"],
            "axes.titlecolor": self.color_dict["axes_title_c"],
            "axes.titlepad": 10.0,
            "text.color": self.color_dict["text_c"],
            "xtick.color": self.color_dict["xtick_c"],
            "ytick.color": self.color_dict["ytick_c"],
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "mathtext.fontset": "custom",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
        }

    def _set_rcParams(self):
        """Sets the plt.rcParams for styling plots using self.rcParams_dict."""
        plt.rcParams.update(self.rcParam_dict)

    def _check_for_seq(
        self,
        df_seq: Union[None, pd.DataFrame],
        debug_list: list[Any],
        df_in: pd.DataFrame,
    ) -> tuple[pd.DataFrame, str, list[Any], list[str], list[int]]:
        """Handles the sequence data for plotting.

        Parameters
        ----------
        df_seq : Union[None, pd.DataFrame]
            The sequence data.
        debug_list : list[Any]
            The debug_list. Will likely an empty list when initially calling this function.

        Returns
        -------
        tuple[pd.DataFrame, str, list[Any], list[str], list[int]]
            The sequence df, a boolean values indicating whether to order samples or not,
            nd updated debug_list (only updated if df_seq = None), a list of unique sample groups,
            and a list of sample indices for each sample group.

        This code was written by Alex Chao.
        """
        if df_seq is None:
            # Sort dataframe columns alphabetically prior to parsing headers
            df_in = df_in.reindex(sorted(df_in.columns), axis=1)  # Remove sorting to
            df_in = df_in[
                ["Feature ID"] + [col for col in df_in.columns if col != "Feature ID"]
            ]  # Move mass column to front of dataframe; if a sample replicate is the first column when parsing headers it loses that replicate from the group

            # Debug_list
            debug_list.append("After sorting: df_in columns")
            debug_list.append(df_in.columns.values)

            # If there is no sequence file, create a dummy sequence dataframe containing the sample names straight from the input data file
            headers = parse_headers(df_in)
            abundance = [item for sublist in headers for item in sublist if len(sublist) > 1]

            # Debug_list
            debug_list.append("Samples from parse_headers")
            debug_list.append(abundance)

            # 5/21/2024 AC: In certain cases if the samples have multiple layers of repetition to their naming,
            # the parse_headers function will grab the mean/CV/std/median columns as samples in addition to the raw samples.
            # Remove these from the sample list below
            column_prefixes_to_remove = [
                "Mean_",
                "Median_",
                "STD_",
                "N_Abun_",
                "CV_",
                "Replicate_Percent_",
                "Occurrence_Count",
            ]
            abundance = [
                entry
                for entry in abundance
                if not any(entry.startswith(prefix) for prefix in column_prefixes_to_remove)
            ]

            df_loc_seq = pd.DataFrame()
            df_loc_seq["Sample Sequence"] = abundance
            order_samples = False
        else:
            df_loc_seq = df_seq
            order_samples = True

        # AC Check if sample sequence file has more than one column, second column would be the sample group column
        if len(df_loc_seq.columns) > 1:
            sample_group_unique = df_loc_seq.iloc[:, 1].unique().tolist()
        else:  # If there is no sample group column, assign all samples to sample group 'Sample'
            sample_group_unique = ["Sample"]
            # Create the sample group column in the dataframe
            df_loc_seq["Sample_Group"] = "Sample"

        # AC Loop through sample group column and get indices of samples for each sample group
        indices_list = []
        for i in range(len(sample_group_unique)):
            temp_indices = df_loc_seq.index[df_loc_seq.iloc[:, 1] == sample_group_unique[i]].tolist()
            indices_list.append(temp_indices)

        return df_loc_seq, order_samples, debug_list, sample_group_unique, indices_list, abundance

    def _clean_seq_data(
        self, df_in: pd.DataFrame, df_loc_seq: pd.DataFrame, order_samples: bool
    ) -> tuple[pd.DataFrame, list[list[str]], list[Any]]:
        """Cleans the sequence data for creating sequence scatter plot.

        Parameters
        ----------
        df_in : pd.DataFrame
            The tracer statistics df.
        df_loc_seq : pd.DataFrame
            The sequence df returned from self._check_for_seq().
        order_samples : bool
            Whether to order samples, as returned from self._check_for_seq()

        Returns
        -------
        tuple[pd.DataFrame, list[list[str]]]
            A dataframe with cleaned/ordered samples and a list containing lists of chemicals --
            there is one nested list for each figure that will be produced (max of 16 per list)
            (e.g., for 17 chemicals we would have [[chem_1, ..., chem_16], [chem_17]]).
        """
        ## start by getting df with chemical names and abundance at each location in sequential order
        abundance = None
        if order_samples:
            col_names = [x for x in df_loc_seq.iloc[:, 0]]
            col_names.insert(0, "Chemical_Name")  # AC 1/4/2024 Add in chemical name column to dataframe
            # df_in["Chemical_Name"] = ["" for x in df_in.iloc[:, 0].values]
            df = df_in[col_names].copy()
        else:
            # Sort dataframe columns alphabetically prior to parsing headers
            df_in = df_in.reindex(sorted(df_in.columns), axis=1)
            df_in = df_in[
                ["Feature ID"] + [col for col in df_in.columns if col != "Feature ID"]
            ]  # Move mass column to front of dataframe; if a sample replicate is the first column when parsing headers it loses that replicate from the group

            headers = parse_headers(df_in)
            abundance = [item for sublist in headers for item in sublist if len(sublist) > 1]
            abundance.insert(0, "Chemical_Name")  # AC 1/4/2024 Add in chemical name column to dataframe
            # abundance.remove('Detection_Count(all_samples)')
            # abundance.remove('Detection_Count(all_samples)(%)')
            # 5/21/2024 AC: In certain cases if the samples have multiple layers of repetition to their naming,
            # the parse_headers function will grab the mean/CV/std/median columns as samples in addition to the raw samples.
            # Remove these from the sample list below
            column_prefixes_to_remove = [
                "Mean_",
                "Median_",
                "STD_",
                "N_Abun_",
                "CV_",
                "Replicate_Percent_",
                "Occurrence_Count",
            ]
            abundance = [
                entry
                for entry in abundance
                if not any(entry.startswith(prefix) for prefix in column_prefixes_to_remove)
            ]

            df = df_in[abundance].copy()

        # our list of final chemical names with appropriate capitalization
        chemical_names = df_in["Chemical_Name"]

        ## split chem names into a nested list, one list of chem_names per plot
        chem_names = [[]]
        og_index = 0
        for c in chemical_names:
            if len(chem_names[og_index]) < 16:
                chem_names[og_index].append(c)
            else:
                chem_names.append([c])
                og_index += 1

        return df, chem_names, abundance

    def _make_subplots(
        self,
        chem_names: list[list[str]],
        same_frame: bool = False,
        share_y: bool = False,
        y_scale: Literal["linear", "log"] = "linear",
    ) -> tuple[tuple[Figure, AxesType, tuple[int], str]]:
        """Creates figures and axes for plotting features.

        Parameters
        ----------
        chem_names : list[list[str]]
            The nested list of list of chemical names to be plotted. Each nested list represents a single
            figure (max 16 per nested list. This input is part of the output of self._clean_seq_data().
        same_frame : bool, default False
            Determines if every figure should have the same shape of (nrows=4, ncols=4).
            If True, ensures that all figures have a shape=(4, 4).
            If False, generates as many shape=(4, 4) figures it can fill,
            then it may make a final figure with a smaller shape depending
            on how many chemicals are left to plot.
        share_y : bool, default False
            If True, each row of plots will share a y-axis labels on the first column.
        y_scale : Literal["linear", "log"], default "linear"
            Whether the y-axis scale will be linear or logarithmic.

        Returns
        -------
        tuple[tuple[Figure, AxesType, tuple[int], str]]
            A tuple contains a single tuple per Figure (i.e. per 16 chemicals to plot). Each tuple contains
            the Figure object, the axes for the figure (1 or 2 dimensional np.ndarray of Axes for n_chems > 1,
            or just an Axes object for n_chems=1), the shape of the figure/axes (e.g., (4, 4) for 4x4 subplots),
            and the subtitle string used for plotting.
        """
        ## ensure each list within chem_names is the right length
        for c_list in chem_names:
            if len(c_list) > 16:
                raise Exception("You have a list within chem_names whose length is longer than 16")

        # create figs and axes for each list within chem_names
        n_plots = len(chem_names)
        i = 0
        subtitle_i = 0
        figs_axes = []
        while i < n_plots:
            c_list = chem_names[i]
            n_chems = len(c_list)

            ## setup fig and axes for this plot based on number of chems; if same_frame, then all plots get a 4x4
            if (n_chems in [13, 14, 15, 16]) or (same_frame == True):
                nrows, ncols = 4, 4
            elif n_chems == 1:
                nrows, ncols = 1, 1
            elif n_chems == 2:
                nrows, ncols = 2, 1
            elif n_chems == 3:
                nrows, ncols = 3, 1
            elif n_chems == 4:
                nrows, ncols = 2, 2
            elif n_chems in [5, 6]:
                nrows, ncols = 2, 3
            elif n_chems in [7, 8, 9]:
                nrows, ncols = 3, 3
            elif n_chems in [10, 11, 12]:
                nrows, ncols = 3, 4

            fig, ax = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
                constrained_layout=True,
            )

            ## set subtitles and turn grid lines on
            if (n_chems == 1) and (same_frame == False):
                # generate the subtitle for the plot to state which chemicals are being plotted
                subtitle = f"Chemical {subtitle_i+1}"
                # turn on grid and get shape
                ax.grid()
                shape = (1, 1)
            else:
                # generate the subtitle for the plot to state which chemicals are being plotted
                subtitle_f = subtitle_i + n_chems
                subtitle = f"Chemicals {subtitle_i+1}-{subtitle_f}"
                subtitle_i = subtitle_f
                # get shape of axis object
                axe = ax.ravel()  # have to unpack gridspec object (from subplots() function)
                gs = axe[0].get_gridspec()
                shape = (gs.nrows, gs.ncols)

                # remove unused subplot axes
                for j in range(n_chems, gs.ncols * gs.nrows):
                    axe[j].set_axis_off()

                # setting axes grids and ticks
                for j in range(0, n_chems):
                    # turn on the grids
                    axe[j].grid()

            # a few more fig aesthetics, append our list of tupples to return, then iterate to next figure
            fig.set_size_inches(14, 8)
            # if share y is not on, we need extra space between plots horizontally (width w_pad)
            if share_y == False and y_scale == "linear":
                fig.tight_layout(pad=2.5, h_pad=1, w_pad=3.5)
            else:
                fig.tight_layout(pad=2.5, h_pad=1, w_pad=4.5)
            figs_axes.append((fig, ax, shape, subtitle))
            i += 1

        return figs_axes

    def _add_legend(self, fig: Figure, sample_group_unique: list[str]) -> Figure:
        """Adds a legend to the figure based on the sample names in sample_group_unique"""
        ## background box
        bg_patch = FancyBboxPatch(
            xy=(0.536, 1.044),
            width=0.416,
            height=0.092,
            boxstyle="round,pad=0.008",
            fc=self.color_dict["legend_bg_c"],
            ec=self.color_dict["legend_edge_c"],
            lw=2,
            transform=fig.transFigure,
            figure=fig,
        )
        fig.patches.extend([bg_patch])
        # AC 1/3/2022 Add legend back to figure with colors denoted by sample group
        # legend innards
        # AC Loop through legend label generation
        legend_x_coord = []  # List of x-coordinates for sample group in legend
        # character_increment = 0.018  # How much to increment x-coordinate per character
        base_separation = 0.025  # Flat base amount of separation between group labels

        for b in range(len(sample_group_unique)):
            # Get x coordinate of sample group legend text based on number of characters
            if b == 0:
                legend_x_coord.append(0.55)  # First x-coordinate is always 0.55
                # char_count = len(sample_group_unique[b])
                # next_x_increment = char_count * character_increment + base_separation
                next_x_increment = determine_string_width(sample_group_unique[b]) + base_separation
            else:
                last_value = legend_x_coord[-1]
                legend_x_coord.append(last_value + next_x_increment)
                # char_count = len(sample_group_unique[b])
                # next_x_increment = char_count * character_increment + base_separation
                next_x_increment = determine_string_width(sample_group_unique[b]) + base_separation
        # Display the legend text for each sample group
        for a in range(len(sample_group_unique)):
            fig.text(
                legend_x_coord[a],
                1.08,
                sample_group_unique[a],
                backgroundcolor=self.color_dict["scatter_colors"][a],
                c=self.color_dict["legend_font_c"],
                fontsize=23,
                fontfamily="serif",
                fontweight=500,
            )

        return fig

    def _add_fit(
        self,
        ax: Axes,
        x_values_list: list[list[float]],
        y_values_list: list[list[float]],
        sample_group_unique: list[str],
    ) -> Axes:
        """Adds a quadratic best fit curve to ax.

        It must be that len(x_values_list) = len(y_values_list) = len(sample_group_unique)

        Parameters
        ----------
        ax : Axes
            The Axes object to add the curve of best fit to.
        x_values_list : list[list[float]]
            The nested lists of x values corresponding to each subplot.
        y_values_list : list[list[float]]
            The nested lists of y values corresponding to each subplot.
        sample_group_unique : list[str]
            The list of sample names.

        Returns
        -------
        Axes
            The updated Axes object with plotted quadratic curves of best fit.
        """
        for b in range(len(sample_group_unique)):
            if len(x_values_list[b]) > 2:
                x_fit = np.linspace(
                    min(x_values_list[b]),
                    max(x_values_list[b]),
                    len(x_values_list[b]),
                )
                coefs = np.polyfit(
                    x_fit,
                    y_values_list[b],
                    2,
                )
                domain = np.linspace(x_fit[0], x_fit[-1], 60)
                y_fit = np.polyval(coefs, domain)
                # QA check, someimtes fit gives a negative value at the edge, looks horrible
                if y_fit[-1] < 0:
                    y_fit = x_fit[:-1]
                    x_fit = x_fit[:-1]
                if y_fit[0] < 0:
                    y_fit = y_fit[1:]
                    x_fit = x_fit[1:]
                ax.plot(domain, y_fit, color=self.color_dict["scatter_colors"][b], lw=3, zorder=100)

        return ax

    def _set_y_ticks(self, ax: Axes, y_scale: Literal["linear", "log"], y_step: str) -> Axes:
        """Updates ax to have y_step evenly spaced (in linear or log space) tick marks."""
        ylims = ax.get_ylim()

        if y_scale == "linear":
            locator = MaxNLocator(nbins=y_step - 2, prune="lower")
            y_ticks = locator.tick_values(vmin=min(ylims), vmax=max(ylims))
            ax.set_yticks(y_ticks)
        else:
            y_ticks_full = np.logspace(np.log10(min(ylims)), np.log10(max(ylims)), y_step)
            ## round to 2 sig figs
            y_ticks = [
                (yy // 10 ** int(np.floor(np.log10(yy)) - 2)) * 10 ** int(np.floor(np.log10(yy)) - 2)
                for yy in y_ticks_full
            ]
            ## setup ytick labels (otherwise it will leave out some labels for some reason)
            y_labels = [
                f"${((np.floor(yy/10**(len(str(int(yy)))-2))*10**(len(str(int(yy)))-2)) / 10**int(np.floor(np.log10(yy))))} \\times 10^{{{int(np.floor(np.log10(yy)))}}}$"
                for yy in y_ticks
            ]
            ax.set_yticks([], minor=True)
            ax.set_yticks(y_ticks, y_labels)

        return ax

    def _plot_seq_scatter(
        self,
        df: pd.DataFrame,
        chem_names: list[list[str]],
        figs_axes: tuple[tuple[Figure, AxesType, tuple[int], str]],
        legend: bool,
        sample_group_unique: list[str],
        indices_list: list[int],
        x_label: str,
        ion_token: str,
        fit: bool,
        n_seq: int,
        y_step: Union[None, "int"],
        y_scale: Literal["linear", "log"],
    ) -> list[Figure]:
        """Does the actual plotting for self.make_seq_scatter()."""
        # plot each chemical in its respective subplot
        # start by iterating through your sublists of chemicals (in groups of 16 or less)
        sublist_index = 0
        chem_index = 0  # index for pulling information from primary df
        figs = []
        x0, x1 = 0, n_seq + 1  # limits for xticks
        while sublist_index < len(chem_names):
            chem_sublist = chem_names[sublist_index]
            fig, ax, shape, subtitle = (
                figs_axes[sublist_index][0],
                figs_axes[sublist_index][1],
                figs_axes[sublist_index][2],
                figs_axes[sublist_index][3],
            )

            # set x and y labels, and plot title
            fig.text(0.5, -0.035, x_label, ha="center", va="center", fontsize=28)
            fig.text(
                -0.042,
                0.5,
                "Abundance",
                ha="center",
                va="center",
                rotation="vertical",
                fontsize=28,
            )
            # title = "Abundance vs. Sequence\n"
            title = "Abundance across Samples\n"
            sub = f"{subtitle}, ESI{ion_token}"
            fig.text(0.05, 1.045, title, fontsize=32)
            fig.text(0.05, 1.045, sub, fontsize=26)

            if legend == True:
                fig = self._add_legend(fig, sample_group_unique)

            nrows, ncols = shape[0], shape[1]  # shape of subplot axis
            row_index, col_index = (
                0,
                0,
            )  # indices for which subplot to put a chemical in

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
                # AC Loop through x/y data generation for each sample group

                x_values_list = []
                y_values_list = []

                # For each sample group, go through each list of indices, pulling out x and y values from each group
                for h in range(len(sample_group_unique)):
                    x_values_temp = [x + 1 for x in indices_list[h][:]]
                    y_values_temp = [y for y in df.iloc[chem_index, x_values_temp]]

                    # get rid of nan values and set x_values
                    y_keep_index = []
                    for i, y in enumerate(y_values_temp):
                        ## if na_filter=False when calling pd.read_excel/read_csv, math.isnan() will not work
                        try:
                            if math.isnan(y) == False:
                                y_keep_index.append(i)
                        except:
                            if y != "":
                                y_keep_index.append(i)

                    y_values_temp = [y_values_temp[i] for i in y_keep_index]
                    x_values_temp = [x_values_temp[i] for i in y_keep_index]

                    x_values_list.append(x_values_temp)
                    y_values_list.append(y_values_temp)

                ## Now that we have values, actually plot. First handle a single chem
                if (nrows == 1) and (ncols == 1):
                    # AC Loop through scatter plot generation
                    # ax.scatter(x_values_list[0], y_values_list[0], color=c_aes[0], s=m_size, zorder=100)
                    for k in range(len(sample_group_unique)):
                        ax.scatter(
                            x_values_list[k],
                            y_values_list[k],
                            color=self.color_dict["scatter_colors"][k],
                            s=m_size,
                            zorder=100,
                        )

                    ax.set_title(chem, fontsize=24, fontweight=600)

                    ## fits to plot
                    if fit == True:
                        ax = self._add_fit(ax, x_values_list, y_values_list, sample_group_unique)

                    ## update xticks
                    ax.set_xlim(x0, x1)

                    if y_step is not None:
                        ax = self._set_y_ticks(ax, y_scale, y_step)

                ## now deal with 2 and 3 chemicals
                elif (nrows != 1) and (ncols == 1):
                    # AC Loop through scatter plot generation
                    for k in range(len(sample_group_unique)):
                        ax[row_index].scatter(
                            x_values_list[k],
                            y_values_list[k],
                            color=self.color_dict["scatter_colors"][k],
                            s=m_size,
                            zorder=100,
                        )

                    ax[row_index].set_title(chem, fontsize=18, fontweight=600)

                    ## add fits to plot
                    if fit == True:
                        ax[row_index] = self._add_fit(ax[row_index], x_values_list, y_values_list, sample_group_unique)

                    ## update xticks
                    xticks = [t for t in ax[0].get_xticks() if t >= x0 and t <= x1]
                    for r in range(nrows):
                        ## make sure all plots have the same xlims
                        ax[r].set_xlim(x0, x1)
                        ## if the plot below has axes on, then remove xticks
                        try:
                            if ax[r + 1].axison:
                                ax[r].set_xticks(
                                    ticks=xticks,
                                    labels=["" for x in xticks],
                                    rotation=60,
                                    fontsize=12,
                                    ha="right",
                                )
                            else:
                                pass
                        except:
                            pass

                    ## yticks
                    if y_step is not None:
                        ax[row_index] = self._set_y_ticks(ax[row_index], y_scale, y_step)

                    ## iterate to next plot
                    row_index += 1

                ## finally deal with any more than 3 chemicals
                else:
                    try:
                        # AC Loop through scatter plot generation
                        for k in range(len(sample_group_unique)):
                            ax[row_index, col_index].scatter(
                                x_values_list[k],
                                y_values_list[k],
                                color=self.color_dict["scatter_colors"][k],
                                s=m_size,
                                zorder=100,
                            )

                        ## Adjust fontsize of subplot based on chemical name length
                        if len(chem) > 20:
                            chem_plot_fontsize = 400 / len(chem)
                        else:
                            chem_plot_fontsize = 18

                        ax[row_index, col_index].set_title(chem, fontsize=chem_plot_fontsize, fontweight=600)

                        ## add fits to plot
                        if fit == True:
                            ax[row_index, col_index] = self._add_fit(
                                ax[row_index, col_index], x_values_list, y_values_list, sample_group_unique
                            )

                        ## yticks
                        if y_step is not None:
                            ax[row_index, col_index] = self._set_y_ticks(ax[row_index, col_index], y_scale, y_step)

                        ## iterate to next plot
                        col_index += 1

                    except:
                        row_index += 1
                        column_index = 0

                        # AC Loop through scatter plot generation
                        for k in range(len(sample_group_unique)):
                            ax[row_index, column_index].scatter(
                                x_values_list[k],
                                y_values_list[k],
                                color=self.color_dict["scatter_colors"][k],
                                s=m_size,
                                zorder=100,
                            )

                        ## Adjust fontsize of subplot based on chemical name length
                        if len(chem) > 20:
                            chem_plot_fontsize = 400 / len(chem)
                        else:
                            chem_plot_fontsize = 18
                        ax[row_index, column_index].set_title(chem, fontsize=chem_plot_fontsize, fontweight=600)

                        ## add fits to plot
                        if fit == True:
                            ax[row_index, column_index] = self._add_fit(
                                ax[row_index, column_index], x_values_list, y_values_list, sample_group_unique
                            )

                        ## update xticks
                        xticks = [t for t in ax[0, 0].get_xticks() if t >= x0 and t <= x1]
                        for r in range(nrows):
                            for c in range(ncols):
                                ## make sure all plots have the same xlims
                                ax[r, c].set_xlim(x0, x1)
                                ## if the plot below has axes on, then remove xticks
                                try:
                                    if ax[r + 1, c].axison:
                                        ax[r, c].set_xticks(
                                            ticks=xticks,
                                            labels=["" for x in xticks],
                                            rotation=60,
                                            fontsize=12,
                                            ha="right",
                                        )
                                    else:
                                        pass
                                except:
                                    pass

                        ## yticks
                        if y_step is not None:
                            ax[row_index, column_index] = self._set_y_ticks(
                                ax[row_index, column_index], y_scale, y_step
                            )

                        ## iterate to next plot
                        col_index = 1

                ## iterate to the next chemical
                chem_index += 1

            figs.append(fig)

            ## iterate to next sublist of chemicals (next figure)
            sublist_index += 1

        return figs

    def make_seq_line(
        self,
        data_path,
        seq_csv,
        ionization,
        y_scale="log",
        share_y=False,
        y_fixed=False,
        y_step=4,
        same_frame=False,
        legend=True,
        chemical_names=None,
        save_image=True,
        image_title=None,
        dark_mode=False,
    ):
        """
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
        """
        ##########################################################
        ###        Check file types and import data            ###
        ##########################################################

        # Find the right sheet_name based on ionization value
        if ionization == "pos":
            sn = "tracer_pos"
            ion_token = "+"  # used for plot title
        elif ionization == "neg":
            sn = "tracer_neg"
            ion_token = "-"  # used for plot title
        else:
            raise Exception('ionization parameter must be ["pos", "neg"]')

        # logic to determine file type, raise exception if not .xlsx or .csv
        if data_path[-5:] == ".xlsx":
            # create the function for importing data
            import_func = lambda file_path: pd.read_excel(file_path, sheet_name=sn)

        elif data_path[-4:] == ".csv":
            # create the function for importing data
            import_func = lambda file_path: pd.read_csv(file_path)

        else:
            raise Exception("File type must be .xlsx or .csv")

        # import data using the lambda functions defined above
        df_tracer = import_func(data_path)

        # now we should read in the csv file with the location/sequence information
        if seq_csv[-4:] == ".csv":
            df_loc_seq = pd.read_csv(seq_csv)
        else:
            raise Exception("seq_csv must be the path to a CSV file")

        ############################################################
        ###   Setting colors for plotting before cleaning data   ###
        ############################################################

        # colors for [Sample, MB, Pooled]
        c_aes = ["teal", "Orange", "magenta"]  # for scatter points
        c_leg_text = "white"  # for legend text
        c_leg_bg = "#f2f2f2"  # for legend background
        c_leg_ec = "#000"  # for legend edgecolor

        # if dark_mode=True, turn teal into cyan; and fix legend colors
        if dark_mode == True:
            c_aes[0] = "cyan"
            c_leg_text = "black"
            c_leg_bg = "#333"
            c_leg_ec = "#fff"

        # get indices of MD and Pooled locations and set conditional colors for plotting later
        try:
            locs = df_loc_seq.iloc[:, 0]
            mb_indices = locs.index[locs.str.startswith("MB")]
            pool_indices = locs.index[locs.str.startswith("Pooled")]
            # set marker colors for plot
            mark_colors = [
                c_aes[1] if i in mb_indices else c_aes[2] if i in pool_indices else c_aes[0]
                for i in range(0, len(locs))
            ]
        # in case there is an error above... this can probably be removed
        except:
            mark_colors = [c_aes[0] for i in range(0, len(locs))]

        ################################################
        ###             Clean the data               ###
        ################################################

        # start by getting df with chemical names and abundance at each location in sequential order
        col_names = [x for x in df_loc_seq.iloc[:, 0]]
        col_names.insert(0, "Chemical_Name")
        df = df_tracer[col_names].copy()
        # need to make a column for lower cased names for sorting alpha-numerically (ignoring case)
        df["chem_name"] = df.loc[:, "Chemical_Name"].str.lower().copy()
        df = df.sort_values("chem_name").copy()
        df = df.drop(["chem_name"], axis=1)

        # create a list of chemical names to work with
        if chemical_names is not None:
            # if user specified a list of chemicals, only keep relavent chems in our df
            df = df[df["Chemical_Name"].isin(chemical_names)]

        # capitalize the first letter of each chemical
        df["Chemical_Name"] = df["Chemical_Name"].apply(capitalize_chems)

        # our list of final chemical names with appropriate capitalization
        chemical_names = df["Chemical_Name"]

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
            figs_axes = make_subplots(
                chem_names,
                y_max,
                y_min,
                y_scale=y_scale,
                y_steps=y_step,
                share_y=share_y,
                same_frame=same_frame,
                dark_mode=dark_mode,
            )
        else:
            figs_axes = make_subplots(
                chem_names,
                share_y=share_y,
                y_scale=y_scale,
                same_frame=same_frame,
                dark_mode=dark_mode,
            )

        ####################################################
        ###      Split up by plot_type and plot          ###
        ####################################################

        x_label = "Sequence"  # label for plot
        n_seq = len(df_loc_seq)  # number of sequences
        x_values = range(1, n_seq + 1)

        # plot each chemical in its respective subplot
        # start by iterating through your sublists of chemicals (in groups of 16 or less)
        sublist_index = 0
        chem_index = 0  # index for pulling information from primary df
        while sublist_index < len(chem_names):
            chem_sublist = chem_names[sublist_index]
            fig, ax, shape, subtitle = (
                figs_axes[sublist_index][0],
                figs_axes[sublist_index][1],
                figs_axes[sublist_index][2],
                figs_axes[sublist_index][3],
            )
            # set x and y labels, and plot title
            fig.text(0.5, -0.035, x_label, ha="center", va="center", fontsize=28)
            fig.text(
                -0.015,
                0.5,
                "Abundance",
                ha="center",
                va="center",
                rotation="vertical",
                fontsize=28,
            )
            title = "Abundance vs. Sequence\n"
            sub = f"{subtitle}, ESI{ion_token}"
            fig.text(0.05, 1.045, title, fontsize=32)
            fig.text(0.05, 1.045, sub, fontsize=26)

            # add legend if needed
            if legend == True:
                # background box
                bg_patch = FancyBboxPatch(
                    xy=(0.536, 1.044),
                    width=0.416,
                    height=0.092,
                    boxstyle="round,pad=0.008",
                    fc=c_leg_bg,
                    ec=c_leg_ec,
                    lw=2,
                    transform=fig.transFigure,
                    figure=fig,
                )
                fig.patches.extend([bg_patch])
                # legend innards
                fig.text(
                    0.55,
                    1.08,
                    "Pooled",
                    backgroundcolor=c_aes[2],
                    c=c_leg_text,
                    fontsize=23,
                    fontfamily="serif",
                    fontweight=500,
                )
                fig.text(
                    0.658,
                    1.08,
                    "Method Blank",
                    backgroundcolor=c_aes[1],
                    c=c_leg_text,
                    fontsize=23,
                    fontfamily="serif",
                    fontweight=500,
                )
                fig.text(
                    0.85,
                    1.08,
                    "Sample",
                    backgroundcolor=c_aes[0],
                    c=c_leg_text,
                    fontsize=23,
                    fontfamily="serif",
                    fontweight=500,
                )

            nrows, ncols = shape[0], shape[1]  # shape of subpot axis
            row_index, col_index = (
                0,
                0,
            )  # indices for which subplot to put a chemical in

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
                        ax.plot(
                            x,
                            y,
                            marker="o",
                            color=c_aes[0],
                            markersize=8,
                            markeredgecolor=col,
                            markerfacecolor=col,
                        )
                    ax.set_title(chem, fontsize=24, fontweight=600)

                # now deal with 2 and 3 chemicals
                elif (nrows != 1) and (ncols == 1):
                    # plot once to get lines with no markers
                    ax[row_index].plot(x_values, y_values, color=c_aes[0])
                    # need to iterate through each color
                    for x, y, col in zip(x_values, y_values, mark_colors):
                        ax[row_index].plot(
                            x,
                            y,
                            marker="o",
                            color=c_aes[0],
                            markersize=8,
                            markeredgecolor=col,
                            markerfacecolor=col,
                        )
                    ax[row_index].set_title(chem, fontsize=22, fontweight=600)
                    row_index += 1

                # finally deal with any more than 3 chemicals
                else:
                    try:
                        # plot once to get lines with no markers
                        ax[row_index, col_index].plot(x_values, y_values, color=c_aes[0])
                        # need to iterate through each color
                        for x, y, col in zip(x_values, y_values, mark_colors):
                            ax[row_index, col_index].plot(
                                x,
                                y,
                                marker="o",
                                color=c_aes[0],
                                markersize=4,
                                markeredgecolor=col,
                                markerfacecolor=col,
                            )
                        ax[row_index, col_index].set_title(chem, fontsize=18, fontweight=600)
                        col_index += 1
                    except:
                        row_index += 1
                        # plot once to get lines with no markers
                        ax[row_index, 0].plot(x_values, y_values, color=c_aes[0])
                        # need to iterate through each color
                        for x, y, col in zip(x_values, y_values, mark_colors):
                            ax[row_index, 0].plot(
                                x,
                                y,
                                marker="o",
                                color=c_aes[0],
                                markersize=4,
                                markeredgecolor=col,
                                markerfacecolor=col,
                            )
                        ax[row_index, 0].set_title(chem, fontsize=18, fontweight=600)
                        col_index = 1

                # iterate to the next chemical
                chem_index += 1

            ########################################################
            ###               Setup shared x-axis                ###
            ########################################################

            # set x_ticks -- need to make sure the axis is so labels go to the highest row of plots
            # first for a single chemical (one plot)
            x0, x1 = 0, n_seq + 1  # limits for xticks
            if (nrows == 1) and (ncols == 1):
                ax.set_xlim(x0, x1)
            # 2 or 3 chemicals
            elif nrows != 1 and ncols == 1:
                xticks = [t for t in ax[0].get_xticks() if t >= x0 and t <= x1]
                for r in range(nrows):
                    # make sure all plots have the same xlims
                    ax[r].set_xlim(x0, x1)
                    # if the plot below has axes on, then remove xticks
                    try:
                        if ax[r + 1].axison:
                            ax[r].set_xticks(
                                ticks=xticks,
                                labels=["" for x in xticks],
                                rotation=60,
                                fontsize=12,
                                ha="right",
                            )
                        else:
                            pass
                    except:
                        pass
            # 4 or more chemicals
            else:
                xticks = [t for t in ax[0, 0].get_xticks() if t >= x0 and t <= x1]
                for r in range(nrows):
                    for c in range(ncols):
                        # make sure all plots have the same xlims
                        ax[r, c].set_xlim(x0, x1)
                        # if the plot below has axes on, then remove xticks
                        try:
                            if ax[r + 1, c].axison:
                                ax[r, c].set_xticks(
                                    ticks=xticks,
                                    labels=["" for x in xticks],
                                    rotation=60,
                                    fontsize=12,
                                    ha="right",
                                )
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
                fig.savefig(png_title, bbox_inches="tight")

            # iterate to the next figure
            sublist_index += 1
        return

    def make_seq_scatter(
        self,
        df_in: pd.DataFrame,
        df_seq: Union[None, pd.DataFrame],
        ionization: Literal["pos", "neg"] = "pos",
        fit: bool = True,
        y_scale: Literal["linear", "log"] = "linear",
        y_step: Union[None, int] = 4,
        share_y: bool = False,
        same_frame: bool = False,
        legend: bool = True,
        dark_mode: bool = False,
    ) -> tuple[list[Any], pd.DataFrame, list[Any]]:
        """Creates a run sequence scatter plot for ESI+ or ESI-.

        This method is mainly for organization, most of the real work is
        handled by calling other methods.

        Parameters
        ----------
        df_in : pd.DataFrame
            The "tracer detection statistics" data for either ESI+ or ESI-.
        df_seq : Union[None, pd.DataFrame]
            The run sequence data.
        ionization : Literal["pos", "neg"], default "pos"
            The ionization mode, used in the plot title. Use "pos" for ESI+, or "neg" for ESI-.
        fit : bool, default True
            If True, fits a quadratic best-fit curve through each sequence that have >=3 points.
        y_scale : Literal["linear", "log"], default "linear"
            The scale of the y-axes. Use "lin" for linear scale, or "log" for logarithmic scale.
        y_step : Union[None, int], default 4
            This parameter forces a specific number of y-ticks to use. If None, matplotlib will automatically
            assign the y-ticks (this is unpredictable and may cause scaling issues - it is recommended to use
            an integer value).
        share_y : bool, default False
            If True, the y-axes for the subplots on each row will be shared.
        same_frame : bool, default False
            Determines if each figure produced will have the same shape of (nrows=4, ncols=4).
            This is useful only when > 16 plots will be made. For example, if there are 17 chemicals of interest
            and same_frame = False, then there will be one 4x4 figure and one 1x1 figure; while if same_frame = True,
            then two 4x4 figures will be made, the first will have 16 subplots, while the latter will have 1.
        legend : bool, default True
            If True, adds a legend to the top right of the figures.
        dark_mode : bool, default False
            If True, sets color scheme of figures to dark mode.

        Returns
        -------
        tuple[list, pd.DataFrame, list]
            A tuple whose first element is a listOfPNGs, second element is the df of data, and whose final element
            is a list of debug items "list of lists/dataframes/etc for debugging purposes".
        """
        ## data validation
        if ionization == "pos":
            ion_token = "+"
        elif ionization == "neg":
            ion_token = "-"
        else:
            raise ValueError(f"The `ionization` parameter must be 'pos' or 'neg', not {ionization}")

        if y_scale not in ("linear", "log"):
            raise ValueError(f"The `y_scale` parameter must be 'linear' or 'log', not {y_scale}")
        pass

        ## setup output datastructures
        listOfPNGs = []
        debug_list = []  # List of lists/dataframes/etc to export out of function for debugging purposes

        debug_list.append("Beginning of make_seq_scatter: df_in columns")
        debug_list.append(df_in.columns.values)

        ## check for sequence df
        seq_results = self._check_for_seq(df_seq, debug_list, df_in)
        df_loc_seq, order_samples, debug_list, sample_group_unique, indices_list, check_seq_abundance = seq_results

        ## clean sequence data
        df, chem_names, abundance = self._clean_seq_data(df_in, df_loc_seq, order_samples)

        ## handle dark mode / light mode options
        if dark_mode == True and self.color_mode != "dark":
            self._set_color_dict("dark")
            self._set_rcParams_dict()
            self._set_rcParams()
        elif dark_mode == False and self.color_mode == "dark":
            self._set_color_dict("light")
            self._set_rcParams_dict()
            self._set_rcParams()

        ## setup figs and axes
        figs_axes = self._make_subplots(chem_names=chem_names, same_frame=same_frame, share_y=share_y, y_scale=y_scale)

        ## now plot
        x_label = "Samples"  # label for plot
        if order_samples:
            n_seq = len(df_loc_seq)  # number of sequences
        else:
            n_seq = len(abundance)  # number of samples

        listOfPNGs = self._plot_seq_scatter(
            df,
            chem_names,
            figs_axes,
            legend,
            sample_group_unique,
            indices_list,
            x_label,
            ion_token,
            fit,
            n_seq,
            y_step,
            y_scale,
        )

        # for i, fig in enumerate(listOfPNGs):
        #     fig.savefig(f"./seq_{ionization}_{y_step}ticks-{str(i+1).zfill(2)}.png", bbox_inches="tight")

        ntaw594log = [check_seq_abundance, abundance]

        return listOfPNGs, df, debug_list, ntaw594log

    def make_loc_plot(
        self,
        data_path,
        seq_csv,
        ionization,
        y_fixed=False,
        y_step=4,
        same_frame=False,
        chemical_names=None,
        save_image=True,
        image_title=None,
        dark_mode=False,
    ):
        """
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
        """
        ##########################################################
        ###        Check file types and import data            ###
        ##########################################################

        # Find the right sheet_name based on ionization value
        if ionization == "pos":
            sn = "tracer_pos"
            ion_token = "+"  # used for plot title
        elif ionization == "neg":
            sn = "tracer_neg"
            ion_token = "-"  # used for plot title
        else:
            raise Exception('ionization parameter must be ["pos", "neg"]')

        # logic to determine file type, raise exception if not .xlsx or .csv
        if data_path[-5:] == ".xlsx":
            # create the function for importing data
            import_func = lambda file_path: pd.read_excel(file_path, sheet_name=sn)

        elif data_path[-4:] == ".csv":
            # create the function for importing data
            import_func = lambda file_path: pd.read_csv(file_path)

        else:
            raise Exception("File type must be .xlsx or .csv")

        # import data using the lambda functions defined above
        df_tracer = import_func(data_path)

        # now we should read in the csv file with the location/sequence information
        if seq_csv[-4:] == ".csv":
            df_loc_seq = pd.read_csv(seq_csv)
        else:
            raise Exception("seq_csv must be the path to a CSV file")

        ############################################################
        ###   Setting colors for plotting before cleaning data   ###
        ############################################################

        # colors for [linecolor, markercolor]
        c_aes = ["k", "k"]  # for scatter points
        c_leg_text = "white"  # for legend text
        c_leg_bg = "#f2f2f2"  # for legend background
        c_leg_ec = "#000"  # for legend edgecolor

        # if dark_mode=True, turn teal into cyan; and fix legend colors
        if dark_mode == True:
            c_aes[0] = "lime"
            c_aes[1] = "lime"
            c_leg_text = "black"
            c_leg_bg = "#333"
            c_leg_ec = "#fff"

        ################################################
        ###             Clean the data               ###
        ################################################

        # start by getting df with chemical names and abundance at each location in sequential order
        col_names = [x for x in df_loc_seq.iloc[:, 0]]
        col_names.insert(0, "Chemical_Name")
        df = df_tracer[col_names].copy()
        # need to make a column for lower cased names for sorting alpha-numerically (ignoring case)
        df["chem_name"] = df.loc[:, "Chemical_Name"].str.lower().copy()
        # df.loc[:,'Chemical_Name'] = df.loc[:,'Chemical_Name'].str.lower().copy()
        df = df.sort_values("chem_name").copy()
        df = df.drop(["chem_name"], axis=1)

        # create a list of chemical names to work with
        if chemical_names is not None:
            # if user specified a list of chemicals, only keep relavent chems in our df
            df = df[df["Chemical_Name"].isin(chemical_names)]

        # capitalize the first letter of each chemical
        df["Chemical_Name"] = df["Chemical_Name"].apply(capitalize_chems)

        # our list of final chemical names with appropriate capitalization
        chemical_names = df["Chemical_Name"]

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
        loc_tup = ("MB", "Pooled")
        loc_types = []

        for col in df.columns[1:]:
            # if we don't have the location name yet, add it to our location list
            if not col.startswith(loc_tup):
                # get the sample location name without the _* ending
                sample_temp = col.split("_", 1)[0]
                loc_types.append(sample_temp)

        # now we have all of the loc types, set MB and Pooled to be the last values for plotting order
        for l in loc_tup:
            loc_types.append(l)

        # make new generalized location df
        df_loc = pd.DataFrame({"Chemical_Name": df["Chemical_Name"]})
        df_loc[loc_types] = ""

        # update our dataframe to have new generalized location columns, whose record for a given chemical
        # is a tuple containing the abundances found at that location type
        # start by iterating through chemicals
        for index, row in df.iterrows():
            chem_loc_dic = {x: [] for x in loc_types}
            # now iterate throught locations for this chemical
            for i, location in enumerate(row[1:].index):
                i += 1  # iterate i so that its index corresponds to the correct column in df
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
            figs_axes = make_subplots(
                chem_names,
                y_max,
                y_min,
                y_steps=y_step,
                same_frame=same_frame,
                dark_mode=dark_mode,
            )
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
        tick_pos = [x for x in range(1, len(tick_labels) + 1)]
        # plot each chemical in its respective subplot
        # start by iterating through your sublists of chemicals (in groups of 16 or less)
        sublist_index = 0
        chem_index = 0  # index for pulling information from primary df
        while sublist_index < len(chem_names):
            chem_sublist = chem_names[sublist_index]
            fig, ax, shape, subtitle = (
                figs_axes[sublist_index][0],
                figs_axes[sublist_index][1],
                figs_axes[sublist_index][2],
                figs_axes[sublist_index][3],
            )
            # set x and y labels, and plot title
            fig.tight_layout(h_pad=1)
            fig.text(0.5, -0.07, x_label, ha="center", va="center", fontsize=28)
            fig.text(
                -0.015,
                0.5,
                "Abundance",
                ha="center",
                va="center",
                rotation="vertical",
                fontsize=28,
            )
            title = "Abundance vs. Location\n"
            sub = f"{subtitle}, ESI{ion_token}"
            fig.text(0.05, 1.045, title, fontsize=32)
            fig.text(0.05, 1.045, sub, fontsize=26)

            nrows, ncols = shape[0], shape[1]  # shape of subpot axis

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
            row_index, col_index = (
                0,
                0,
            )  # indices for which subplot to put a chemical in
            while (chem_index - 16 * sublist_index) < ax_count:  # and (chem_index < n_chems):
                # for loop to set up x and y values for each plot in a figure
                x_values, y_values = [], []
                for i, v in enumerate(df.iloc[chem_index, 1:]):
                    y_values.append(v)
                    x_app = [i + 1 for x in range(len(v))]
                    x_values.append(x_app)

                # first deal with a single chemical
                if (nrows == 1) and (ncols == 1):
                    # while loop to actually plot all of the x and yvalues on a plot
                    i = 0
                    while i < len(x_values):
                        # plot, set ticks to be blank to immitate share_x option, set chemical name as plot title
                        ax.plot(
                            x_values[i],
                            y_values[i],
                            color=c_aes[0],
                            marker="o",
                            markersize=10,
                            markeredgecolor=c_aes[1],
                            markerfacecolor=c_aes[1],
                        )
                        ax.set_xticks(
                            ticks=tick_pos,
                            labels=tick_labels,
                            rotation=50,
                            fontsize=18,
                            ha="right",
                        )
                        ax.set_xlim(0, tick_pos[-1] + 0.75)
                        ax.set_title(chemical_names.iloc[chem_index], fontsize=18, fontweight=600)
                        i += 1  # iterate to next location for plotting this chemical

                # now deal with 2 and 3 chemicals
                elif (nrows != 1) and (ncols == 1):
                    i = 0
                    while i < len(x_values):
                        # plot, set ticks to be blank to immitate share_x option, set chemical name as plot title
                        ax[row_index].plot(
                            x_values[i],
                            y_values[i],
                            color=c_aes[0],
                            marker="o",
                            markersize=10,
                            markeredgecolor=c_aes[1],
                            markerfacecolor=c_aes[1],
                            lw=2,
                        )
                        ax[row_index].set_xticks(
                            ticks=tick_pos,
                            labels=tick_labels,
                            rotation=50,
                            fontsize=18,
                            ha="right",
                        )
                        ax[row_index].set_xlim(0, tick_pos[-1] + 0.75)
                        ax[row_index].set_title(chemical_names.iloc[chem_index], fontsize=18, fontweight=600)
                        # set up the shared x-axis
                        if row_index == nrows - 1:
                            ax[row_index].set_xticks(
                                ticks=tick_pos,
                                labels=tick_labels,
                                rotation=50,
                                fontsize=18,
                                ha="right",
                            )
                        else:
                            ax[row_index].set_xticks(
                                ticks=tick_pos,
                                labels=["" for x in tick_labels],
                                rotation=0,
                                fontsize=1,
                                ha="right",
                            )
                        i += 1  # iterate to next location for plotting this chemical
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
                        ax[row_index, col_index].plot(
                            x_values[i],
                            y_values[i],
                            color=c_aes[0],
                            marker="o",
                            markersize=4,
                            markeredgecolor=c_aes[1],
                            markerfacecolor=c_aes[1],
                            lw=2,
                        )
                        ax[row_index, col_index].set_xticks(
                            ticks=tick_pos,
                            labels=["" for x in tick_labels],
                            rotation=0,
                            fontsize=1,
                            ha="right",
                        )
                        ax[row_index, col_index].set_xlim(0, tick_pos[-1] + 0.75)
                        ax[row_index, col_index].set_title(chemical_names.iloc[chem_index], fontsize=18, fontweight=600)
                        i += 1  # iterate to next location for plotting this chemical

                    # set x_ticks -- need to make sure the axis is so labels go to the highest row of plots
                    for c in range(ncols):
                        if ax[-1, c].axison:
                            ax[-1, c].set_xticks(
                                ticks=tick_pos,
                                labels=tick_labels,
                                rotation=60,
                                fontsize=12,
                                ha="right",
                            )
                        elif ax[-2, c].axison:
                            ax[-2, c].set_xticks(
                                ticks=tick_pos,
                                labels=tick_labels,
                                rotation=60,
                                fontsize=12,
                                ha="right",
                            )
                        elif ax[-3, c].axison:
                            ax[-3, c].set_xticks(
                                ticks=tick_pos,
                                labels=tick_labels,
                                rotation=60,
                                fontsize=12,
                                ha="right",
                            )
                        else:
                            ax[-4, c].set_xticks(
                                ticks=tick_pos,
                                labels=tick_labels,
                                rotation=60,
                                fontsize=12,
                                ha="right",
                            )

                chem_index += 1  # iterate to the next chemical for plotting
                col_index += 1  # iterate to the next plot in this figure

            # save image and iterate to the next sublist of 16 or fewer chemicals
            if save_image:
                if image_title is not None:
                    png_title = f"{image_title}_"
                else:
                    png_title = f"ab_vs_seq_ESI_{ion_token}_"
                png_title += f"{sublist_index}".zfill(3)
                fig.savefig(png_title, bbox_inches="tight")

            # iterate to the next figure
            sublist_index += 1
        return


def capitalize_chems(s):
    """
    Function that takes in a string, and will capitalize the first letter of the compound
    after any hyphenated values
    Will ignore certain fully alphabetical prefixes ['alpha-', 'beta-', 'ortho-', 'cis-', '(R)-', etc]
    e.g., if s = 13C,d2-hydrochlorothiazide; this function returns 13C,d2-Hydrochlorothiazide
          if s = 2-hydroxy-ibuprofen; this function returns 2-Hydroxy-ibuprofen
          if s = alpha-glucose; tjis function returns alpha-Glucose
    ----------------------------------------------------------------------------------------
    s (str):
        The string that will be capitalized
    """
    # create a list of prefixes to ignore
    ignore_list = [
        "ortho",
        "meta",
        "para",
        "N",
        "O",
        "alpha",
        "beta",
        "gamma",
        "sec",
        "tert",
        "cis",
        "trans",
        "(E)",
        "(Z)",
        "(R)",
        "(S)",
        "D",
        "L",
    ]

    # split the chemical name by hyphens
    s_parts = s.split("-")

    # iterate through our substrings the find and capitalize the first compound name to capitalize
    # This is done by finding the first substring that does not contain a number and does not show
    # up in our ignore list
    for i, s_sub in enumerate(s_parts):
        if not ((s_sub in ignore_list) or any(chr.isdigit() for chr in s_sub)):
            s_parts[i] = s_sub[0].upper() + s_sub[1:]
            return "-".join(s_parts)


def make_subplots(
    chem_names,
    y_max=None,
    y_min=None,
    y_steps=4,
    share_y=True,
    y_scale="log",
    same_frame=False,
    dark_mode=False,
):
    """
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
    """

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
        # axes params
        plt.rcParams.update(
            {
                "axes.facecolor": "#0d0d0d",
                "axes.edgecolor": "#fff",
                "axes.titlecolor": "#fff",
            }
        )
        # tick params
        plt.rcParams.update({"xtick.color": "#fff", "ytick.color": "#fff"})
        # figure params
        plt.rcParams.update({"figure.facecolor": "#000"})
        # text params
        plt.rcParams.update({"text.color": "#fff"})

    # now parameters for light_mode (dark_mode=False)
    else:
        # axes params
        plt.rcParams.update(
            {
                "axes.facecolor": "#fff",
                "axes.edgecolor": "#000",
                "axes.titlecolor": "#000",
            }
        )
        # tick params
        plt.rcParams.update({"xtick.color": "#000", "ytick.color": "#000"})
        # figure params
        plt.rcParams.update({"figure.facecolor": "#e6e6e6"})
        # text params
        plt.rcParams.update({"text.color": "#000"})

    # now set params that are needed for dark and light mode
    # font params
    plt.rcParams.update({"font.family": "serif"})
    # tick params
    plt.rcParams.update(
        {
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )

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
            fig, ax = plt.subplots(
                nrows=4,
                ncols=4,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems == 1:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems == 2:
            fig, ax = plt.subplots(
                nrows=2,
                ncols=1,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems == 3:
            fig, ax = plt.subplots(
                nrows=3,
                ncols=1,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems == 4:
            fig, ax = plt.subplots(
                nrows=2,
                ncols=2,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems in [5, 6]:
            fig, ax = plt.subplots(
                nrows=2,
                ncols=3,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems in [7, 8, 9]:
            fig, ax = plt.subplots(
                nrows=3,
                ncols=3,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems in [7, 8, 9]:
            fig, ax = plt.subplots(
                nrows=3,
                ncols=3,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )
        elif n_chems in [10, 11, 12]:
            fig, ax = plt.subplots(
                nrows=3,
                ncols=4,
                sharex=False,
                sharey=share_y,
                subplot_kw={"yscale": y_scale},
            )

        ######################################################
        ###        Setting figure/axes aesthetics          ###
        ######################################################

        # need to handle 1x1 plot differently since axes object isnt an array
        if (n_chems == 1) and (same_frame == False):
            # generate the subtitle for the plot to state which chemicals are being plotted
            subtitle = f"Chemical {subtitle_i+1}"
            # turn on grid and get shape
            ax.grid()
            shape = (1, 1)
            # if y_fixed == True then set the ticks
            if (y_max is not None) and (y_min is not None):
                # Need to find the floor-nearest-power-of-ten for ymax and ymin
                y_max_pow = int(np.floor(np.log10(y_max)))
                y_min_pow = int(np.floor(np.log10(y_min)))
                n = y_steps  # number of ticks on y-axis
                # set y value ranges
                ax.set_ylim([10 ** (y_min_pow - 1), 10 ** (y_max_pow + 1)])
                # pick the tick locations and labels
                ax.set_yticks(
                    ticks=[10**x for x in np.linspace(y_min_pow, y_max_pow, n)],
                    labels=[f"$10^{{{int(x)}}}$" for x in np.linspace(y_min_pow, y_max_pow, n)],
                )
        else:
            # generate the subtitle for the plot to state which chemicals are being plotted
            subtitle_f = subtitle_i + n_chems
            subtitle = f"Chemicals {subtitle_i+1}-{subtitle_f}"
            subtitle_i = subtitle_f
            # get shape of axis object
            axe = ax.ravel()  # have to unpack gridspec object (from subplots() function)
            gs = axe[0].get_gridspec()
            shape = (gs.nrows, gs.ncols)

            # remove unused subplot axes
            for j in range(n_chems, gs.ncols * gs.nrows):
                axe[j].set_axis_off()

            # setting axes grids and ticks
            for j in range(0, n_chems):
                # turn on the grids
                axe[j].grid()

                # if y_fixed == True then set the ticks
                # if (y_max is not None) and (y_min is not None):
                # Need to find the floor-nearest-power-of-ten for ymax and ymin
                y_max_pow = int(np.floor(np.log10(y_max)))
                y_min_pow = int(np.floor(np.log10(y_min)))
                n = y_steps  # number of ticks on y-axis
                # set y value ranges
                axe[j].set_ylim([10 ** (y_min_pow - 1), 10 ** (y_max_pow + 1)])
                # pick the tick locations and labels
                axe[j].set_yticks(
                    ticks=[10**x for x in np.linspace(y_min_pow, y_max_pow, n)],
                    labels=[f"$10^{{{int(x)}}}$" for x in np.linspace(y_min_pow, y_max_pow, n)],
                )

        # a few more fig aesthetics, append our list of tupples to return, then iterate to next figure
        fig.set_size_inches(14, 8)
        # if share y is not on, we need extra space between plots horizontally (width w_pad)
        if share_y == False and y_scale == "linear":
            fig.tight_layout(h_pad=1, w_pad=3.2)
        else:
            fig.tight_layout(h_pad=1)
        figs_axes.append((fig, ax, shape, subtitle))
        print(type(fig), type(ax[0][0]), type(shape), type(subtitle))
        i += 1

    return figs_axes
