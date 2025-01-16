"""
Class for producing the CV scatterplot for the WebApp
2024/05/07
TMF
"""

from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import traceback

# from .functions_Universal_v3 import parse_headers
from .task_functions import parse_headers
import io
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.ms1")
logger.setLevel(logging.INFO)

# import seaborn as sns
try:
    import seaborn as sns
except ModuleNotFoundError:
    logger.error("Seaborn is not installed. Please run 'pip install seaborn' to install it.")


def plot_cv_scatterplot(
    dfs,
    tracer_df,
    max_replicate_cv_value,
):
    # Create class object, cv_scatterplot
    plot = CV_Scatterplot(
        dfs,
        tracer_df,
        max_replicate_cv_value,
    )
    # Try to plot scatterplot
    try:
        output = plot.execute()
    # Raise exception if plotting scatterplot fails
    except Exception as e:
        trace = traceback.format_exc()
        logger.info(trace)
        fail_step = plot.get_step()
        plot.set_status("Failed on step: " + fail_step)
        error = repr(e)
        plot.set_except_message(error)
        raise e

    return output


class CV_Scatterplot:
    """
    A Class object that results in the plotting of the CV scatterplot.

    Inputs:
        dfs (list of Pandas dataframes; Pos and Neg dataframes from NTARun)
        tracer_df (Pandas dataframe; optionally input tracer information)
        max_replicate_cv (float; value for CV flags)
    Outputs:
        graphic_output (IDK; some file type that can be stored in NTARun's self.data_map dictionary)
    """

    def __init__(
        self,
        dfs=None,
        tracer_df=None,
        max_replicate_cv_value=None,
    ):
        self.dfs = dfs
        self.tracer_df = tracer_df
        self.max_replicate_cv = max_replicate_cv_value
        self.except_message = None
        self.graphic_output = None

    def execute(self):
        # Set defaults
        plt.rcdefaults()
        # Set title
        titleText = "CV vs. Abundance"
        # Get user input CV threshold, convert to float
        max_replicate_cv_value = self.max_replicate_cv
        max_replicate_cv_value = float(max_replicate_cv_value)
        # get 'Tracer Detection Statistics' if it exists else None
        dfTracer = self.tracer_df
        # Add conditional; if tracer exists reformat
        if dfTracer is not None:
            tracers = dfTracer[["Observed Mass", "Observed Retention Time"]].copy()
            tracers.rename({"Observed Mass": "Mass"}, axis=1, inplace=True)
            tracers.rename({"Observed Retention Time": "Retention Time"}, axis=1, inplace=True)
            tracers["spike"] = 1
            logger.info("cv scatterplot tracers columns= {}".format(tracers.columns.values))
        # combine the two input dataframes, ignore non-existing dataframes
        dfCombined = (
            pd.concat([self.dfs[0], self.dfs[1]], axis=0, ignore_index=True, sort=False)
            if self.dfs[0] is not None and self.dfs[1] is not None
            else self.dfs[0]
            if self.dfs[0] is not None
            else self.dfs[1]
            if self.dfs[1] is not None
            else None
        )
        # Get sample headers
        all_headers = parse_headers(dfCombined)
        non_samples = ["MRL"]
        sam_headers = [
            sublist[0][:-1]
            for sublist in all_headers
            if len(sublist) > 1
            if not any(x in sublist[0] for x in non_samples)
        ]
        # Isolate sample_groups from stats columns
        prefixes = ["Mean ", "Median ", "CV ", "STD ", "Detection Count ", "Detection Percentage "]
        sample_groups = [item for item in sam_headers if not any(x in item for x in prefixes)]
        # Find CV cols from df, subset cv_df from df
        cv_cols = ["CV " + col for col in sample_groups]
        cv_df = dfCombined[cv_cols]
        # Find CV cols from df, subset cv_df from df
        mean_cols = ["Mean " + col for col in sample_groups]
        mean_df = dfCombined[mean_cols]
        # Carry over Mass and Retention_Time
        cv_df["Mass"] = dfCombined["Mass"]
        cv_df["Retention Time"] = dfCombined["Retention Time"]
        # AC 2/8/2024 Get minimum and maximum abundance values of dataframe (mean columns) for the purposes of setting the x-axis range
        min_abundance_value = mean_df.min(numeric_only=True).min()
        max_abundance_value = mean_df.max(numeric_only=True).max()
        if (
            min_abundance_value == 0
        ):  # If minimum abundance value is zero, then set minimum limit to zero (to avoid log issues on zero)
            min_abundance_limit = 0
        else:
            min_abundance_limit = 10 ** math.floor(math.log10(min_abundance_value))
        max_abundance_limit = 10 ** math.ceil(math.log10(max_abundance_value))
        # Create list, define blank strings
        li = []
        blanks = ["MB1", "BLK", "Blank", "BLANK", "blank", "MB", "mb"]
        # Loop through sample groups
        for x in sample_groups:
            # Take each sample's CV and mean, store in dummy variable
            cv = "CV " + x
            mean = "Mean " + x
            dum = pd.concat([cv_df[cv], mean_df[mean]], axis=1)
            dum.rename({cv: "CV"}, axis=1, inplace=True)
            dum.rename({mean: "Mean"}, axis=1, inplace=True)
            dum["sample"] = x
            dum["Mass"] = cv_df["Mass"]
            dum["Retention Time"] = cv_df["Retention Time"]
            # Add sample type (blank or sample)
            if any(i in x for i in blanks):
                dum["type"] = "blank"
            else:
                dum["type"] = "sample"
            # Append to list
            li.append(dum)

        # Concatenate plot, drop NAs
        plot = pd.concat(li)
        plot.dropna(axis=0, subset=["CV", "Mean"], how="any", inplace=True)
        logger.info("cv scatterplot plot columns= {}".format(plot.columns.values))

        # Conditional for if tracers are present:
        if dfTracer is not None:
            # Merge df with tracers to get labels
            plot2 = pd.merge(plot, tracers, how="left", on=["Mass", "Retention Time"])
        else:
            # If tracer plot doesn't exist, still need to create a spike column that is empty
            plot["spike"] = ""
            plot2 = plot.copy()

        plot2.replace(np.nan, 0, inplace=True)
        # Define subplots, set height and width
        f, axes = plt.subplots(1, 2)
        f.set_figheight(5)
        f.set_figwidth(15)
        # Set palette
        palette = ["whitesmoke", "firebrick"]
        sns.set_palette(palette, 2)
        # Blank plot
        a = sns.scatterplot(
            data=plot2.loc[((plot2["type"] == "blank")), :].sort_values("spike"),
            x="Mean",
            y="CV",
            hue="spike",
            edgecolor="black",
            alpha=0.5,
            ax=axes[0],
        )
        # Add CV red dashed line
        a.axhline(
            y=max_replicate_cv_value,
            color="red",
            linestyle="dashed",
            linewidth=1.5,
            alpha=1,
        )
        a.text(
            max_abundance_limit / 5,
            max_replicate_cv_value + 0.1,
            "CV = {}".format(max_replicate_cv_value),
            ha="center",
            va="center_baseline",
            weight="bold",
            size=14,
        )
        # Perform occurrence counts above and below CV by sample type
        red_count = len(plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 1)), "CV"])
        red_flag_count = sum(
            plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 1)), "CV"] > max_replicate_cv_value
        )

        white_count = len(plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 0)), "CV"])
        white_flag_count = sum(
            plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 0)), "CV"] > max_replicate_cv_value
        )

        # Only generate legend if tracers are submitted -- THIS ISN'T TRUE RIGHT NOW
        legend = a.legend(title="Unfiltered Occurrences", fontsize=14, title_fontsize=16)
        # Set legend labels
        if dfTracer is not None:
            legend.get_texts()[0].set_text(f"unknowns ({white_flag_count} of {white_count} above line)")
            try:
                legend.get_texts()[1].set_text(
                    f"tracers ({red_flag_count} of {red_count} above line)"
                )  # If tracers are present, add secondary legend label
            except IndexError:  # If no tracers found in blanks, set alternate legend
                # legend.set_text("tracers 0 of 0 above line)")
                pass
        # Make it pretty
        frame = legend.get_frame()  # sets up for color, edge, and transparency
        frame.set_facecolor("lightgray")  # color of legend
        frame.set_edgecolor("black")  # edge color of legend
        frame.set_alpha(1)  # deals with transparency
        # Adjust axes labels
        axes[0].set_title(titleText + ": Blanks", fontsize=18, weight="bold")
        axes[0].set_xlabel("Mean Abundance", fontsize=14)
        axes[0].set_ylabel("CV", fontsize=14)
        axes[0].set_ylim(0, 2.5)
        axes[0].set_xlim(
            min_abundance_limit, max_abundance_limit
        )  # Set x-axis to scale based on the min/max data points
        axes[0].set(xscale="log")
        axes[0].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        axes[0].tick_params(axis="both", which="both", labelsize=12)

        # Sample plot
        b = sns.scatterplot(
            data=plot2.loc[((plot2["type"] != "blank")), :].sort_values("spike"),
            x="Mean",
            y="CV",
            hue="spike",
            edgecolor="black",
            alpha=0.5,
            ax=axes[1],
        )
        # Add CV red dashed line
        b.axhline(
            y=max_replicate_cv_value,
            color="red",
            linestyle="dashed",
            linewidth=1.5,
            alpha=1,
        )
        b.text(
            max_abundance_limit / 5,
            max_replicate_cv_value + 0.1,
            "CV = {}".format(max_replicate_cv_value),
            ha="center",
            va="center_baseline",
            weight="bold",
            size=14,
        )
        # Perform occurrence counts above and below CV by sample type
        red_count = len(plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 1)), "CV"])
        red_flag_count = sum(
            plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 1)), "CV"] > max_replicate_cv_value
        )
        white_count = len(plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 0)), "CV"])
        white_flag_count = sum(
            plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 0)), "CV"] > max_replicate_cv_value
        )
        # Only generate legend if tracers are submitted -- THIS ISN'T TRUE RIGHT NOW
        legend = b.legend(title="Unfiltered Occurrences", fontsize=14, title_fontsize=16)
        # Set legend labels
        if dfTracer is not None:
            legend.get_texts()[0].set_text(f"unknowns ({white_flag_count} of {white_count} above line)")
            legend.get_texts()[1].set_text(
                f"tracers ({red_flag_count} of {red_count} above line)"
            )  # If tracers are present, add secondary legend label
        # Make it pretty
        frame = legend.get_frame()  # sets up for color, edge, and transparency
        frame.set_facecolor("lightgray")  # color of legend
        frame.set_edgecolor("black")  # edge color of legend
        frame.set_alpha(1)  # deals with transparency
        # Adjust axes labels
        axes[1].set_title(titleText + ": Non-blanks", fontsize=18, weight="bold")
        axes[1].set_xlabel("Mean Abundance", fontsize=14)
        axes[1].set_ylabel("CV", fontsize=14)
        axes[1].set_ylim(0, 2.5)
        axes[1].set_xlim(min_abundance_limit, max_abundance_limit)
        axes[1].set(xscale="log")
        axes[1].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        axes[1].tick_params(axis="both", which="both", labelsize=12)
        # Convert the plot to a bytes-like object
        buffer = io.BytesIO()
        plt.savefig(buffer)
        buffer.seek(0)
        # Store in class variable
        self.cv_scatterplots_out.append(buffer.getvalue())
        # Map to outputs
        self.cv_scatterplot_map["cv_scatterplot"] = self.cv_scatterplots_out[0]
        project_name = self.parameters["project_name"][1]
        self.gridfs.put(
            "&&".join(self.cv_scatterplot_map.keys()),
            _id=self.jobid + "_cv_scatterplots",
            encoding="utf-8",
            project_name=project_name,
        )
        self.mongo_save(self.cv_scatterplot_map["cv_scatterplot"], step="cv_scatterplot")
        # reset plt
        plt.clf()
        return
