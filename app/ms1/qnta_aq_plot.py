# Function for producing the CV scatterplot for INTERPRET NTA

from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import traceback

from .task_functions import parse_headers
import io
import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches

logger = logging.getLogger("nta_app.ms1")

# import seaborn as sns
try:
    import seaborn as sns
except ModuleNotFoundError:
    logger.error("Seaborn is not installed. Please run 'pip install seaborn' to install it.")



def AQ_plots(validation_out, long_form=False, LOO=True):
    """
    Creates plots of the Accuracy Quotient (AQ) qNTA performance metric,
    including a boxplot of AQ values and a scatterplot of AQ vs. ConcTargeted.

    Parameters
    ----------
    validation_out : pandas DataFrame
        DataFrame output by method RF_boot_validation. Contains columns for AQ per sample (wide form)
        or one AQ column with per-sample, per-chemical values per row (long form).
    long_form : Boolean, optional
        Indicates whether the validation_out DataFrame is wide or long form. The default is False.
    LOO : Boolean, optional
        Indicates whether AQ values from leave-one-out qNTA estimates are present in the validation_out DataFrame. The default is True.


    Returns
    -------
    Seaborn figure object

    """
    AQ_plot_data = validation_out.copy()
    # Change to long form if in wide form
    if long_form is False:
        # Get targeted concentration columns and make ConcTargeted column
        conc_cols = AQ_plot_data.columns[AQ_plot_data.columns.str.endswith("_")].tolist()
        conc_data = AQ_plot_data.copy().loc[:, ["Chemical Name"] + conc_cols]
        targeted_cols = pd.melt(
            conc_data, id_vars="Chemical Name", value_vars=conc_cols, var_name="Sample", value_name="ConcTargeted"
        )
        targeted_cols["Sample"] = [s[:-1] for s in targeted_cols["Sample"]]
        # Change from wide to long form
        AQ_plot_data = pd.melt(
            AQ_plot_data,
            id_vars="Chemical Name",
            value_vars=AQ_plot_data.columns[AQ_plot_data.columns.str.contains("AQ")].tolist(),
            var_name="SampleMetric",
            value_name="AQ",
        )
        AQ_plot_data[["Sample", "MetricName"]] = AQ_plot_data["SampleMetric"].str.split("__", expand=True)
        AQ_plot_data = AQ_plot_data.drop(columns=["SampleMetric"]).pivot(
            index=["Chemical Name", "Sample"], columns="MetricName"
        )
        AQ_plot_data.columns = AQ_plot_data.columns.droplevel(0)
        AQ_plot_data = AQ_plot_data.reset_index()
        # Add ConcTargeted column
        AQ_plot_data = pd.merge(AQ_plot_data, targeted_cols, on=["Chemical Name", "Sample"], how="left")

    # Remove AQ 0 (AAQ Inf)
    AQ_plot_data = AQ_plot_data[AQ_plot_data["AQ"] > 0]

    # Add leave-one-out AQs if available
    if LOO:
        # Format dataframe for plotting
        cols = ["AQ", "AQ (LOO)"]
        plot2 = (
            AQ_plot_data[["Feature ID", "Sample", "ConcTargeted", "AQ", "AQ_LOO"]]
            .copy()
            .rename(columns={"AQ_LOO": "AQ (LOO)"})
        )
        plot = pd.melt(
            plot2, id_vars=["Feature ID", "Sample"], value_vars=cols, var_name="Metric", value_name="Value"
        )

        """SEABORN ATTEMPT"""
        # Instantiate subplots
        f, ax = plt.subplots(1, 2)
        # Set figure params
        f.set_figheight(5)
        f.set_figwidth(15)
        # Set style
        sns.set_style("ticks")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        palette = ["firebrick", "darkgoldenrod"]
        sns.set_palette(palette, 2)
        # Boxplot
        # First axis plot
        a = sns.boxplot(
            data=plot,
            x="Metric",
            y="Value",
            hue="Metric",
            width=0.5,
            whis=(5, 95),
            fliersize=0,
            linewidth=2,
            boxprops=dict(alpha=0.25),
            ax=ax[0],
        )
        b = sns.stripplot(
            data=plot,
            x="Metric",
            y="Value",
            hue="Metric",
            size=7,
            edgecolor="black",
            linewidth=1,
            alpha=0.75,
            ax=ax[0],
        )
        # Modify plot
        a.set(yscale="log")
        a.set_xlabel("Metric", fontsize=16)
        a.set_ylabel("Value ($log_{10}$ scale)", fontsize=16)
        a.set_title("AQ Distributions", fontsize=18)
        a.tick_params(axis="y", labelsize=14)
        a.tick_params(axis="x", labelsize=14)

        # Scatterplot
        c = sns.scatterplot(
            data=plot2,
            x="ConcTargeted",
            y="AQ",
            color="firebrick",
            s=50,
            edgecolor="black",
            linewidth=1,
            alpha=0.75,
            ax=ax[1],
        )
        d = sns.scatterplot(
            data=plot2,
            x="ConcTargeted",
            y="AQ (LOO)",
            color="darkgoldenrod",
            s=50,
            edgecolor="black",
            linewidth=1,
            alpha=0.75,
            ax=ax[1],
        )
        # Modify plot
        c.set(xscale="log", yscale="log")
        c.set_xlabel("Targeted Concentration ($log_{10}$ scale)", fontsize=16)
        c.set_ylabel("AQ Value ($log_{10}$ scale)", fontsize=16)
        c.set_title("Concentrations vs AQ Values", fontsize=18)
        c.tick_params(axis="y", labelsize=14)
        c.tick_params(axis="x", labelsize=14)
        # Legend
        red_patch = mpatches.Patch(facecolor="firebrick", label="AQ", edgecolor="black")
        yellow_patch = mpatches.Patch(facecolor="darkgoldenrod", label="AQ (LOO)", edgecolor="black")
        legend = c.legend(handles=[red_patch, yellow_patch], loc="upper right", fontsize=14)
        frame = legend.get_frame()  # sets up for color, edge, and transparency
        frame.set_facecolor("lightgray")  # color of legend
        frame.set_edgecolor("black")  # edge color of legend
        frame.set_alpha(1)

    else:
        """Need to convert this plot to Seaborn for non-LOO scenario"""
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
        # Boxplot
        ax1.boxplot(AQ_plot_data["AQ"])
        # Add points with x scatter
        ax1.scatter(np.random.normal(1, 0.01, len(AQ_plot_data)), AQ_plot_data["AQ"], alpha=0.6)
        ax1.set_title("Boxplot of $log_{10}$ (AQ)")
        ax1.set_yscale("log")
        # Scatter plot
        ax2.scatter(AQ_plot_data["ConcTargeted"], AQ_plot_data["AQ"])
        ax2.set_title("Scatterplot of log10(AQ) vs. log10(ConcTargeted)")
        ax2.set_xscale("log")
        ax2.set_yscale("log")

    # Set plot layout
    plt.tight_layout()
    # Convert the plot to a bytes-like object
    buffer = io.BytesIO()
    plt.savefig(buffer)
    buffer.seek(0)
    # reset plt
    plt.clf()
    return buffer.getvalue()