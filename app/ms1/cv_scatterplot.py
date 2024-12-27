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

# def cv_scatterplot(self, input_dfs):
#     # Set defaults
#     plt.rcdefaults()
#     # Set title
#     titleText = "CV vs. Abundance"
#     # Get user input CV threshold, convert to float
#     max_replicate_cv_value = self.parameters["max_replicate_cv"][1]
#     max_replicate_cv_value = float(max_replicate_cv_value)

#     # get dataframe 'All Detection Statistics (Pos)' if it exists else None
#     dfPos = (
#         self.data_map["All Detection Statistics (Pos)"] if "All Detection Statistics (Pos)" in self.data_map else None
#     )
#     # get dataframe 'All Detection Statistics (Neg)' if it exists else None
#     dfNeg = (
#         self.data_map["All Detection Statistics (Neg)"] if "All Detection Statistics (Neg)" in self.data_map else None
#     )
#     # get 'Tracer Detection Statistics' if it exists else None
#     dfTracer = self.data_map["Tracer Detection Statistics"] if "Tracer Detection Statistics" in self.data_map else None
#     # Add conditional; if tracer exists reformat
#     if dfTracer is not None:
#         tracers = dfTracer[["Observed Mass", "Observed Retention Time"]].copy()
#         tracers.rename({"Observed Mass": "Mass"}, axis=1, inplace=True)
#         tracers.rename({"Observed Retention Time": "Retention Time"}, axis=1, inplace=True)
#         tracers["spike"] = 1
#         logger.info("cv scatterplot tracers columns= {}".format(tracers.columns.values))
#     # combine the two dataframes, ignore non-existing dataframes
#     dfCombined = (
#         pd.concat([dfPos, dfNeg], axis=0, ignore_index=True, sort=False)
#         if dfPos is not None and dfNeg is not None
#         else dfPos
#         if dfPos is not None
#         else dfNeg
#         if dfNeg is not None
#         else None
#     )
#     # Get sample headers
#     all_headers = parse_headers(dfCombined)
#     non_samples = ["MRL"]
#     sam_headers = [
#         sublist[0][:-1] for sublist in all_headers if len(sublist) > 1 if not any(x in sublist[0] for x in non_samples)
#     ]
#     # Isolate sample_groups from stats columns
#     prefixes = ["Mean ", "Median ", "CV ", "STD ", "Detection Count ", "Detection Percentage "]
#     sample_groups = [item for item in sam_headers if not any(x in item for x in prefixes)]
#     # Find CV cols from df, subset cv_df from df
#     cv_cols = ["CV " + col for col in sample_groups]
#     cv_df = dfCombined[cv_cols]
#     # Find CV cols from df, subset cv_df from df
#     mean_cols = ["Mean " + col for col in sample_groups]
#     mean_df = dfCombined[mean_cols]
#     # Carry over Mass and Retention_Time
#     cv_df["Mass"] = dfCombined["Mass"]
#     cv_df["Retention Time"] = dfCombined["Retention Time"]
#     # AC 2/8/2024 Get minimum and maximum abundance values of dataframe (mean columns) for the purposes of setting the x-axis range
#     min_abundance_value = mean_df.min(numeric_only=True).min()
#     max_abundance_value = mean_df.max(numeric_only=True).max()
#     if (
#         min_abundance_value == 0
#     ):  # If minimum abundance value is zero, then set minimum limit to zero (to avoid log issues on zero)
#         min_abundance_limit = 0
#     else:
#         min_abundance_limit = 10 ** math.floor(math.log10(min_abundance_value))
#     max_abundance_limit = 10 ** math.ceil(math.log10(max_abundance_value))
#     # Create list, define blank strings
#     li = []
#     blanks = ["MB1", "BLK", "Blank", "BLANK", "blank", "MB", "mb"]
#     # Loop through sample groups
#     for x in sample_groups:
#         # Take each sample's CV and mean, store in dummy variable
#         cv = "CV " + x
#         mean = "Mean " + x
#         dum = pd.concat([cv_df[cv], mean_df[mean]], axis=1)
#         dum.rename({cv: "CV"}, axis=1, inplace=True)
#         dum.rename({mean: "Mean"}, axis=1, inplace=True)
#         dum["sample"] = x
#         dum["Mass"] = cv_df["Mass"]
#         dum["Retention Time"] = cv_df["Retention Time"]
#         # Add sample type (blank or sample)
#         if any(i in x for i in blanks):
#             dum["type"] = "blank"
#         else:
#             dum["type"] = "sample"
#         # Append to list
#         li.append(dum)

#     # Concatenate plot, drop NAs
#     plot = pd.concat(li)
#     plot.dropna(axis=0, subset=["CV", "Mean"], how="any", inplace=True)
#     logger.info("cv scatterplot plot columns= {}".format(plot.columns.values))

#     # Conditional for if tracers are present:
#     if dfTracer is not None:
#         # Merge df with tracers to get labels
#         plot2 = pd.merge(plot, tracers, how="left", on=["Mass", "Retention Time"])
#     else:
#         # If tracer plot doesn't exist, still need to create a spike column that is empty
#         plot["spike"] = ""
#         plot2 = plot.copy()

#     plot2.replace(np.nan, 0, inplace=True)
#     # Define subplots, set height and width
#     f, axes = plt.subplots(1, 2)
#     f.set_figheight(5)
#     f.set_figwidth(15)
#     # Set palette
#     palette = ["whitesmoke", "firebrick"]
#     sns.set_palette(palette, 2)
#     # Blank plot
#     a = sns.scatterplot(
#         data=plot2.loc[((plot2["type"] == "blank")), :].sort_values("spike"),
#         x="Mean",
#         y="CV",
#         hue="spike",
#         edgecolor="black",
#         alpha=0.5,
#         ax=axes[0],
#     )
#     # Add CV red dashed line
#     a.axhline(
#         y=max_replicate_cv_value,
#         color="red",
#         linestyle="dashed",
#         linewidth=1.5,
#         alpha=1,
#     )
#     a.text(
#         max_abundance_limit / 5,
#         max_replicate_cv_value + 0.1,
#         "CV = {}".format(max_replicate_cv_value),
#         ha="center",
#         va="center_baseline",
#         weight="bold",
#         size=14,
#     )
#     # Perform occurrence counts above and below CV by sample type
#     red_count = len(plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 1)), "CV"])
#     red_flag_count = sum(plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 1)), "CV"] > max_replicate_cv_value)

#     white_count = len(plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 0)), "CV"])
#     white_flag_count = sum(
#         plot2.loc[((plot2["type"] == "blank") & (plot2["spike"] == 0)), "CV"] > max_replicate_cv_value
#     )

#     # Only generate legend if tracers are submitted -- THIS ISN'T TRUE RIGHT NOW
#     legend = a.legend(title="Unfiltered Occurrences", fontsize=14, title_fontsize=16)
#     # Set legend labels
#     if dfTracer is not None:
#         legend.get_texts()[0].set_text(f"unknowns ({white_flag_count} of {white_count} above line)")
#         try:
#             legend.get_texts()[1].set_text(
#                 f"tracers ({red_flag_count} of {red_count} above line)"
#             )  # If tracers are present, add secondary legend label
#         except IndexError:  # If no tracers found in blanks, set alternate legend
#             # legend.set_text("tracers 0 of 0 above line)")
#             pass
#     # Make it pretty
#     frame = legend.get_frame()  # sets up for color, edge, and transparency
#     frame.set_facecolor("lightgray")  # color of legend
#     frame.set_edgecolor("black")  # edge color of legend
#     frame.set_alpha(1)  # deals with transparency
#     # Adjust axes labels
#     axes[0].set_title(titleText + ": Blanks", fontsize=18, weight="bold")
#     axes[0].set_xlabel("Mean Abundance", fontsize=14)
#     axes[0].set_ylabel("CV", fontsize=14)
#     axes[0].set_ylim(0, 2.5)
#     axes[0].set_xlim(min_abundance_limit, max_abundance_limit)  # Set x-axis to scale based on the min/max data points
#     axes[0].set(xscale="log")
#     axes[0].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
#     axes[0].tick_params(axis="both", which="both", labelsize=12)

#     # Sample plot
#     b = sns.scatterplot(
#         data=plot2.loc[((plot2["type"] != "blank")), :].sort_values("spike"),
#         x="Mean",
#         y="CV",
#         hue="spike",
#         edgecolor="black",
#         alpha=0.5,
#         ax=axes[1],
#     )
#     # Add CV red dashed line
#     b.axhline(
#         y=max_replicate_cv_value,
#         color="red",
#         linestyle="dashed",
#         linewidth=1.5,
#         alpha=1,
#     )
#     b.text(
#         max_abundance_limit / 5,
#         max_replicate_cv_value + 0.1,
#         "CV = {}".format(max_replicate_cv_value),
#         ha="center",
#         va="center_baseline",
#         weight="bold",
#         size=14,
#     )
#     # Perform occurrence counts above and below CV by sample type
#     red_count = len(plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 1)), "CV"])
#     red_flag_count = sum(plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 1)), "CV"] > max_replicate_cv_value)
#     white_count = len(plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 0)), "CV"])
#     white_flag_count = sum(
#         plot2.loc[((plot2["type"] != "blank") & (plot2["spike"] == 0)), "CV"] > max_replicate_cv_value
#     )
#     # Only generate legend if tracers are submitted -- THIS ISN'T TRUE RIGHT NOW
#     legend = b.legend(title="Unfiltered Occurrences", fontsize=14, title_fontsize=16)
#     # Set legend labels
#     if dfTracer is not None:
#         legend.get_texts()[0].set_text(f"unknowns ({white_flag_count} of {white_count} above line)")
#         legend.get_texts()[1].set_text(
#             f"tracers ({red_flag_count} of {red_count} above line)"
#         )  # If tracers are present, add secondary legend label
#     # Make it pretty
#     frame = legend.get_frame()  # sets up for color, edge, and transparency
#     frame.set_facecolor("lightgray")  # color of legend
#     frame.set_edgecolor("black")  # edge color of legend
#     frame.set_alpha(1)  # deals with transparency
#     # Adjust axes labels
#     axes[1].set_title(titleText + ": Non-blanks", fontsize=18, weight="bold")
#     axes[1].set_xlabel("Mean Abundance", fontsize=14)
#     axes[1].set_ylabel("CV", fontsize=14)
#     axes[1].set_ylim(0, 2.5)
#     axes[1].set_xlim(min_abundance_limit, max_abundance_limit)
#     axes[1].set(xscale="log")
#     axes[1].set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
#     axes[1].tick_params(axis="both", which="both", labelsize=12)
#     # Convert the plot to a bytes-like object
#     buffer = io.BytesIO()
#     plt.savefig(buffer)
#     buffer.seek(0)
#     # Store in class variable
#     self.cv_scatterplots_out.append(buffer.getvalue())
#     # Map to outputs
#     self.cv_scatterplot_map["cv_scatterplot"] = self.cv_scatterplots_out[0]
#     project_name = self.parameters["project_name"][1]
#     self.gridfs.put(
#         "&&".join(self.cv_scatterplot_map.keys()),
#         _id=self.jobid + "_cv_scatterplots",
#         encoding="utf-8",
#         project_name=project_name,
#     )
#     self.mongo_save(self.cv_scatterplot_map["cv_scatterplot"], step="cv_scatterplot")
#     # reset plt
#     plt.clf()
