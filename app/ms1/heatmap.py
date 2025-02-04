"""
Class for producing the heatmap for the WebApp
2024/05/07
TMF
"""

from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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


# def occurrence_heatmap(self, input_dfs):
#     plt.rcdefaults()
#     # Get user input CV and Replicate thresholds
#     max_replicate_cv_value = self.parameters["max_replicate_cv"][1]
#     min_replicate_hits_percent = self.parameters["min_replicate_hits"][1]
#     min_replicate_blanks_hit_percent = self.parameters["min_replicate_hits_blanks"][1]
#     MRL_mult = float(self.parameters["mrl_std_multiplier"][1])
#     # convert max_replicate_cv_value to a numeric value
#     max_replicate_cv_value = pd.to_numeric(self.parameters["max_replicate_cv"][1], errors="coerce")
#     # convert min_replicate_hits_percent to a numeric value
#     min_replicate_hits_percent = pd.to_numeric(self.parameters["min_replicate_hits"][1], errors="coerce")
#     # convert min_replicate_blanks_hits_percent to a numeric value
#     min_replicate_blanks_hits_percent = pd.to_numeric(
#         self.parameters["min_replicate_hits_blanks"][1], errors="coerce"
#     )
#     # get dataframe 'All Detection Statistics (Pos)' if it exists else None
#     dfPos = (
#         self.data_map["All Detection Statistics (Pos)"]
#         if "All Detection Statistics (Pos)" in self.data_map
#         else None
#     )
#     # get dataframe 'All Detection Statistics (Neg)' if it exists else None
#     dfNeg = (
#         self.data_map["All Detection Statistics (Neg)"]
#         if "All Detection Statistics (Neg)" in self.data_map
#         else None
#     )
#     # combine the two dataframes. Ignnore non-existing dataframes
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
#         sublist[0][:-1]
#         for sublist in all_headers
#         if len(sublist) > 1
#         if not any(x in sublist[0] for x in non_samples)
#     ]
#     # Isolate sample_groups from stats columns
#     prefixes = ["Mean ", "Median ", "CV ", "STD ", "Detection Count ", "Detection Percentage "]
#     sample_groups = [item for item in sam_headers if not any(x in item for x in prefixes)]
#     logger.info("sample_groups= {}".format(sample_groups))
#     # Blank_MDL - need to check what the blank samples are actually named
#     blank_strings = ["MB", "Mb", "mb", "BLANK", "Blank", "blank", "BLK", "Blk"]
#     blank_col = [item for item in sample_groups if any(x in item for x in blank_strings)]
#     logger.info("blank_col= {}".format(blank_col))
#     blank_mean = "Mean " + blank_col[0]
#     blank_std = "STD " + blank_col[0]
#     # AC Add blank replicate percentage column grab NTAW574
#     blank_rper = "Detection Percentage " + blank_col[0]

#     # Calculate MDL
#     # AC 6/18/2024: Need to pull in MRL multiplier for MRL calculation
#     dfCombined["MDL"] = dfCombined[blank_mean] + MRL_mult * dfCombined[blank_std]
#     dfCombined["MDL"] = dfCombined["MDL"].fillna(dfCombined[blank_mean])
#     dfCombined["MDL"] = dfCombined["MDL"].fillna(0)
#     # AC Where blank replicate percentage column fails, zero out MDL - NTAW574
#     dfCombined.loc[dfCombined[blank_rper] < min_replicate_blanks_hits_percent, "MDL"] = 0
#     # Find CV, Rep_Percent, and Mean cols from df
#     cv_cols = ["CV " + col for col in sample_groups]
#     rper_cols = ["Detection Percentage " + col for col in sample_groups]
#     mean_cols = ["Mean " + col for col in sample_groups]
#     # Subset CV cols from df
#     cv_df = dfCombined[cv_cols]

#     # Get number of occurrences from the CV dataframe
#     titleText = (
#         "Heatmap of Feature Occurrences (n = "
#         + str(cv_df.size)
#         + ")\nSample Rep. Threshold = {}%; Blank Rep. Threshold = {}%; CV Threshold = {}; MRL Multiplier = {}".format(
#             min_replicate_hits_percent, min_replicate_blanks_hits_percent, max_replicate_cv_value, MRL_mult
#         )
#     )

#     # Blank out cvs in samples with <2 samples
#     for x, y, z in zip(cv_cols, rper_cols, mean_cols):
#         # Replace cv_df values with nan in cv_col for n_abun and MDL cutoffs
#         # Check if replicate column is the blank column to determine which filter to apply - NTAW574
#         if y == blank_rper:
#             cv_df.loc[dfCombined[y] < min_replicate_blanks_hits_percent, x] = np.nan
#         else:
#             cv_df.loc[dfCombined[y] < min_replicate_hits_percent, x] = np.nan
#         cv_df.loc[dfCombined[y] < min_replicate_hits_percent, x] = np.nan
#         cv_df.loc[dfCombined[z] <= dfCombined["MDL"], x] = np.nan
#     # Add sum of Trues for condition applied to cv dataframe
#     cv_df["below count"] = (cv_df <= max_replicate_cv_value).sum(axis=1)
#     # Sort values by how many detects are present
#     cv_df = cv_df.sort_values("below count")
#     # Remove below count column
#     del cv_df[cv_df.columns[-1]]
#     # Create masks for CV cutoffs
#     above = cv_df > max_replicate_cv_value
#     below = cv_df <= max_replicate_cv_value
#     nan_ = cv_df.isna()
#     # Use masks to changes values in cv_df to 1, 0, -1
#     dum = np.where(above, 1, cv_df)
#     dum = np.where(below, 0, dum)
#     dum = np.where(nan_, -1, dum)
#     # Create matrix from discretized dataframe
#     cv_df_discrete = pd.DataFrame(dum, index=cv_df.index, columns=[col[3:] for col in cv_df.columns])
#     cv_df_trans = cv_df_discrete.transpose()
#     # Set Figure size and title
#     plt.figure(figsize=(40, 15))
#     plt.title(titleText, fontsize=40, pad=30, linespacing=1.5)
#     # Create custom color mapping
#     myColors = ((0.8, 0.8, 0.8, 1.0), (1.0, 1.0, 1.0, 1.0), (1, 0.0, 0.2, 1.0))
#     cmap = LinearSegmentedColormap.from_list("Custom", myColors, len(myColors))
#     # Plot heatmap
#     ax = sns.heatmap(cv_df_trans, cmap=cmap, cbar_kws={"shrink": 0.2, "pad": 0.01})
#     ax.set_ylabel("Sample Set", fontsize=28)
#     ax.set_xlabel("Feature ID (n = " + str(len(cv_df)) + ")", fontsize=28)
#     ax.set(xticklabels=[])
#     ax.tick_params(axis="y", which="both", labelsize=24, labelrotation=0)
#     # Add outside border
#     ax.patch.set_edgecolor("black")
#     ax.patch.set_linewidth(2)
#     # Manually specify colorbar labelling after it's been generated
#     colorbar = ax.collections[0].colorbar
#     colorbar.ax.tick_params(labelsize=32)
#     colorbar.set_ticks([-0.667, 0, 0.667])
#     colorbar.set_ticklabels(
#         [
#             "no occurrence ({})".format(nan_.sum().sum()),
#             "CV <= {} ({})".format(max_replicate_cv_value, below.sum().sum()),
#             "CV > {} ({})".format(max_replicate_cv_value, above.sum().sum()),
#         ]
#     )
#     # Convert the plot to a bytes-like object
#     buffer = io.BytesIO()
#     plt.savefig(buffer)
#     buffer.seek(0)
#     # Store in class variable
#     self.occurrence_heatmaps_out.append(buffer.getvalue())
#     # Map to outputs
#     self.occurrence_heatmap_map["occurrence_heatmap"] = self.occurrence_heatmaps_out[0]
#     project_name = self.parameters["project_name"][1]
#     self.gridfs.put(
#         "&&".join(self.occurrence_heatmap_map.keys()),
#         _id=self.jobid + "_occurrence_heatmaps",
#         encoding="utf-8",
#         project_name=project_name,
#     )
#     self.mongo_save(self.occurrence_heatmap_map["occurrence_heatmap"], step="occurrence_heatmap")
#     # reset plt
#     plt.clf()
