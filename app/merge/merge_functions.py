###########################################################################
# Written for Placenta data
# This code reads in three separate sources of results:
# 1) WebApp MS1 results (df_ms1)
# 2) Reference library serach results (df_pcdl)
# 3) CFM-ID MS2 results (df_cfmid)
#
# Once the sets are read in, it converts all three dataframes into lists
# For faster matching. It matches the MS1 results against MS2 CFMID and MS2 PCDL
#
# The resulting match information is concatenated (i.e. multiple MS2 files matched to a MS1 feature/chemicals)
# And then merged back onto the initial dataframe of results from the WebApp
# i.e. the final result is the same exact file from MS1 WebApp, with extra columns appended on for
# the match information.
###########################################################################aa

import pandas as pd
import glob
import numpy as np
from functools import reduce
import logging

logger = logging.getLogger("nta_app.merge")


def process_MS2_data(ms1_data, ms2_data_list, mass_accuracy=10, rt_accuracy=0.2):
    # NTAW-158: Adjust sheet names pulled from MS1 results
    matched_df = ms1_data if isinstance(ms1_data, pd.DataFrame) else ms1_data["chemical_results"]
    # matched_df = ms1_data if isinstance(ms1_data, pd.DataFrame) else ms1_data["dsstox_search"]

    matched_df.rename(columns={"DTXCID_INDIVIDUAL_COMPONENT": "DTXCID"}, inplace=True)

    for ms2_data in ms2_data_list:
        filename = ms2_data["file_name"]
        cfmid_df = ms2_data["file_df"]
        # mass_col, rt_col, score_col, q_score_col, percentile_col = (f"MASS_MGF_{filename}", f"RT_{filename}", f"SUM_SCORE_{filename}", f"QUOTIENT_SCORE_{filename}", f"PERCENTILE_SCORE_{filename}")
        mass_col, rt_col, score_col, q_score_col, percentile_col = (
            f"MASS_MGF_{filename}",
            f"RT_{filename}",
            f"SUM_SCORE_{filename}",
            f"QUOTIENT_SCORE_{filename}",
            f"PERCENTILE_SCORE_{filename}",
        )
        # logger.info('mass_col, rt_col, score_col')
        # logger.info(mass_col, rt_col, score_col)

        # NTAW-158: Adjust columns renamed based on new MS2 column names
        # NTAW-158: Grab the neutral mass column from the MS2 data as this is going to be compared to the neutral mass from the MS1 data
        cfmid_df.rename(
            columns={
                "MASS_NEUTRAL": mass_col,
                "RT": rt_col,
                "SUM_SCORE": score_col,
                "Q-SCORE": q_score_col,
                "PERCENTILE": percentile_col,
            },
            inplace=True,
        )
        # cfmid_df.rename(columns={"MASS_MGF": mass_col, "RT": rt_col, "SUM_SCORE": score_col}, inplace=True)
        # cfmid_df.rename(columns = {'MASS_MGF': mass_col, 'RT': rt_col, 'SUM_SCORE' : score_col}, inplace = True)
        # cfmid_df.rename(columns={"MASS_in_MGF": mass_col, "RT": rt_col, "energy_sum": score_col}, inplace=True)

        matched_df = matched_df.merge(
            cfmid_df[
                [
                    "DTXCID",
                    f"MASS_MGF_{filename}",
                    f"RT_{filename}",
                    f"SUM_SCORE_{filename}",
                    f"QUOTIENT_SCORE_{filename}",
                    f"PERCENTILE_SCORE_{filename}",
                ]
            ],
            how="left",
            on="DTXCID",
        )
        matched_df["mass_diff"] = abs(matched_df["Mass"] - matched_df[f"MASS_MGF_{filename}"])
        # NTAW-158: Retention time units of input MS1 are in minutes, input MS2 are in seconds, convert MS2 units to minutes by dividing by 60
        # matched_df["rt_diff"] = abs(matched_df["Retention_Time"] - matched_df[f"RT_{filename}"])
        matched_df["rt_diff"] = abs(matched_df["Retention_Time"] - matched_df[f"RT_{filename}"] / 60)
        matched_df["sum_diff"] = [
            mass_diff + rt_diff if mass_diff <= mass_accuracy and rt_diff <= rt_accuracy else np.nan
            for mass_diff, rt_diff in zip(matched_df["mass_diff"], matched_df["rt_diff"])
        ]
        matched_df[[mass_col, rt_col, score_col, q_score_col, percentile_col]] = matched_df[
            [mass_col, rt_col, score_col, q_score_col, percentile_col]
        ].where(
            (matched_df["mass_diff"] < mass_accuracy) & (matched_df["rt_diff"] < rt_accuracy),
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        )

    matched_df.drop(columns=["mass_diff", "rt_diff", "sum_diff"], inplace=True)
    matched_df["Median_MS2_Mass"] = matched_df[[col for col in matched_df.columns if "MASS_" in col]].apply(
        np.median, axis=1
    )
    matched_df["Median_MS2_RT"] = matched_df[[col for col in matched_df.columns if "RT_" in col]].apply(
        np.median, axis=1
    )
    matched_df["Median_Score"] = matched_df[[col for col in matched_df.columns if "SUM_SCORE_" in col]].apply(
        np.median, axis=1
    )
    return matched_df
