import pandas as pd
import numpy as np
from operator import itemgetter
from difflib import SequenceMatcher
from itertools import groupby
import os
import re
import logging


logger = logging.getLogger("nta_app.ms1")


"""UTILITY FUNCTIONS (many from Functions_Universal_v3)"""


def assign_feature_id(df_in, start=1):
    """
    A function to assign unique feature ids to a nta dataset
    :param df_in: the dataframe to assign ids to
    :param start: assign ids starting at this integer
    :return: returns the new df with unique feature ids added
    """
    df = df_in.copy()
    row_nums = list(range(0, len(df.index)))
    to_assign = [x + start for x in row_nums]
    df.insert(0, "Feature ID", to_assign.copy())
    return df


def differences(s1, s2):
    """
    find the number of different characters between two strings (headers)
    """
    s1 = re.sub(re.compile(r"\([^)]*\)"), "", s1)
    s2 = re.sub(re.compile(r"\([^)]*\)"), "", s2)
    count = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
    return count


def formulas(df):
    """
    Return list of formulas tagged 'For_Dashboard_Search'
    """
    df.drop_duplicates(subset="Formula", keep="first", inplace=True)
    formulas = df.loc[df["For_Dashboard_Search"] == "1", "Formula"].values
    formulas_list = [str(i) for i in formulas]
    return formulas_list


def masses(df):
    """
    Return list of masses tagged 'For_Dashboard_Search'
    """
    masses = df.loc[df["For_Dashboard_Search"] == "1", "Mass"].values
    logger.info("# of masses for dashboard search: {} out of {}".format(len(masses), len(df)))
    masses_list = [str(i) for i in masses]
    return masses_list


def parse_headers(df_in):
    """
    A function to group the dataframe's column headers into sets of similar names which represent replicates
    :param df_in: the dataframe of features
    :return: a list of groups of column labels
    """
    df = df_in.copy()
    headers = df.columns.values.tolist()
    countS = 0
    countD = 0
    new_headers = []
    for s in range(0, len(headers) - 1):
        if "blank" or "Blank" or "MB" in headers[s]:
            if differences(str(headers[s]), str(headers[s + 1])) < 2:  # 3 is more common
                countS += 1
            if differences(str(headers[s]), str(headers[s + 1])) >= 2:
                countD += 1
                countS = countS + 1
        else:
            if differences(str(headers[s]), str(headers[s + 1])) < 2:  # 2 is more common
                countS += 1
            if differences(str(headers[s]), str(headers[s + 1])) >= 2:
                countD += 1
                countS = countS + 1
            # print "These are different "
        if "_Flags" in headers[s]:
            break
        new_headers.append([headers[countS], countD])
        new_headers.sort(key=itemgetter(1))
    groups = groupby(new_headers, itemgetter(1))
    new_headers_list = [[item[0] for item in data] for (key, data) in groups]
    return new_headers_list


def flags(df):
    """
    A function to develop optional flags
    """
    SCORE = 90  # formula match is 90
    df["Neg_Mass_Defect"] = np.where((df.Mass - df.Mass.round(0)) < 0, "1", "0")
    df["Halogen"] = np.where(df.Compound.str.contains("F|l|r|I"), "1", "0")
    df["Formula_Match"] = np.where(df.Score != df.Score, "0", "1")  # check if it does not have a score
    df["Formula_Match_Above90"] = np.where(df.Score >= SCORE, "1", "0")
    df["X_NegMassDef_Below90"] = np.where(
        ((df.Score < SCORE) & (df.Neg_Mass_Defect == "1") & (df.Halogen == "1")),
        "1",
        "0",
    )
    # df['For_Dashboard_Search'] = np.where(((df.Formula_Match_Above90 == '1') | (df.X_NegMassDef_Below90 == '1')) , '1', '0')
    df["For_Dashboard_Search"] = np.where(
        ((df.Formula_Match_Above90 == "1") | (df.X_NegMassDef_Below90 == "1")), "1", "1"
    )  # REMOVE THIS LINE AMRL UNCOMMENT ABOVE
    df.sort_values(
        [
            "Formula_Match",
            "For_Dashboard_Search",
            "Formula_Match_Above90",
            "X_NegMassDef_Below90",
        ],
        ascending=[False, False, False, False],
        inplace=True,
    )
    # df.to_csv('input-afterflag.csv', index=False)
    # print df1
    df.sort_values("Compound", ascending=True, inplace=True)
    return df


"""PASS-THROUGH COLUMNS FUNCTION"""


def passthrucol(df_in):
    """
    Find all columns in dfs that aren't necessary (i.e., not Mass and RT) and store
    these columns to be later appended to the output -- TMF 11/20/23
    """
    # Make a copy of the input df
    df = df_in.copy()
    # Parse headers
    all_headers = parse_headers(df)
    # Define active_cols: Keep 'Feature ID' in pt_headers to merge later
    active_cols = ["Retention_Time", "Mass", "Ionization_Mode"]
    # Create list of pass through headers that are not in the active columns
    pt_headers = ["Feature ID"] + [
        item
        for sublist in all_headers
        for item in sublist
        if len(sublist) == 1 and not any(x in sublist for x in active_cols)
    ]
    headers = ["Feature ID"] + [
        item for sublist in all_headers for item in sublist if not any(x in item for x in pt_headers)
    ]
    # Save pass through columns in df
    df_pt = df[pt_headers]
    df_trim = df[headers]
    # Return passthrough columns data and dataframe minus passthrough columns
    return df_pt, df_trim


"""ADDUCT IDENTIFICATION FUNCTIONS"""


def adduct_matrix(df, a_name, delta, Mass_Difference, Retention_Difference, ppm):
    """
    Modified version of Jeff's 'adduct_identifier' function. This function executes
    the matrix portion of the old function -- TMF 10/27/23
    """
    # 'Mass' to matrix, 'Retention Time' to matrix, 'Feature ID' to matrix
    mass = df["Mass"].to_numpy()
    rts = df["Retention_Time"].to_numpy()
    ids = df["Feature ID"].to_numpy()
    # Reshape 'masses', 'rts', and 'ids'
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts), 1))
    ids_matrix = np.reshape(ids, (1, len(ids)))
    # Create difference matrices
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose()
    # Create array of 0s
    unique_adduct_number = np.zeros(len(df.index))
    # Add 'diff_mass_matrix' by 'delta' (adduct mass)
    is_adduct_diff = abs(diff_matrix_mass - delta)
    has_adduct_diff = abs(diff_matrix_mass + delta)
    # Adjust matrix if units are 'ppm'
    if ppm:
        has_adduct_diff = (has_adduct_diff / masses_matrix) * 10**6
        is_adduct_diff = (is_adduct_diff / masses_matrix) * 10**6
    # Replace cells in 'has_adduct_diff' below 'Mass_Difference' and 'Retention_Difference' with 1, else 0
    is_adduct_matrix = np.where(
        (is_adduct_diff < Mass_Difference) & (abs(diff_matrix_rt) < Retention_Difference),
        1,
        0,
    )
    has_adduct_matrix = np.where(
        (has_adduct_diff < Mass_Difference) & (abs(diff_matrix_rt) < Retention_Difference),
        1,
        0,
    )
    # Remove self matches
    np.fill_diagonal(is_adduct_matrix, 0)
    np.fill_diagonal(has_adduct_matrix, 0)
    # check if all values in is_adduct_matrix are 0
    if np.all(is_adduct_matrix == 0):
        # skip matrix math if no adduct matches
        pass
    else:
        # Define 'row_num', 'is_id_matrix'
        row_num = len(mass)
        is_id_matrix = np.tile(ids_matrix, (row_num, 1))
        # Matrix multiplication, keep highest # row if multiple adducts
        is_adduct_number = is_adduct_matrix * is_id_matrix
        # if is adduct of multiple, keep highest # row
        is_adduct_number_flat = np.max(is_adduct_number, axis=1)
        # Matrix multiplication, keep highest # row if multiple adducts
        has_adduct_number = has_adduct_matrix * is_id_matrix
        # if is adduct of multiple, keep highest # row
        has_adduct_number_flat = np.max(has_adduct_number, axis=1)  # these will all be the same down columns
        unique_adduct_number = np.where(
            has_adduct_number_flat != 0, has_adduct_number_flat, is_adduct_number_flat
        ).astype(int)
        # Edit 'df['Has Adduct or Loss?']' column
        df["Has Adduct or Loss?"] = np.where(
            (has_adduct_number_flat > 0) & (df["Is Adduct or Loss?"] == 0),
            df["Has Adduct or Loss?"] + 1,
            df["Has Adduct or Loss?"],
        )
        # Edit 'df['Is Adduct or Loss?']' column
        df["Is Adduct or Loss?"] = np.where(
            (is_adduct_number_flat > 0) & (df["Has Adduct or Loss?"] == 0),
            1,
            df["Is Adduct or Loss?"],
        )
        # Edit 'df['Adduct or Loss Info']' column
        df["Adduct or Loss Info"] = np.where(
            (has_adduct_number_flat > 0) & (df["Is Adduct or Loss?"] == 0),
            df["Adduct or Loss Info"] + unique_adduct_number.astype(str) + "({});".format(a_name),
            df["Adduct or Loss Info"],
        )
        # Edit 'df['Adduct or Loss Info']' column
        df["Adduct or Loss Info"] = np.where(
            (is_adduct_number_flat > 0) & (df["Has Adduct or Loss?"] == 0),
            df["Adduct or Loss Info"] + unique_adduct_number.astype(str) + "({});".format(a_name),
            df["Adduct or Loss Info"],
        )
    # Return dataframe with three new adduct info columns
    return df


def window_size(df_in, mass_diff_mass=112.985586):
    """
    # Estimate a sliding window size from the input data by finding
    the maximum distance between indices differing by 'mass_diff_mass' -- TMF 10/27/23
    """
    df = df_in.copy()
    masses = df["Mass"].tolist()
    li = []
    for i in range(len(masses) - 1):
        val = masses[i] + mass_diff_mass
        if df["Mass"].max() > val:
            ind = df.index[df["Mass"] > val].tolist()[0]
            li.append(ind - i)
    window_size = pd.Series(li)
    val = window_size.max()
    # Return singular int value that ensures we won't miss any possible adducts
    return val


def chunk_adducts(df_in, n, step, a_name, delta, Mass_Difference, Retention_Difference, ppm):
    """
    Function that takes the input data, chunks it based on window size, then loops through chunks
    and sends them to 'adduct_matrix' for calculation -- TMF 10/27/23
    """
    # Create copy
    df = df_in.copy()
    # Create chunks of df based on how much Web App can handle (n) and step size that captures all adducts (step)
    to_test_list = [df[i : i + n] for i in range(0, df.shape[0], step)]
    to_test_list = [i for i in to_test_list if (i.shape[0] > n / 2)]
    # Create list, iterate through df chunks and append results to list
    li = []
    for x in to_test_list:
        dum = adduct_matrix(x, a_name, delta, Mass_Difference, Retention_Difference, ppm)
        li.append(dum)
    # Concatenate results together, removing overlapping sections
    output = pd.concat(li, axis=0).drop_duplicates(subset=["Mass", "Retention_Time"], keep="last")
    # Return dataframe with three adduct info columns added
    return output


def adduct_identifier(df_in, adduct_selections, Mass_Difference, Retention_Difference, ppm, ionization):
    """
    Function that does the front-end of the old 'adduct_identifier'; we trim the input data by identifying
    features that are near to adduct distance from another feature. This shortened dataframe is used to
    calculate a window size, then loop through possible adducts, passing to 'chunk_adducts' -- TMF 10/27/23
    """
    # Copy df_in, only need 'Feature ID', 'Mass', and 'Retention Time'
    df = df_in[["Feature ID", "Mass", "Retention_Time"]].copy()
    # Round columns
    df["Rounded Mass"] = df["Mass"].round(2)
    df["Rounded RT"] = df["Retention_Time"].round(1)
    # Create tuple of 'Rounded RT' and 'Rounded Mass'
    df["Rounded_RT_Mass_Pair"] = list(zip(df["Rounded RT"], df["Rounded Mass"]))

    # Define pos/neg/neutral adduct lists
    # Proton subtracted - we observe Mass+(H+) and Mass+(Adduct)
    pos_adduct_li = [
        ("Na", 21.981942),
        ("K", 37.955882),
        ("NH4", 17.026547),
    ]
    # Proton added - we observe Mass-(H+) and Mass+(Adduct)
    neg_adduct_li = [
        ("Cl", 35.976678),
        ("Br", 79.926161),
        ("HCO2", 46.005477),
        ("CH3CO2", 60.021127),
        ("CF3CO2", 113.992862),
        ("FA", 46.00550),
    ]
    # no change to neutral losses
    neutral_losses_li = [
        ("H2O", 18.010565),
        ("2H2O", 36.02113),
        ("3H2O", 54.031695),
        ("4H2O", 72.04226),
        ("5H2O", 90.052825),
        ("NH3", 17.0265),
        ("O", 15.99490),
        ("CO", 29.00220),
        ("CO2", 43.989829),
        ("C2H4", 28.03130),
        ("HFA", 46.00550),
        ("HAc", 60.02110),
        ("MeOH", 32.02620),
        ("ACN", 41.02650),
        ("IsoProp", 60.05810),
    ]
    # Determine possible adduct dictionary according to ionization
    if ionization == "positive":
        possible_adduct_deltas = [item for item in pos_adduct_li if item[0] in adduct_selections[0]]
        possible_adduct_deltas = possible_adduct_deltas + [
            item for item in neutral_losses_li if item[0] in adduct_selections[2]
        ]
        possible_adduct_deltas = dict(possible_adduct_deltas)
    else:
        possible_adduct_deltas = [item for item in neg_adduct_li if item[0] in adduct_selections[1]]
        possible_adduct_deltas = possible_adduct_deltas + [
            item for item in neutral_losses_li if item[0] in adduct_selections[2]
        ]
        possible_adduct_deltas = dict(possible_adduct_deltas)
    # Create empty list to hold mass shift/RT tuples
    list_of_mass_shifts_RT_pairs = []
    # Logic gate for no adducts selected
    if len(possible_adduct_deltas) > 0:
        # Loop through possible adducts, add/subtract adduct mass from each feature, append
        # 'Rounded RT', 'Rounded Mass' tuples to 'list_of_mass_shifts_RT_pairs' for both addition
        # and subtraction.
        for k, v in possible_adduct_deltas.items():
            col1 = "Mass - " + k
            col2 = "Mass + " + k
            df[col1] = (df["Mass"] - v).round(2)
            df[col2] = (df["Mass"] + v).round(2)
            list_of_mass_shifts_RT_pairs.append(list(zip(df["Rounded RT"], df[col1])))
            list_of_mass_shifts_RT_pairs.append(list(zip(df["Rounded RT"], df[col2])))
        # Extend list
        list_of_mass_shifts_RT_pairs = [item for sublist in list_of_mass_shifts_RT_pairs for item in sublist]
        # Remove duplicate tuples (sets don't carry duplicates)
        list_of_mass_shifts_RT_pairs = list(set(list_of_mass_shifts_RT_pairs))
        # Filter df for features to check for adducts
        to_test = df[df["Rounded_RT_Mass_Pair"].isin(list_of_mass_shifts_RT_pairs)]
        to_test = to_test.sort_values("Mass", ignore_index=True)
        # Add columns to be changed by 'adduct_matrix'
        to_test["Has Adduct or Loss?"] = 0
        to_test["Is Adduct or Loss?"] = 0
        to_test["Adduct or Loss Info"] = ""
        # Set 'n' to tested memory capacity of WebApp for number of features in 'adduct_matrix'
        n = 12000
        # If 'to_test' is less than n, send it straight to 'adduct_matrix'
        if to_test.shape[0] <= n:
            for a_name, delta in possible_adduct_deltas.items():
                to_test = adduct_matrix(to_test, a_name, delta, Mass_Difference, Retention_Difference, ppm)
        # Else, calculate the moving window size and send 'to_test' to 'chunk_adducts'
        else:
            step = n - window_size(to_test)
            # Loop through possible adducts, perform 'adduct_matrix'
            for a_name, delta in possible_adduct_deltas.items():
                to_test = chunk_adducts(to_test, n, step, a_name, delta, Mass_Difference, Retention_Difference, ppm)
        # Concatenate 'Has Adduct or Loss?', 'Is Adduct or Loss?', 'Adduct or Loss Info' to df
        df_in = pd.merge(
            df_in,
            to_test[
                [
                    "Mass",
                    "Retention_Time",
                    "Has Adduct or Loss?",
                    "Is Adduct or Loss?",
                    "Adduct or Loss Info",
                ]
            ],
            how="left",
            on=["Mass", "Retention_Time"],
        )
    # Use fillna() to fill in 0s
    df_in["Is Adduct or Loss?"].fillna(0, inplace=True)
    df_in["Has Adduct or Loss?"].fillna(0, inplace=True)
    # Return dataframe with three adduct info columns added
    return df_in


"""DUPLICATE REMOVAL FUNCTIONS"""


def chunk_dup_remove(df_in, n, step, mass_cutoff, rt_cutoff, ppm):
    """
    Wrapper function for passing manageable-sized dataframe chunks to 'dup_matrix' -- TMF 10/27/23
    """
    # Create copy of df_in
    df = df_in.copy()
    # Chunk df based on n (# of features WebApp can handle) and step
    to_test_list = [df[i : i + n] for i in range(0, df.shape[0], step)]
    # Remove small remainder tail, if present
    to_test_list = [i for i in to_test_list if (i.shape[0] > n / 2)]
    # Create list and duplicate list
    li = []
    dupe_li = []
    # Pass list to 'dup_matrix'
    for x in to_test_list:
        # dum = dup_matrix(x, mass_cutoff, rt_cutoff) # Deprecated 10/30/23 -- TMF
        dum, dupes = dup_matrix_remove(x, mass_cutoff, rt_cutoff, ppm)
        li.append(dum)
        dupe_li.append(dupes)
    # Concatenate results, drop duplicates from overlap
    output = pd.concat(li, axis=0).drop_duplicates(subset=["Mass", "Retention_Time"], keep="first")
    dupe_df = pd.concat(dupe_li, axis=0).drop_duplicates(subset=["Mass", "Retention_Time"], keep="first")
    # Return de-duplicated dataframe (output) and removed duplicates (dupe_df)
    return output, dupe_df


def dup_matrix_remove(df_in, mass_cutoff, rt_cutoff, ppm):
    """
    Matrix portion of 'duplicates' function; takes a filtered 'to_test' df, does matrix math,
    returns 'passed' items (i.e., de-duplicated dataframe) -- TMF 10/27/23
    """
    # Create matrices from df_in
    mass = df_in["Mass"].to_numpy()
    rts = df_in["Retention_Time"].to_numpy()
    # Reshape matrices
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts), 1))
    # Perform matrix transposition
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose()
    # Find indices where differences are less than 'mass_cutoff' and 'rt_cutoff'
    if ppm:
        duplicates_matrix = np.where(
            (abs(diff_matrix_mass / masses_matrix) * 10**6 <= mass_cutoff) & (abs(diff_matrix_rt) <= rt_cutoff),
            1,
            0,
        )
    else:
        duplicates_matrix = np.where(
            (abs(diff_matrix_mass) <= mass_cutoff) & (abs(diff_matrix_rt) <= rt_cutoff),
            1,
            0,
        )
    np.fill_diagonal(duplicates_matrix, 0)
    # Find # of duplicates for each row
    row_sums = np.sum(duplicates_matrix, axis=1)
    # Calculate lower triangle of matrix
    duplicates_matrix_lower = np.tril(duplicates_matrix)
    lower_row_sums = np.sum(duplicates_matrix_lower, axis=1)
    # Store features with no duplicates in 'passed'
    passed = df_in[(row_sums == 0) | (lower_row_sums == 0)].copy()
    # Store features with any dupi=licates in 'dupes'
    dupes = df_in.loc[df_in[(row_sums != 0) & (lower_row_sums != 0)].index, :]
    # Return de-duplicated dataframe (passed) and duplicates (dupes)
    return passed, dupes


def chunk_dup_flag(df_in, n, step, mass_cutoff, rt_cutoff, ppm):
    """
    Wrapper function for passing manageable-sized dataframe chunks to 'dup_matrix_flag' -- TMF 04/11/24
    """
    # Create copy of df_in
    df = df_in.copy()
    # Chunk df based on n (# of features WebApp can handle) and step
    to_test_list = [df[i : i + n] for i in range(0, df.shape[0], step)]
    # Remove small remainder tail, if present
    to_test_list = [i for i in to_test_list if (i.shape[0] > n / 2)]
    # Create list
    li = []
    # Pass list to 'dup_matrix'
    for x in to_test_list:
        # dum = dup_matrix(x, mass_cutoff, rt_cutoff) # Deprecated 10/30/23 -- TMF
        dum = dup_matrix_flag(x, mass_cutoff, rt_cutoff, ppm)
        li.append(dum)
    # Concatenate results, drop duplicates from overlap
    output = pd.concat(li, axis=0).drop_duplicates(subset=["Mass", "Retention_Time"], keep="first")
    # Return dataframe with flagged duplicates (output)
    return output


def dup_matrix_flag(df_in, mass_cutoff, rt_cutoff, ppm):
    """
    Matrix portion of 'duplicates' function; takes a filtered 'to_test' df, does matrix math,
    returns 'df' with duplicates flagged in 'Duplicate Feature?' column -- TMF 04/11/24
    """
    # Create matrices from df_in
    mass = df_in["Mass"].to_numpy()
    rts = df_in["Retention_Time"].to_numpy()
    # Reshape matrices
    masses_matrix = np.reshape(mass, (len(mass), 1))
    rts_matrix = np.reshape(rts, (len(rts), 1))
    # Perform matrix transposition
    diff_matrix_mass = masses_matrix - masses_matrix.transpose()
    diff_matrix_rt = rts_matrix - rts_matrix.transpose()
    # Find indices where differences are less than 'mass_cutoff' and 'rt_cutoff'
    if ppm:
        duplicates_matrix = np.where(
            (abs(diff_matrix_mass / masses_matrix) * 10**6 <= mass_cutoff) & (abs(diff_matrix_rt) <= rt_cutoff),
            1,
            0,
        )
    else:
        duplicates_matrix = np.where(
            (abs(diff_matrix_mass) <= mass_cutoff) & (abs(diff_matrix_rt) <= rt_cutoff),
            1,
            0,
        )
    np.fill_diagonal(duplicates_matrix, 0)
    # Find # of duplicates for each row
    row_sums = np.sum(duplicates_matrix, axis=1)
    # Calculate lower triangle of matrix
    duplicates_matrix_lower = np.tril(duplicates_matrix)
    lower_row_sums = np.sum(duplicates_matrix_lower, axis=1)
    # Flag duplicates in new column
    output = df_in.copy()
    output["Duplicate Feature?"] = np.where((row_sums != 0) | (lower_row_sums != 0), 1, 0)
    # Return de-duplicated dataframe (passed) and duplicates (dupes)
    return output


def duplicates(df_in, mass_cutoff, rt_cutoff, ppm, remove):
    """
    Drop duplicates from input dataframe, based on mass_cutoff and rt_cutoff.
    Includes logic statement for determining if the dataframe is too large to
    be processed in a single pass -- TMF 10/27/23

    A new keyword argument 'remove=False' is included here, and now results in
    duplicates using the flag set of functions instead of the remove set of
    functions. This can be coded as a user choice in the future -- TMF 04/11/24
    """
    # Copy the dataframe
    df = df_in.copy()
    # Parse headers to find sample columns
    all_headers = parse_headers(df)
    sam_headers = [item for sublist in all_headers for item in sublist if len(sublist) > 1]
    # Calculate 'all_sample_mean', sort df by 'all_sample_mean', reset index
    df["all_sample_mean"] = df[sam_headers].mean(axis=1)  # mean intensity across all samples
    df.sort_values(by=["all_sample_mean"], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    # Define feature limit of WebApp
    n = 12000
    step = 6000
    # 'if' statement for chunker: if no chunks needed, send to 'dup_matrix', else send to 'chunk_duplicates'
    if remove:
        if df.shape[0] <= n:
            output, dupe_df = dup_matrix_remove(df, mass_cutoff, rt_cutoff, ppm)
        else:
            output, dupe_df = chunk_dup_remove(df, n, step, mass_cutoff, rt_cutoff, ppm)
    else:
        if df.shape[0] <= n:
            output = dup_matrix_flag(df, mass_cutoff, rt_cutoff, ppm)
        else:
            output = chunk_dup_flag(df, n, step, mass_cutoff, rt_cutoff, ppm)
    # Sort output by 'Mass', reset the index, drop 'all_sample_mean'
    output.sort_values(by=["Mass"], inplace=True)
    output.reset_index(drop=True, inplace=True)
    output.drop(["all_sample_mean"], axis=1, inplace=True)
    if remove:
        dupe_df.drop(["all_sample_mean"], axis=1, inplace=True)
        # Return output dataframe with duplicates removed and duplicate dataframe
        return output, dupe_df
    # fillna() to replace nans with 0s
    output["Duplicate Feature?"].fillna(0, inplace=True)
    # Return output dataframe with duplicates flagged
    return output


"""CALCULATE STATISTICS FUNCTIONS"""


def statistics(df_in):
    """
    Calculates statistics (mean, median, std, CV, N_Abun, & Percent Abun) on
    the dataframe. Includes logic statement for determining if the dataframe is
    too large to be processed in a single pass -- TMF 10/27/23
    """
    # Create copy
    df = df_in.copy()
    # Parse headers, get sample headers
    all_headers = parse_headers(df_in)
    sam_headers = [i for i in all_headers if len(i) > 1]
    # Create column names for each statistics from sam_headers
    mean_cols = ["Mean " + i[0][:-1] for i in sam_headers]
    med_cols = ["Median " + i[0][:-1] for i in sam_headers]
    std_cols = ["STD " + i[0][:-1] for i in sam_headers]
    cv_cols = ["CV " + i[0][:-1] for i in sam_headers]
    nabun_cols = ["Detection Count " + i[0][:-1] for i in sam_headers]
    rper_cols = ["Detection Percentage " + i[0][:-1] for i in sam_headers]
    # Concatenate list comprehensions to calculate each statistic for each sample
    means = pd.concat(
        [df[x].mean(axis=1).round(4).rename(col) for x, col in zip(sam_headers, mean_cols)],
        axis=1,
    )
    medians = pd.concat(
        [df[x].median(axis=1, skipna=True).round(4).rename(col) for x, col in zip(sam_headers, med_cols)],
        axis=1,
    )
    stds = pd.concat(
        [df[x].std(axis=1, skipna=True).round(4).rename(col) for x, col in zip(sam_headers, std_cols)],
        axis=1,
    )
    cvs = pd.concat(
        [(stds[scol] / means[mcol]).round(4).rename(col) for mcol, scol, col in zip(mean_cols, std_cols, cv_cols)],
        axis=1,
    )
    nabuns = pd.concat(
        [df[x].count(axis=1).round(0).rename(col) for x, col in zip(sam_headers, nabun_cols)],
        axis=1,
    )
    rpers = pd.concat(
        [
            ((nabuns[ncol] / len(x)).round(4) * 100).rename(col)
            for x, ncol, col in zip(sam_headers, nabun_cols, rper_cols)
        ],
        axis=1,
    )
    # Concatenate all statistics together
    output = pd.concat([df, means, medians, stds, cvs, nabuns, rpers], axis=1)
    # Return dataframe with new statistics columns appended
    return output


def chunk_stats(df_in, mrl_multiplier=3):
    """
    Wrapper function for passing manageable-sized dataframe chunks to 'statistics' -- TMF 10/27/23
    """
    # Create copy
    df = df_in.copy()
    # Set chunk size (i.e., # rows)
    n = 5000
    # 'if' statement for chunks: if no chunks needed, send to 'statistics', else chunk and iterate
    if df.shape[0] < n:
        output = statistics(df)
    else:
        # Create list of Data.Frame chunks
        list_df = [df[i : i + n] for i in range(0, df.shape[0], n)]
        # Instantiate empty list
        li = []
        # iterate through list_df, calculating 'statistics' on chunks and appending to li
        for df in list_df:
            li.append(statistics(df))
        # concatenate li, sort, and calculate 'Rounded_Mass' + 'Max CV Across Samples'
        output = pd.concat(li, axis=0)
    # Sort output mass and add two new columns
    output.sort_values(["Mass", "Retention_Time"], ascending=[True, True], inplace=True)
    output["Rounded_Mass"] = output["Mass"].round(0)
    output["Max CV Across Samples"] = output.filter(regex="CV ").max(axis=1)
    # Define lists to calculate MRL for inclusion in 'Feature_statistics' outputs
    blanks = ["MB", "mb", "mB", "Mb", "blank", "Blank", "BLANK"]
    Mean = output.columns[output.columns.str.contains(pat="Mean ")].tolist()
    Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
    Std = output.columns[output.columns.str.contains(pat="STD ")].tolist()
    Std_MB = [md for md in Std if any(x in md for x in blanks)]
    # Calculate feature MRL
    output["Selected MRL"] = (mrl_multiplier * output[Std_MB[0]]) + output[Mean_MB[0]]
    output["Selected MRL"] = output["Selected MRL"].fillna(output[Mean_MB[0]])
    output["Selected MRL"] = output["Selected MRL"].fillna(0)

    # Calculate 3x, 5x, 10x MRL values explicitly for use by the logic tree - NTAW-377 AC 6/24/2024
    output["MRL (3x)"] = (3 * output[Std_MB[0]]) + output[Mean_MB[0]]
    output["MRL (3x)"] = output["MRL (3x)"].fillna(output[Mean_MB[0]])
    output["MRL (3x)"] = output["MRL (3x)"].fillna(0)
    output["MRL (5x)"] = (5 * output[Std_MB[0]]) + output[Mean_MB[0]]
    output["MRL (5x)"] = output["MRL (5x)"].fillna(output[Mean_MB[0]])
    output["MRL (5x)"] = output["MRL (5x)"].fillna(0)
    output["MRL (10x)"] = (10 * output[Std_MB[0]]) + output[Mean_MB[0]]
    output["MRL (10x)"] = output["MRL (10x)"].fillna(output[Mean_MB[0]])
    output["MRL (10x)"] = output["MRL (10x)"].fillna(0)

    # Return dataframe with statistics calculated and MRL included, to be output as data_feature_stats
    return output


def column_sort_DFS(df_in, passthru):
    """
    Function that sorts columns for the data_feature_statistics outputs -- TMF 11/21/23
    """
    # Copy df and passthru
    df = df_in.copy()
    pt = passthru.copy()
    # Parse headers
    all_headers = parse_headers(df)
    # Get all cols, group roots (i.e., drop unique value from sample groups)
    all_cols = df.columns.tolist()
    non_samples = ["MRL"]
    group_cols = [
        sublist[0][:-1] for sublist in all_headers if len(sublist) > 1 if not any(x in sublist[0] for x in non_samples)
    ]
    # Create list of prefixes to remove non-samples
    prefixes = [
        "Mean ",
        "Median ",
        "CV ",
        "STD ",
        "Detection Count ",
        "Detection Percentage ",
        "Detection",
    ]
    # Isolate sample_groups from prefixes columns
    groups = [item for item in group_cols if not any(x in item for x in prefixes)]
    # Organize front matter
    front_matter = [item for item in all_cols if not any(x in item for x in groups)]
    pt_info = pt.columns.tolist()
    ordering = [
        "Ionization_Mode",
        "Mass",
        "Retention_Time",
        "Selected MRL",
        "MRL (3x)",
        "MRL (5x)",
        "MRL (10x)",
        "Duplicate Feature?",
        "Is Adduct or Loss?",
        "Has Adduct or Loss?",
        "Adduct or Loss Info",
        "Max CV Across Samples",
    ]
    front_matter = [item for item in ordering if item in front_matter]
    front_matter = pt_info + front_matter
    # Organize stats columns
    cols = []
    for sam in groups:
        group_stats = [item for item in all_cols if sam in item]
        cols.append(group_stats)
    stats_cols = sum(cols, [])
    # Combine into new column list
    new_col_org = front_matter + stats_cols
    # Combine df and passthrough
    df = pd.merge(df, pt, how="left", on=["Feature ID"])
    # Subset data with new column list
    df_reorg = df[new_col_org]
    df_reorg["Ionization_Mode"] = df_reorg["Ionization_Mode"].replace("Esi+", "ESI+")
    df_reorg["Ionization_Mode"] = df_reorg["Ionization_Mode"].replace("Esi-", "ESI-")
    df_reorg.rename(columns={"Ionization_Mode": "Ionization Mode", "Retention_Time": "Retention Time"}, inplace=True)
    # Return re-organized dataframe
    return df_reorg


def column_sort_TSR(df_in, passthru):
    """
    Function that sorts columns for the tracer_sample_results outputs -- TMF 11/21/23
    """
    df = df_in.copy()
    pt = passthru.copy()
    all_cols = df.columns.tolist()
    pt_info = pt.columns.tolist()
    # Create list of prefixes to remove non-samples
    prefixes = [
        "Feature ID",
        "Mass",
        "Retention",
        "Ionization_Mode",
        "MRL",
        "Adduct",
        "Duplicate",
        "Total",
        "Max CV",
        "Chemical_Name",
        "Formula",
        "DTXSID",
    ]
    # Isolate sample_groups from prefixes columns
    back_matter = [item for item in all_cols if not any(x in item for x in prefixes)]
    # Organize front matter
    front_matter = [
        "Chemical_Name",
        "DTXSID",
        "Ionization_Mode",
        "Monoisotopic_Mass",
        "Observed Mass",
        "Mass Error (PPM)",
        "Retention_Time",
        "Observed Retention Time",
        "Retention Time Difference",
        "Selected MRL",
        "MRL (3x)",
        "MRL (5x)",
        "MRL (10x)",
        "Duplicate Feature?",
        "Is Adduct or Loss?",
        "Has Adduct or Loss?",
        "Adduct or Loss Info",
        "Total Detection Count",
        "Total Detection Percentage",
        "Max CV Across Samples",
    ]
    front_matter = pt_info + front_matter
    # Combine into new column list
    new_col_org = front_matter + back_matter
    # Combine df and passthrough
    df = pd.merge(df, pt, how="left", on=["Feature ID"])
    # Subset data with new column list
    if "DTXSID" not in df.columns:
        df["DTXSID"] = ""
    df_reorg = df[new_col_org]
    df_reorg["Ionization_Mode"] = df_reorg["Ionization_Mode"].replace("Esi+", "ESI+")
    df_reorg["Ionization_Mode"] = df_reorg["Ionization_Mode"].replace("Esi-", "ESI-")
    df_reorg.rename(
        columns={
            "Monoisotopic_Mass": "Mass",
            "Chemical_Name": "Chemical Name",
            "Ionization_Mode": "Ionization Mode",
            "Retention_Time": "Retention Time",
        },
        inplace=True,
    )
    # Return re-organized dataframe
    return df_reorg


"""FUNCTION FOR CHECKING TRACERS"""


def check_feature_tracers(df, tracers_file, Mass_Difference, Retention_Difference, ppm):
    """
    Function that takes dataframe and the optional tracers input, identifies which features
    are a tracer, and which samples the tracers are present in -- TMF 12/11/23
    """
    df1 = df.copy()
    # logger.info("cft df columns= {}".format(df1.columns.values))
    df2 = tracers_file.copy()
    # logger.info("cft tracers columns= {}".format(df2.columns.values))
    # Get sample names
    prefixes = [
        "Mean ",
        "Median ",
        "CV ",
        "STD ",
        "Detection Count ",
        "Detection Percentage ",
        "Detection",
        "MRL",
        "Selected MRL",
    ]
    all_headers = parse_headers(df1)
    samples = [
        item
        for subgroup in all_headers
        for item in subgroup
        if ((len(subgroup) > 1) and not any(x in item for x in prefixes))
    ]
    # Replace all caps or all lowercase ionization mode with "Esi" in order to match correctly to sample data dataframe
    df2["Ionization_Mode"] = df2["Ionization_Mode"].replace("ESI+", "Esi+")
    df2["Ionization_Mode"] = df2["Ionization_Mode"].replace("esi+", "Esi+")
    df2["Ionization_Mode"] = df2["Ionization_Mode"].replace("ESI-", "Esi-")
    df2["Ionization_Mode"] = df2["Ionization_Mode"].replace("esi-", "Esi-")
    # Create 'Rounded_Mass' variable to merge on
    df2["Rounded_Mass"] = df2["Monoisotopic_Mass"].round(0)
    df1.rename(
        columns={"Mass": "Observed Mass", "Retention_Time": "Observed Retention Time"},
        inplace=True,
    )
    df1["Rounded_Mass"] = df1["Observed Mass"].round(0)
    # Merge df and tracers
    dft = pd.merge(df2, df1, how="left", on=["Rounded_Mass", "Ionization_Mode"])
    # logger.info("cft dfts columns= {}".format(dft.columns.values))
    if ppm:
        dft["Matches"] = np.where(
            (
                abs((dft["Monoisotopic_Mass"] - dft["Observed Mass"]) / dft["Monoisotopic_Mass"]) * 1000000
                <= Mass_Difference
            )
            & (abs(dft["Retention_Time"] - dft["Observed Retention Time"]) <= Retention_Difference),
            1,
            0,
        )
    else:
        dft["Matches"] = np.where(
            (abs(dft["Monoisotopic_Mass"] - dft["Observed Mass"]) <= Mass_Difference)
            & (abs(dft["Retention_Time"] - dft["Observed Retention Time"]) <= Retention_Difference),
            1,
            0,
        )
    dft = dft[dft["Matches"] == 1]
    # Caculate Occurrence Count and % in tracers
    dft["Total Detection Count"] = dft[samples].count(axis=1)
    dft["Total Detection Percentage"] = (dft["Total Detection Count"] / len(samples)) * 100
    # Get 'Matches' info into main df
    dum = dft[["Observed Mass", "Observed Retention Time", "Matches"]].copy()
    # logger.info("cft dum columns= {}".format(dum.columns.values))
    dfc = pd.merge(df1, dum, how="left", on=["Observed Mass", "Observed Retention Time"])
    # logger.info("cft dfc columns= {}".format(dfc.columns.values))
    dfc.rename(
        columns={
            "Observed Mass": "Mass",
            "Observed Retention Time": "Retention_Time",
            "Matches": "Tracer Chemical Match?",
        },
        inplace=True,
    )
    # Drop columns
    dft.drop(["Rounded_Mass", "Matches"], axis=1, inplace=True)
    # np.where to replace nans with 0s
    dfc["Tracer Chemical Match?"].fillna(0, inplace=True)
    # Returns tracers data (dft) and dataframe with 'Tracer Chemical Match?' appended (dfc)
    # logger.info("cft dfts columns= {}".format(dft.columns.values))
    # logger.info("cft dfc columns= {}".format(dfc.columns.values))
    return dft, dfc


"""FUNCTIONS FOR CLEANING FEATURES"""


def replicate_flag(
    df,
    docs,
    controls,
    missing,
    Mean_Samples,
    Replicate_Percent_Samples,
    Mean_MB,
    Std_MB,
    Replicate_Percent_MB,
):
    """
    Function that takes df, docs, controls, and missing, and applies an R flag
    where values in df fail the user-defined value in controls. This is done
    for sample and blank occurrences in docs - blanks are set to 0 in df. -- TMF 04/19/24
    """
    # Flag sample occurrences where feature presence is less than some replicate percentage cutoff
    for mean, N in zip(Mean_Samples, Replicate_Percent_Samples):
        docs[mean] = docs[mean].astype(object)
        docs.loc[((df[N] < controls[0]) & (~missing[mean])), mean] = "R"
    # Flag blanks occurrences where feature presence is less than some replicate percentage cutoff, and remove from blanks
    for mean, Std, N in zip(Mean_MB, Std_MB, Replicate_Percent_MB):
        docs[mean] = docs[mean].astype(object)
        docs.loc[df[N] < controls[2], mean] = "R"
        df.loc[df[N] < controls[2], mean] = 0
        df.loc[df[N] < controls[2], Std] = 0
    return df, docs


def cv_flag(df, docs, controls, Mean_Samples, CV_Samples, missing):
    """
    Function that takes df, docs, controls, and missing, and applies a CV flag
    where values in df fail the user-defined value in controls. This is done
    for sample occurrences in docs, nothing in df changes. -- TMF 04/19/24
    """
    # Create a mask for df based on sample-level CV threshold
    cv_not_met = pd.DataFrame().reindex_like(df[Mean_Samples])
    for mean, CV in zip(Mean_Samples, CV_Samples):
        cv_not_met[mean] = df[CV] > controls[1]
    # Create empty cell mask from the docs dataframe
    cell_empty = docs[Mean_Samples].isnull()
    # append CV flag (CV > threshold) to documentation dataframe
    docs[Mean_Samples] = np.where(cv_not_met & cell_empty & ~missing, "CV", docs[Mean_Samples])
    docs[Mean_Samples] = np.where(
        cv_not_met & ~cell_empty & ~missing,
        docs[Mean_Samples] + ", CV",
        docs[Mean_Samples],
    )
    return docs


def MRL_calc(df, docs, df_flagged, controls, Mean_Samples, Mean_MB, Std_MB):
    """
    Function that calculates a MRL (BlkStd_cutoff) in df, df_flagged, and
    sets it to docs. MRL is 1) mean + 3*std, then 2) mean, then 3) 0. Finally,
    a mask is generated that finds detects in df. -- TMF 04/19/24
    """
    # Get mrl_std_multiplier
    multiplier = controls[3]
    # Calculate feature MRL
    df["BlkStd_cutoff"] = (multiplier * df[Std_MB[0]]) + df[Mean_MB[0]]
    df["BlkStd_cutoff"] = df["BlkStd_cutoff"].fillna(df[Mean_MB[0]])
    df["BlkStd_cutoff"] = df["BlkStd_cutoff"].fillna(0)
    df_flagged["BlkStd_cutoff"] = (multiplier * df_flagged[Std_MB[0]]) + df_flagged[Mean_MB[0]]
    df_flagged["BlkStd_cutoff"] = df_flagged["BlkStd_cutoff"].fillna(df_flagged[Mean_MB[0]])
    df_flagged["BlkStd_cutoff"] = df_flagged["BlkStd_cutoff"].fillna(0)
    docs["BlkStd_cutoff"] = df["BlkStd_cutoff"]
    # Create a mask for docs based on sample-level MRL threshold
    MRL_sample_mask = pd.DataFrame().reindex_like(df[Mean_Samples])
    for x in Mean_Samples:
        # Count the number of detects
        MRL_sample_mask[x] = df[x] > df["BlkStd_cutoff"]
    return df, docs, df_flagged, MRL_sample_mask


def calculate_detection_counts(df, docs, df_flagged, MRL_sample_mask, Std_MB, Mean_MB, Mean_Samples):
    """
    Function that takes df, docs, controls, and the MRL_sample_mask and calculates
    detection counts in df and df_flagged. -- TMF 04/19/24
    """
    # Calculate Detection_Count - sum mask
    df["Detection_Count(non-blank_samples)"] = MRL_sample_mask.sum(axis=1)
    df_flagged["Detection_Count(non-blank_samples)"] = MRL_sample_mask.sum(axis=1)
    # Determine total number of samples
    mean_samples = len(Mean_Samples)
    # Calculate percentage of samples that have a value and store in new column 'Detection_Count(non-blank_samples)(%)'
    df["Detection_Count(non-blank_samples)(%)"] = (df["Detection_Count(non-blank_samples)"] / mean_samples) * 100
    df["Detection_Count(non-blank_samples)(%)"] = df["Detection_Count(non-blank_samples)(%)"].round(1)
    df_flagged["Detection_Count(non-blank_samples)(%)"] = (
        df_flagged["Detection_Count(non-blank_samples)"] / mean_samples
    ) * 100
    df_flagged["Detection_Count(non-blank_samples)(%)"] = df_flagged["Detection_Count(non-blank_samples)(%)"].round(1)
    # Assign to docs
    docs["Detection_Count(non-blank_samples)"] = df["Detection_Count(non-blank_samples)"]
    docs["Detection_Count(non-blank_samples)(%)"] = df["Detection_Count(non-blank_samples)(%)"]
    return df, docs, df_flagged


def MRL_flag(docs, Mean_Samples, MRL_sample_mask, missing):
    """
    Function that takes docs, missing, and the MRL_sample_mask and flags
    non-detects in df (via the MRL_sample_mask) as MRL. -- TMF 04/19/24
    """
    # Update empty cell masks from the docs and df dataframes
    cell_empty = docs[Mean_Samples].isnull()
    # append MRL flag (occurrence < MRL) to documentation dataframe
    docs[Mean_Samples] = np.where(~MRL_sample_mask & cell_empty & ~missing, "MRL", docs[Mean_Samples])
    docs[Mean_Samples] = np.where(
        ~MRL_sample_mask & ~cell_empty & ~missing,
        docs[Mean_Samples] + ", MRL",
        docs[Mean_Samples],
    )
    return docs


def populate_doc_values(df, docs, Mean_Samples, Mean_MB):
    """
    Function that takes df, docs, and populates a value in all cells of docs
    where there is not currently an occurrence flag. Nothing happens to df. -- TMF 04/19/24
    """
    # Mask, add sample values back to doc
    data_values = docs[Mean_Samples].isnull()
    docs[Mean_Samples] = np.where(data_values, df[Mean_Samples], docs[Mean_Samples])
    # Mask, add blank values back to doc
    blank_values = docs[Mean_MB].isnull()
    docs[Mean_MB] = np.where(blank_values, df[Mean_MB], docs[Mean_MB])
    return docs


def feat_removal_flag(docs, Mean_Samples, missing):
    """
    Function that takes docs, and determines whether features should be removed
    by counting the number real occurrences and then labels 'Feature Removed?' by counting
    the number of each type of occurrence flag. Return docs. -- TMF 05/23/24
    """
    # Set all values of feature removed to ""
    docs["Feature Removed?"] = ""
    # Set Possible Occurrence Count to len of Mean_Samples
    docs["Possible Occurrence Count"] = len(Mean_Samples)
    # Generate mask of float values in docs (i.e., occurrences with flags or NaN are False)
    num_mask = pd.concat(
        [pd.to_numeric(docs[mean], errors="coerce").notnull() for mean in Mean_Samples],
        axis=1,
    )
    docs["Final Occurrence Count"] = num_mask.sum(axis=1)
    # Count number of missing samples from missing mask
    docs["# of missing occurrences"] = missing.sum(axis=1)
    docs["Unfiltered Occurrence Count"] = docs["Possible Occurrence Count"] - docs["# of missing occurrences"]
    # Generate mask of str values in docs (i.e., occurrences with ANY flags are True)
    str_mask = pd.concat([docs[mean].str.contains("R|CV|MRL") for mean in Mean_Samples], axis=1)
    docs["Unfiltered Occurrence Removed Count"] = docs["Unfiltered Occurrence Count"] - str_mask.sum(axis=1)
    # Generate mask of str values in docs (i.e., occurrences with R and MRL flags are True)
    str_mask = pd.concat([docs[mean].str.contains("R|MRL") for mean in Mean_Samples], axis=1)
    docs["Unfiltered Occurrence Removed Count (with flags)"] = docs["Unfiltered Occurrence Count"] - str_mask.sum(
        axis=1
    )
    docs["Final Occurrence Count (with flags)"] = (
        docs["Unfiltered Occurrence Count"] - docs["Unfiltered Occurrence Removed Count (with flags)"]
    )

    # Count # of times an occurrence flag contains R, CV, or MRL, and count # of just CV flags
    contains_R = pd.concat([docs[mean].str.contains("R") for mean in Mean_Samples], axis=1)
    contains_CV = pd.concat([docs[mean].str.contains("CV") for mean in Mean_Samples], axis=1)
    is_CV = docs[Mean_Samples] == "CV"
    contains_MRL = pd.concat([docs[mean].str.contains("MRL") for mean in Mean_Samples], axis=1)
    docs["# contains R flag"] = contains_R.sum(axis=1)
    docs["# contains CV flag"] = contains_CV.sum(axis=1)
    docs["# is CV flag"] = is_CV.sum(axis=1)
    docs["# contains MRL flag"] = contains_MRL.sum(axis=1)
    # Determine if any samples are dropped for a feature
    docs["Any Occurrences Removed?"] = np.where(
        (docs["# contains R flag"] > 0) | (docs["# contains CV flag"] > 0) | (docs["# contains MRL flag"] > 0),
        1,
        0,
    )
    # Append feature level flags to features with no real occurrences
    # Feature flag because no occurrences present in input data
    docs["Feature Removed?"] = np.where(
        docs["# of missing occurrences"] == len(Mean_Samples), "NO DETECTIONS ", docs["Feature Removed?"]
    )
    # Feature flag because occurrences fail detection threshold
    docs["Feature Removed?"] = np.where(
        (docs["Final Occurrence Count"] == 0) & (docs["# contains MRL flag"] > 0),
        docs["Feature Removed?"] + "MRL, ",
        docs["Feature Removed?"],
    )
    # Feature flag because occurrences fail CV threshold
    docs["Feature Removed?"] = np.where(
        (docs["Final Occurrence Count"] == 0) & (docs["# contains CV flag"] > 0),
        docs["Feature Removed?"] + "CV, ",
        docs["Feature Removed?"],
    )
    # Feature flag because occurrences fail Replication threshold
    docs["Feature Removed?"] = np.where(
        (docs["Final Occurrence Count"] == 0) & docs["# contains R flag"] > 0,
        docs["Feature Removed?"] + "R ",
        docs["Feature Removed?"],
    )
    return docs


def occ_drop_df(df, docs, df_flagged, Mean_Samples):
    """
    Function that takes df, docs, df_flagged and creates a mask for each filter
    applied to docs (R, CV, MRL). All masks applied to df, only R and MRL masks applied
    to df_flagged. Return df and df_flagged. -- TMF 04/19/24
    """
    # Copy 'Any Occurrences Removed?' to df and df_flagged
    df["Any Occurrences Removed?"] = docs["Any Occurrences Removed?"]
    df_flagged["Any Occurrences Removed?"] = docs["Any Occurrences Removed?"]
    # Create mask of occurrences dropped for replicate flag
    rep_fails = pd.concat([docs[mean].str.contains("R") for mean in Mean_Samples], axis=1).fillna(False)
    # Mask df and df_flagged
    df[Mean_Samples] = df[Mean_Samples].mask(rep_fails)
    df_flagged[Mean_Samples] = df_flagged[Mean_Samples].mask(rep_fails)
    # Create mask of occurrences dropped for replicate flag
    non_detects = pd.concat([docs[mean].str.contains("MRL") for mean in Mean_Samples], axis=1).fillna(False)
    # Mask df and df_flagged
    df[Mean_Samples] = df[Mean_Samples].mask(non_detects)
    df_flagged[Mean_Samples] = df_flagged[Mean_Samples].mask(non_detects)
    # Create mask of occurrences dropped for replicate flag
    cv_fails = pd.concat([docs[mean].str.contains("CV") for mean in Mean_Samples], axis=1).fillna(False)
    # Mask df
    df[Mean_Samples] = df[Mean_Samples].mask(cv_fails)
    # Add columns from docs to df / df_flagged
    df["Possible Occurrence Count"] = docs["Possible Occurrence Count"]
    df_flagged["Possible Occurrence Count"] = docs["Possible Occurrence Count"]
    # df["Possible Occurrences Removed Count"] = docs["Possible Occurrences Removed Count"]
    # df_flagged["Possible Occurrences Removed Count"] = docs["Possible Occurrences Removed Count"]
    df["Final Occurrence Count"] = docs["Final Occurrence Count"]
    df_flagged["Final Occurrence Count (with flags)"] = docs["Final Occurrence Count (with flags)"]
    # Calculate Final Occurrence Percentage
    df["Final Occurrence Percentage"] = (
        (df["Final Occurrence Count"] / df["Possible Occurrence Count"]).astype(float).round(2)
    )
    df_flagged["Final Occurrence Percentage (with flags)"] = (
        (df_flagged["Final Occurrence Count (with flags)"] / df_flagged["Possible Occurrence Count"])
        .astype(float)
        .round(2)
    )
    return df, df_flagged


def feat_drop_df(df, docs, df_flagged):
    """
    Function that takes df, docs, df_flagged, and uses the Feature Removed? column
    from docs to subset df and df_flagged. All features that have a removal flag
    are removed from df, only features with the R, MRL, and CV flags are removed
    from df_flagged. -- TMF 04/19/24
    """
    # Copy 'Feature Removed?' column onto df and df_flagged
    df["Feature Removed?"] = docs["Feature Removed?"]
    df_flagged["Feature Removed?"] = docs["Feature Removed?"]
    # Subset df and df_flagged
    df = df.loc[df["Feature Removed?"] == "", :]
    df_flagged = df_flagged.loc[(df_flagged["Feature Removed?"] == "") | (docs["# is CV flag"] > 0), :]
    # Drop 'Feature Removed?' from df
    df.drop(columns=["Feature Removed?"], inplace=True)
    df_flagged.drop(columns=["Feature Removed?"], inplace=True)
    return df, df_flagged


def clean_features(df_in, controls, tracer_df=False):
    """
    Function that removes (blanks out) observations at feature and occurrence level
    based on user-defined thresholds for replicate percent and CV threshold, and
    the calculated MRL value. The removed features/ocurrences are documented in
    an additional dataframe (docs). Sample-level detection counts are also calculated.
    This is an object-oriented version of the original fucntion that importantly
    makes occurrence and feature removal decisions based directly on flags in docs. -- TMF 04/19/24
    """
    # Make dataframe copy, create docs in df's image
    df = df_in.copy()
    df["Any Occurrences Removed?"] = np.nan
    docs = pd.DataFrame().reindex_like(df)
    docs["Mass"] = df["Mass"]
    docs["Retention_Time"] = df["Retention_Time"]
    docs["Feature ID"] = df["Feature ID"]
    docs["Duplicate Feature?"] = df["Duplicate Feature?"]
    if tracer_df:
        docs["Tracer Chemical Match?"] = df["Tracer Chemical Match?"]
    # Define lists
    blanks = ["MB", "mb", "mB", "Mb", "blank", "Blank", "BLANK"]
    Abundance = df.columns[df.columns.str.contains(pat="Detection Percentage ")].tolist()
    Replicate_Percent_MB = [N for N in Abundance if any(x in N for x in blanks)]
    Replicate_Percent_Samples = [N for N in Abundance if not any(x in N for x in blanks)]
    Mean = df.columns[df.columns.str.contains(pat="Mean ")].tolist()
    Mean_Samples = [md for md in Mean if not any(x in md for x in blanks)]
    Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
    Std = df.columns[df.columns.str.contains(pat="STD ")].tolist()
    Std_MB = [md for md in Std if any(x in md for x in blanks)]
    CV = df.columns[df.columns.str.startswith("CV ")].tolist()
    CV_Samples = [C for C in CV if not any(x in C for x in blanks)]
    missing = df[Mean_Samples].isnull()
    """REPLICATE FLAG"""
    # Implement replicate flag
    df, docs = replicate_flag(
        df,
        docs,
        controls,
        missing,
        Mean_Samples,
        Replicate_Percent_Samples,
        Mean_MB,
        Std_MB,
        Replicate_Percent_MB,
    )
    # Create a copy of df prior to CV flag/filter step - this DF will not remove occurrences/features failing CV threshold
    df_flagged = df.copy()
    """CV FLAG"""
    # Implement CV flag
    docs = cv_flag(df, docs, controls, Mean_Samples, CV_Samples, missing)
    """MRL CALCULATION/MRL MASK GENERATION"""
    # Calculate feature MRL
    df, docs, df_flagged, MRL_sample_mask = MRL_calc(df, docs, df_flagged, controls, Mean_Samples, Mean_MB, Std_MB)
    """CALCULATE DETECTION COUNTS"""
    # Calculate Detection_Count
    df, docs, df_flagged = calculate_detection_counts(
        df, docs, df_flagged, MRL_sample_mask, Std_MB, Mean_MB, Mean_Samples
    )
    """MRL FLAG"""
    # Implement MRL flag
    docs = MRL_flag(docs, Mean_Samples, MRL_sample_mask, missing)
    """ADD VALUES TO DOC"""
    # Populate docs values
    docs = populate_doc_values(df, docs, Mean_Samples, Mean_MB)
    """DOCUMENT DROP FEATURES FROM DF"""
    # Annotation feature removal
    docs = feat_removal_flag(docs, Mean_Samples, missing)
    """DROP OCCURRENCES FROM DF"""
    # Remove proper occurrences from df and df_flagged
    df, df_flagged = occ_drop_df(df, docs, df_flagged, Mean_Samples)
    """DROP FEATURES FROM DF"""
    # Remove features from df and df_flagged
    df, df_flagged = feat_drop_df(df, docs, df_flagged)
    return df, docs, df_flagged


def Blank_Subtract_Mean(df_in):
    """
    Calculate the mean blank intensity for each feature and subtract that value from
    each sample's mean value for that feature
    """
    df = df_in.copy()
    # Define lists; blanks, means, sample means, and blank means
    blanks = ["MB", "mb", "mB", "Mb", "blank", "Blank", "BLANK"]
    Mean = df.columns[df.columns.str.contains(pat="Mean ")].tolist()
    Mean_Samples = [md for md in Mean if not any(x in md for x in blanks)]
    Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
    # Iterate through sample means, subtracting blank mean into new column
    for mean in Mean_Samples:
        # Create new column, do subtraction
        df["BlankSub " + str(mean)] = df[mean].sub(df[Mean_MB[0]], axis=0)
        # Clip values at 0, replace 0s with NaN
        df["BlankSub " + str(mean)] = df["BlankSub " + str(mean)].clip(lower=0).replace({0: np.nan})
    return df


"""FUNCTIONS FOR COMBINING DATAFRAMES / FILE PREPARATION"""


def combine(df1, df2):
    """
    Function to combine positive and negative mode dataframes into df_combined
    """
    # Recombine dfs
    if df1 is not None and df2 is not None:
        dfc = pd.concat([df1, df2], sort=True)  # fixing pandas FutureWarning
        dfc = dfc.reindex(columns=df1.columns)
    elif df1 is not None:
        dfc = df1.copy()
    else:
        dfc = df2.copy()
    # Get column names
    columns = dfc.columns.values.tolist()
    # Drop duplicates (should not be any)
    dfc = dfc.drop_duplicates(subset=["Mass", "Retention_Time"])
    # Get sample Means
    Mean_list = dfc.columns[
        (dfc.columns.str.contains(pat="Mean ") == True)
        & (dfc.columns.str.contains(pat="MB|blank|blanks|BlankSub|_x|_y") == False)
    ].tolist()
    # Count sample-level occurrences and median of means
    dfc["N_Abun_Samples"] = dfc[Mean_list].count(axis=1, numeric_only=True)
    dfc["Mean_Abun_Samples"] = dfc[Mean_list].median(axis=1, skipna=True).round(0)
    # Sort by 'Mass' and 'Retention_Time'
    dfc = dfc[columns].sort_values(["Mass", "Retention_Time"], ascending=[True, True])
    # Return combined dataframe
    return dfc


def combine_doc(doc1, doc2, tracer_df=False):
    """
    Function to combine positive and negative mode docs for filter_documentation sheet
    """
    # Define blank sub-strings
    blanks = ["MB", "mb", "mB", "Mb", "blank", "Blank", "BLANK"]
    # Recombine doc and dupe
    if doc1 is not None and doc2 is not None:
        # Get Mean columns for blanks and samples
        Mean = doc1.columns[doc1.columns.str.contains(pat="Mean ")].tolist()
        Mean_Samples = [md for md in Mean if not any(x in md for x in blanks)]
        Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
        dfc = pd.concat([doc1, doc2], sort=True)  # fixing pandas FutureWarning
        dfc = dfc.reindex(columns=doc1.columns)
    elif doc1 is not None:
        # Get Mean columns for blanks and samples
        Mean = doc1.columns[doc1.columns.str.contains(pat="Mean ")].tolist()
        Mean_Samples = [md for md in Mean if not any(x in md for x in blanks)]
        Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
        dfc = doc1.copy()
    else:
        # Get Mean columns for blanks and samples
        Mean = doc2.columns[doc2.columns.str.contains(pat="Mean ")].tolist()
        Mean_Samples = [md for md in Mean if not any(x in md for x in blanks)]
        Mean_MB = [md for md in Mean if any(x in md for x in blanks)]
        dfc = doc2.copy()
    # Select columns for keeping, with tracer conditional
    if tracer_df:
        to_keep = (
            [
                "Feature ID",
                "BlkStd_cutoff",
                "Tracer Chemical Match?",
                "Duplicate Feature?",
                "Feature Removed?",
                "Possible Occurrence Count",
                "Unfiltered Occurrence Count",
                "Unfiltered Occurrence Removed Count",
                "Final Occurrence Count",
                "Unfiltered Occurrence Removed Count (with flags)",
                "Final Occurrence Count (with flags)",
            ]
            + Mean_MB
            + Mean_Samples
        )
    else:
        to_keep = (
            [
                "Feature ID",
                "BlkStd_cutoff",
                "Duplicate Feature?",
                "Feature Removed?",
                "Possible Occurrence Count",
                "Unfiltered Occurrence Count",
                "Unfiltered Occurrence Removed Count",
                "Final Occurrence Count",
                "Unfiltered Occurrence Removed Count (with flags)",
                "Final Occurrence Count (with flags)",
            ]
            + Mean_MB
            + Mean_Samples
        )
    # Subset with columns to keep; change 'BlkStd_cutoff' to MRL
    dfc = dfc[to_keep]
    dfc.rename({"BlkStd_cutoff": "Selected MRL"}, axis=1, inplace=True)
    # Sort by 'Mass' and 'Retention_Time'
    dfc = dfc.sort_values(["Feature ID"], ascending=[True])
    # Return filter_documentation dataframe with removed duplicates appended
    return dfc


def MPP_Ready(dft, pts, tracer_df=False, flagged=False, directory="", file=""):
    """
    Function that re-combines the pass-through columns with the processed dataframe
    plus some final column sorting
    """
    # If/elif/else to combine pass through columns with dft
    # Assign pass through columns to pt_cols for re_org
    if pts[0] is not None and pts[1] is not None:
        pt_com = pd.concat([pts[0], pts[1]], axis=0)
        dft = pd.merge(dft, pt_com, how="left", on=["Feature ID"])
        pt_cols = pts[0].columns.tolist()
        # pt_cols = [col for col in pt_cols if "Feature ID" not in col]
    elif pts[0] is not None:
        dft = pd.merge(dft, pts[0], how="left", on=["Feature ID"])
        pt_cols = pts[0].columns.tolist()
        # pt_cols = [col for col in pt_cols if "Feature ID" not in col]
    else:
        dft = pd.merge(dft, pts[1], how="left", on=["Feature ID"])
        pt_cols = pts[1].columns.tolist()
        # pt_cols = [col for col in pt_cols if "Feature ID" not in col]
    # Parse headers, get sample values and blank subtracted means
    Headers = parse_headers(dft)
    raw_samples = [
        item
        for sublist in Headers
        for item in sublist
        if (len(sublist) > 2)
        if ("BlankSub" not in item)
        if not any(x in item for x in pt_cols)
    ]
    blank_subtracted_means = dft.columns[dft.columns.str.contains(pat="BlankSub")].tolist()

    # Check for 'Formula' (should be deprecated), then check for tracer_df
    # Format front matter accordingly, add pt_cols, raw_samples, blank_subtracted_means
    if flagged:
        if tracer_df:
            cols = (
                pt_cols
                + [
                    "Ionization_Mode",
                    "Mass",
                    "Retention_Time",
                    "Tracer Chemical Match?",
                    "Duplicate Feature?",
                    "Is Adduct or Loss?",
                    "Has Adduct or Loss?",
                    "Adduct or Loss Info",
                    "Final Occurrence Count (with flags)",
                    "Final Occurrence Percentage (with flags)",
                ]
                + raw_samples
                + blank_subtracted_means
            )
            dft = dft[cols]
        else:
            cols = (
                pt_cols
                + [
                    "Ionization_Mode",
                    "Mass",
                    "Retention_Time",
                    "Duplicate Feature?",
                    "Is Adduct or Loss?",
                    "Has Adduct or Loss?",
                    "Adduct or Loss Info",
                    "Final Occurrence Count (with flags)",
                    "Final Occurrence Percentage (with flags)",
                ]
                + raw_samples
                + blank_subtracted_means
            )
            dft = dft[cols]
    else:
        if tracer_df:
            cols = (
                pt_cols
                + [
                    "Ionization_Mode",
                    "Mass",
                    "Retention_Time",
                    "Tracer Chemical Match?",
                    "Duplicate Feature?",
                    "Is Adduct or Loss?",
                    "Has Adduct or Loss?",
                    "Adduct or Loss Info",
                    "Final Occurrence Count",
                    "Final Occurrence Percentage",
                ]
                + raw_samples
                + blank_subtracted_means
            )
            dft = dft[cols]
        else:
            cols = (
                pt_cols
                + [
                    "Ionization_Mode",
                    "Mass",
                    "Retention_Time",
                    "Duplicate Feature?",
                    "Is Adduct or Loss?",
                    "Has Adduct or Loss?",
                    "Adduct or Loss Info",
                    "Final Occurrence Count",
                    "Final Occurrence Percentage",
                ]
                + raw_samples
                + blank_subtracted_means
            )
            dft = dft[cols]
    # Rename columns
    dft["Ionization_Mode"] = dft["Ionization_Mode"].replace("Esi+", "ESI+")
    dft["Ionization_Mode"] = dft["Ionization_Mode"].replace("Esi-", "ESI-")
    dft.rename({"Ionization_Mode": "Ionization Mode", "Retention_Time": "Retention Time"}, axis=1, inplace=True)
    # Return re-combined, sorted dataframe for output as 'Cleaned_feature_results_reduced' and 'Results_flagged'
    return dft


def calc_toxcast_percent_active(df):
    dft = df.copy()

    # Extract out the total and active numeric values from the TOTAL_ASSAYS_TESTED column
    TOTAL_ASSAYS = "\/([0-9]+)"  # a regex to find the digits after a slash
    dft["TOTAL_ASSAYS_TESTED"] = (
        dft["TOXCAST_NUMBER_OF_ASSAYS/TOTAL"].astype("str").str.extract(TOTAL_ASSAYS, expand=True)
    )
    NUMBER_ASSAYS = "([0-9]+)\/"  # a regex to find the digits before a slash
    dft["NUMBER_ACTIVE_ASSAYS"] = (
        dft["TOXCAST_NUMBER_OF_ASSAYS/TOTAL"].astype("str").str.extract(NUMBER_ASSAYS, expand=True)
    )

    # Convert the value columns to floats and do division to get the percent active value
    dft["TOTAL_ASSAYS_TESTED"] = dft["TOTAL_ASSAYS_TESTED"].astype(float)
    dft["NUMBER_ACTIVE_ASSAYS"] = dft["NUMBER_ACTIVE_ASSAYS"].astype(float)
    dft["TOXCAST_PERCENT_ACTIVE"] = dft["NUMBER_ACTIVE_ASSAYS"] / dft["TOTAL_ASSAYS_TESTED"] * 100
    dft["TOXCAST_PERCENT_ACTIVE"] = dft["TOXCAST_PERCENT_ACTIVE"].apply(lambda x: round(x, 2))

    # Clean up and remove the temporary value columns
    dft = dft.drop(["TOTAL_ASSAYS_TESTED", "NUMBER_ACTIVE_ASSAYS"], axis=1)

    return dft


# The following function calculates a "width" of a string based on the characters within, as some characters are large, medium or skinny
# These widths are used to determine the spacing of the group labels on the run sequence plot
def determine_string_width(input_string):
    big_letters = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "K",
        "M",
        "O",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "m",
    ]
    medium_letters = [
        "E",
        "F",
        "L",
        "N",
        "P",
        "a",
        "b",
        "c",
        "d",
        "e",
        "g",
        "h",
        "k",
        "n",
        "o",
        "p",
        "q",
        "s",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
    ]
    skinny_letters = ["I", "J", "f", "i", "j", "l", "r", "t", "1"]
    big_increment = 0.02
    medium_increment = 0.015
    skinny_increment = 0.007

    temp_increment = 0
    for j in range(len(input_string)):
        if input_string[j] in big_letters:
            temp_increment = temp_increment + big_increment
            print("big")
        elif input_string[j] in medium_letters:
            temp_increment = temp_increment + medium_increment
            print("medium")
        else:
            print("skinny")
            temp_increment = temp_increment + skinny_increment
    return temp_increment
