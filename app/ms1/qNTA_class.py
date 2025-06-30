import pandas as pd
import numpy as np
from scipy import stats
import logging

from matplotlib import pyplot as plt
import seaborn as sns
import re
from numba import jit

logger = logging.getLogger("nta_app.ms1")


class qNTAClass:
    """
    Class used to store qNTA data and produce data quality review visualizations
    and concentration estimates with confidence intervals, with calculated performance
    metrics if validation data is provided.

    ...

    Attributes
    ----------
    surrogate_cal_data : pandas DataFrame
                        A DataFrame containing the qNTA surrogate statistics from INTERPRET NTA
    surrogate_cal_data_long : pandas DataFrame
                        A DataFrame containing the qNTA surrogate statistics from
                        INTERPRET NTA in long format
    surrogate_cal_data_long_nonzero: pandas DataFrame
                        A DataFrame containing the qNTA surrogate statistics from
                        INTERPRET NTA in long format for only BlankSub Mean abundances > 0
    surrogate_cal_data_long_nonzero_chems : pandas Series
                        A Series containing the unique chemicals in surrogate_cal_data_long_nonzero
    all_cal_models : list
                        A list containing tuples of sm.OLS object (linear model) for a chemical and
                        str object of the chemical name, as output by fit_cal_curve_model
    all_cal_plots : list
                        A list containing tuples of matplotlib figure for a calibration curve and
                        str object of the plotted chemical, as output by plot_cal_curve
    occurrence_data : pandas DataFrame
                        A DataFrame containing abundance occurrence data (columns are samples, rows
                        are features) for which qNTA estimates will be made
    validation_data : pandas DataFrame
                        A DataFrame containing targeted analysis concentrations (columns are samples,
                        rows are features) to compare against qNTA estimates and perform validation
                        using performance metrics
    """

    # This uses an intermediate output from INTERPRET NTA, which first takes in the Detection Matrix
    # input file and qNTA surrogate input file
    # Pass in Ionization Mode as a str argument or determine it from the Ionization Mode column?
    # (Currently separating the input data beforehand)
    def __init__(self, surrogate_cal_data, validation_input=None, occurrence_input=None, parameters=None):
        """
        Pivots surrogate_cal_data DataFrame from wide to long format, keeping only
            blank-subtracted means > 0
        Adds LogAbun and LogCon columns (log10-transformed)
        Stores unique surrogate chemicals in surrogate_cal_data_long_nonzero as
            surrogate_cal_data_long_nonzero_chems
        """
        # Required uploaded files
        # Replace NA values with 0 to avoid issues with mathematical operations on DataFrame
        self.parameters = parameters
        self.surrogate_cal_data = surrogate_cal_data.fillna(0)
        self.validation_data = validation_input
        self.occurrence_data = occurrence_input
        # Dataframes to be calculated
        self.surrogate_cal_data_long_nonzero = None
        self.surrogate_cal_data_long_nonzero_chems = None
        self.occurrences_data_nonzero = None
        self.RF_estimate_out = None
        self.percentiles = None
        self.RF_percs = None
        self.RF_array = None
        self.validation_out = None
        self.summary_out = None
        # Plot holders
        self.all_cal_models = None
        self.all_cal_plots = None
        self.cc_metrics = None
        self.bootstrap_array = None

    def execute(self):
        """Perform data manipulation functions"""
        # logger.info("val data type = {}".format(type(self.validation_data)))
        self.check_parameters()
        # logger.info("val data type = {}".format(type(self.validation_data)))
        self.check_occurrences()
        # logger.info("val data type = {}".format(type(self.validation_data)))
        self.check_RF_input()
        # logger.info("val data type = {}".format(type(self.validation_data)))
        self.check_validation_data()
        # logger.info("val data type = {}".format(type(self.validation_data)))
        """Perform Calibration Curve Methods"""
        self.cal_curve_all_metrics()
        """Perform Bootstrap Methods"""
        # Calculate Response factor percentiles
        self.RF_array = self.make_RF_array(
            self.surrogate_cal_data_long_nonzero,
        )
        self.RF_percs = pd.DataFrame(
            self.RF_bootstrap_numba_full(
                self.RF_array,
                seed=self.parameters["seed"],
                reps=self.parameters["reps"],
                alpha=self.parameters["alpha"],
            ),
            index=pd.Index(["Minimum", "Median", "Maximum"], name="Response Factor Percentile Estimate"),
            columns=[str(x) + "th" for x in self.percentiles],
        ).reset_index()
        # Calculate RF estimates for each chemical
        self.RF_estimate_out = self.RF_boot_estimate(
            self.surrogate_cal_data_long_nonzero,
            self.occurrence_data,
            seed=self.parameters["seed"],
            reps=self.parameters["reps"],
            alpha=self.parameters["alpha"],
            rep_range=self.parameters["rep_range"],
            long_form=self.parameters["long_form"],
        )
        logger.info("val_out data type = {}".format(type(self.validation_out)))
        # Perform bootstrap validation
        self.validation_out = self.RF_boot_validation(
            seed=self.parameters["seed"],
            reps=self.parameters["reps"],
            alpha=self.parameters["alpha"],
            rep_range=self.parameters["rep_range"],
            long_form=self.parameters["long_form"],
            LOO=self.parameters["LOO"],
            internal=self.parameters["internal"],
        )
        logger.info("val_out data type = {}".format(type(self.validation_out)))
        # Check status of self.validation_out
        if self.validation_out is not None and len(self.validation_out) > 0:
            # Create validation summary
            self.summary_out = self.validation_summary(
                self.validation_out,
                long_form=self.parameters["long_form"],
                LOO=self.parameters["LOO"],
            )

    """DATA MANIPULATION FUNCTIONS"""

    def check_parameters(self):
        if self.parameters is not None:
            self.percentiles = np.multiply([self.parameters["alpha"] / 2, 0.5, 1 - (self.parameters["alpha"] / 2)], 100)
        else:
            self.parameters = {
                "seed": 1,
                "reps": 10000,
                "alpha": 0.05,
                "rep_range": True,
                "long_form": True,
                "LOO": True,
                "internal": False,
            }
            self.percentiles = np.multiply([self.parameters["alpha"] / 2, 0.5, 1 - (self.parameters["alpha"] / 2)], 100)

    def check_occurrences(self):
        """
        Set attribute occurrence_data in qNTAClass object.

        WE WANT ALL FEATURES ALL SAMPLES, ELSE USE INTERNAL - THIS WILL READ IN FROM
        dfs and what we want is all features, sample BlankSub columns

        Parameters
        ----------
        occurrence_data : pandas DataFrame
            DataFrame containing columns for "Chemical Name" and samples, with
            samples column names beginning with "BlankSub Mean " and values
            containing BlankSub Mean abundances.

        Returns
        -------
        None.

        """
        # Copy input
        occ = self.occurrence_data
        val = self.validation_data
        if occ is not None:
            # Get cols (only take columns also in val; e.g., no Pool)
            front = [col for col in occ.columns if any(x in col for x in ["Feature", "Chemical", "Retention"])]
            if val is not None:
                back = [
                    col for col in occ.columns if col.startswith("BlankSub Mean") and any(x in col for x in val.columns)
                ]
            else:
                back = [col for col in occ.columns if col.startswith("BlankSub Mean")]
            # Pare occ down to front + back
            self.occurrence_data = occ[front + back]
        else:
            # Copy input
            surr = self.surrogate_cal_data.copy()
            # Get cols
            front = ["Feature ID", "Chemical Name", "Retention Time"]
            back = [col for col in surr.columns if col.startswith("BlankSub Mean")]
            cols = front + back
            # Subset columns from self.surrogate_cal_data, store
            occ = surr[cols]
            self.occurrence_data = occ

    def check_RF_input(self):
        """
        Set attribute surrogate_cal_data_long_nonzero and surrogate_cal_data_long_nonzero_chems
        in the qNTAClass object.

        Parameters
        ----------
        self : uses self.surrogate_cal_data pandas dataframe

        Returns
        -------
        None.
        """
        # Copy input
        surr = self.surrogate_cal_data.copy()
        # Define controls
        controls = ["control", "Control", "CONTROL"]
        # Check for Control - if present we want ControlSub, else we want BlankSub
        if any(item for item in surr.columns if any(x in item for x in controls)):
            col = "ControlSub BlankSub Mean"
        else:
            col = "BlankSub Mean"
        # Get cols
        prefixes = ["Mean", "STD", "CV", "Detection Count", "Detection Percentage", "Conc", "RF"] + [col]
        cols = ["Feature ID", "Chemical Name", "Retention Time", "Ionization Mode"] + [
            col for col in surr.columns if any(col.startswith(x) for x in prefixes)
        ]
        # Pivot surrogate_cal_data wide to long
        long = pd.wide_to_long(surr[cols], stubnames=prefixes, i="Feature ID", j="Cal Level", sep=" ", suffix="\\w+")
        # Change Conc column to numeric
        long["Conc"] = pd.to_numeric(long["Conc"])
        # Keep only BlankSub Mean abundances > 0 to avoid problems with log-10 transform
        # we also don't want to have RFs of 0 in the surrogate set
        long_nz = long.loc[long[col] > 0, :]
        # Add log-10 transformed columns for BlankSub Mean Abundance and Concentration
        long_nz = long_nz.assign(LogAbun=np.log10(long_nz[col]), LogConc=np.log10(long_nz["Conc"]))
        # Store unique chemical names in class variable
        self.surrogate_cal_data_long_nonzero_chems = np.unique(long_nz["Chemical Name"])
        # Store df in class variable
        self.surrogate_cal_data_long_nonzero = long_nz

    def check_validation_data(self):
        """
        Set attribute validation_data in qNTAClass object.

        Parameters
        ----------
        validation_data : pandas DataFrame
            DataFrame containing columns for samples, whose values are targeted
            concentrations; the units of concentration must be the same as those
            used to generate the qNTA surrogate RF data.

        Returns
        -------
        None.

        """
        # Copy input
        val = self.validation_data
        surr = self.surrogate_cal_data.copy()
        occ = self.occurrence_data
        # Check if val has been submitted
        if val is not None:
            # Make sure val columns match occ columns
            # Get occ cols without 'BlankSub Mean ' header
            occ_cols = [col[14:] for col in occ.columns if col.startswith("BlankSub")]
            # Get all val columns that match any strings/substrings in occ_cols
            val_cols = [col for col in val.columns if any(col in x for x in occ_cols)]
            # Check for exact equality
            if not pd.Series(occ_cols).equals(pd.Series(val_cols)):
                # If not equal, create dict to swap val_cols with occ_cols
                col_swap = {key: val for key in val_cols for val in occ_cols if key in val}
                val = val.rename(columns=col_swap)
            # Check for 'Feature ID' in val
            if "Feature ID" in val.columns:
                # If val contains 'Feature ID' do nothing
                pass
            else:
                # if val doesn't contain 'Feature ID', merge column from
                val = pd.merge(val, surr[["Feature ID", "Chemical Name"]], how="left", on="Chemical Name")
                self.validation_data = val.copy()
        # If val has not been submitted
        else:
            blanks = ["Blank", "blank", "BLANK", "MB", "Mb", "mb", "mB"]
            controls = ["control", "Control", "CONTROL"]
            li = blanks + controls
            # Get Conc col root names, avoiding any blanks or controls
            cols = [col for col in surr.columns if "Conc " in col if not any(x in col for x in li)]
            # Set Chemical Name as ID column for future joins
            val = surr.loc[:, ["Chemical Name", "Feature ID"] + cols]
            # Rename to remove "Conc "
            val = val.rename(columns={col: col[5:] for col in val.columns if "Conc " in col})
            # Use copy to avoid overwriting original data
            self.validation_data = val.copy()
            # Set parameters 'internal' to True
            self.parameters["internal"] = True

    """CALIBRATION CURVE METHODS"""

    def cal_curve_metrics(
        self,
        chem,
    ):
        """
        Subsets qNTA surrogate calibration data (long form) to data from the chemical
        provided as input ('chem').
        Uses the atrribute 'surrogate_cal_data_long_nonzero'.
        Calculates and returns calibration curve (LogAbun vs LogConc) metrics.

        Parameters
        ----------
        chem : str
            Name of qNTA surrogate chemical used for calibration curve model

        Returns
        -------
        If the subset qNTA surrogate calibration data has fewer than three points,
        returns string with error message
        Else, return tuple with chemical name, slope, and R^2 value

        """
        # Copy df
        surr = self.surrogate_cal_data_long_nonzero.copy()
        # Subset by chem
        cal_data = surr.loc[surr["Chemical Name"] == chem]
        # Get Ionization Mode value
        im = cal_data["Ionization Mode"].values[0]
        # Check if there are more than 3 points
        if len(cal_data) < 3:
            # If no, return string
            return "Fewer than 3 calibration points"
        else:
            # Calculate slope, intercept, r_value, p_value, and std_err from df, given x_col and y_col
            slope, intercept, r_value, p_value, std_err = stats.linregress(cal_data["LogConc"], cal_data["LogAbun"])
            # Calculate r_squared value
            r_squared = r_value**2
            # Return tuple
            return (
                chem,
                im,
                slope.round(3),
                intercept.round(3),
                r_squared.round(3),
            )

    def cal_curve_all_metrics(
        self,
    ):
        """
        Calculate chemical-wise calibration curve metrics for all qNTA surrogate
        chemicals in the qNTA surrogate statistics DataFrame. Store calibration
        curve metrics (slope and R^2 values) to self.cc_metrics

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # Create list of tuples from plot_cal_curve() on all items in all_call_models
        cc_tuples = [self.cal_curve_metrics(i) for i in self.surrogate_cal_data_long_nonzero_chems]
        cc_tuples = [i for i in cc_tuples if "Fewer than 3 calibration points" not in i]
        # Generate and save dataframe
        self.cc_metrics = pd.DataFrame(
            cc_tuples, columns=["Chemical Name", "Ionization Mode", "Slope", "Intercept", "R-squared"]
        )

    """RESPONSE FACTOR BOOTSTRAP METHODS"""

    @staticmethod
    def make_RF_array(RF_data):
        """
        Prepares numpy array that is input to RF_bootstrap_numba_full

        Parameters
        ----------
        RF_data : pandas DataFrame
            DataFrame containing "Chemical_Name" and "RF" columns

        Returns
        -------
        numpy array where [0] is a numeric chemical identifier and [1] is an RF value from a surrogate chemical

        """
        RF_data_row_num = pd.merge(
            RF_data.copy(),
            pd.DataFrame(
                {
                    "Chemical Name": pd.unique(RF_data["Chemical Name"]),
                    "row_number": np.arange(0, len(pd.unique(RF_data["Chemical Name"]))),
                }
            ),
        )
        RF_array = np.array([RF_data_row_num["row_number"], RF_data_row_num["RF"]])
        return RF_array

    @staticmethod
    @jit(nopython=True)
    def RF_bootstrap_numba_full(
        RF_array, seed=1, reps=10000, alpha=0.05
    ):  # Function is compiled to machine code when called the first time
        """ "
        Performs hierarchical response factor bootstrap (choosing one chemical, then one of its RFs)

        Parameters
        ----------
        RF_array : numpy array
            2D array, where [0] contains the chemical index and [1] contains the qNTA surrogate RF
        seed : int, optional
            Seed used for the random bootstrap sampling (np.random.choice()). The default is 1.
        reps : int, optional
            Number of bootstrap repetitions. The default is 10000.
        alpha: float, optional
            Used to determine the RF percentiles. The default is 0.05.

        Returns
        -------
        numpy array with median, minimum, and maximum of percentiles across bootstrap repetitions

        """
        # Use number of unique chemicals as sample size
        sample_size = len(np.unique(RF_array[0]))
        logger.info("Length of RF_array chems at top of RFbnf = {}".format(sample_size))
        # Set seed for bootstrap random resampling
        np.random.seed(seed)
        chem_num_sampled = np.empty(sample_size * reps, dtype=np.uint64)
        RFs_sampled = np.empty(sample_size * reps, dtype=np.float64)
        for idx in np.ndindex(sample_size * reps):
            chem_num_sampled[idx] = np.random.randint(sample_size)
            RFs_to_sample = RF_array[1][RF_array[0] == chem_num_sampled[idx]]
            curr_RFs_sampled = np.random.choice(RFs_to_sample)
            RFs_sampled[idx] = curr_RFs_sampled
        RFs_sampled_split = np.split(RFs_sampled, reps)
        alpha_perc = alpha * 100
        percentiles = np.array([np.float64(alpha_perc / 2), np.float64(50), np.float64(100 - (alpha_perc / 2))])
        quant_lower_per_sample = np.empty(reps, dtype=np.float64)
        quant_median_per_sample = np.empty(reps, dtype=np.float64)
        quant_upper_per_sample = np.empty(reps, dtype=np.float64)
        k = 0
        for i in RFs_sampled_split:
            quant_lower_per_sample[k] = np.percentile(i, percentiles[0])
            quant_median_per_sample[k] = np.percentile(i, percentiles[1])
            quant_upper_per_sample[k] = np.percentile(i, percentiles[2])
            k = k + 1
        quantile_per_sample = np.concatenate(
            (quant_lower_per_sample, quant_median_per_sample, quant_upper_per_sample)
        ).reshape((3, reps))
        # Get the medians for each quantile across resamples
        logger.info(f"quantile_per_sample[0] size = {quantile_per_sample[0].size}")
        logger.info(f"quantile_per_sample[1] size = {quantile_per_sample[1].size}")
        logger.info(f"quantile_per_sample[2] size = {quantile_per_sample[2].size}")
        RF_quantiles = np.array(
            [np.median(quantile_per_sample[0]), np.median(quantile_per_sample[1]), np.median(quantile_per_sample[2])]
        )
        # Save minimum and maximum across bootstrap replicates in addition to median
        RF_rep_min = np.array(
            [np.min(quantile_per_sample[0]), np.min(quantile_per_sample[1]), np.min(quantile_per_sample[2])]
        )
        RF_rep_max = np.array(
            [np.max(quantile_per_sample[0]), np.max(quantile_per_sample[1]), np.max(quantile_per_sample[2])]
        )
        return np.concatenate((RF_rep_min, RF_quantiles, RF_rep_max)).reshape((3, 3))

    def RF_boot_estimate(
        self,
        long_nz,
        occ,
        seed=1,
        reps=10000,
        alpha=0.05,
        rep_range=True,
        long_form=True,
    ):
        """
        Performs qNTA concentration estimation on occurrence_data using RF bootstrap percentiles

        Parameters
        ----------
        RF_data : pandas DataFrame
            DataFrame containing "Chemical_Name" and "RF" columns for qNTA surrogates
        occurrence_data : pandas DataFrame
            DataFrame containing "Chemincal Name" column and columns with BlankSub Mean abundances (named using "BlankSub Mean {Sample}")
        seed : int, optional
            Seed used for the random bootstrap sampling (np.random.choice()). The default is 1.
        reps : int, optional
            Number of bootstrap repetitions. The default is 10000.
        alpha : float, optional
            Alpha for the confidence level, determines the RF percentiles used. The default is 0.05.
        rep_range : Boolean, optional
            Provide minimum and maximum for each RF percentile across bootstrap repetitions, in addition to median. The default is True.
        long_form : Boolean, optional
            Return long form DataFrame (columns for "ConcLCL","ConcEst","ConcUCL", rows are unique chemical-sample combinations)

        Returns
        -------
        RF_estimate_out : pandas DataFrame
            If long_form, contains "Chemical_Name", "Sample", "ConcLCL", "ConcEst", "ConcEst" columns
            Else, contains "Chemical_Name" column and "{Sample}_ConcLCL","{Sample}_ConcEst", "{Sample}_UCL" for all samples

        """
        # Get required attributes
        RF_estimate_out = occ.copy()
        RF_data = long_nz.copy()
        logger.info("RF_estimate_out (occ) size = {}".format(len(RF_estimate_out)))
        logger.info("RF_data (long_nz) size = {}".format(len(RF_data)))
        # Get bootstrap percentile estimates
        RF_array = self.make_RF_array(RF_data)
        RF_percs = self.RF_bootstrap_numba_full(RF_array, seed, reps, alpha)
        if long_form:
            # Change data to long form
            RF_estimate_out = pd.melt(
                RF_estimate_out,
                id_vars=["Feature ID"],
                value_vars=RF_estimate_out.columns[RF_estimate_out.columns.str.startswith("BlankSub Mean ")].tolist(),
                var_name="Sample",
                value_name="BlankSub Mean",
            )
            # Remove "BlankSub Mean " from sample names
            RF_estimate_out["Sample"] = [i[14:] for i in RF_estimate_out["Sample"]]
            # Divide BlankSub Mean abundance by RF percentiles to get concentration estimates
            # Account for data shape of RF_estimate_out (if minimum and maximum of percentiles estimates across repetitions are present)
            if rep_range:
                RF_estimate_out["ConcLCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[1][2]
                RF_estimate_out["ConcEst"] = RF_estimate_out["BlankSub Mean"] / RF_percs[1][1]
                RF_estimate_out["ConcUCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[1][0]
            else:
                RF_estimate_out["ConcLCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[2]
                RF_estimate_out["ConcEst"] = RF_estimate_out["BlankSub Mean"] / RF_percs[1]
                RF_estimate_out["ConcUCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[0]
        else:
            abun_cols = RF_estimate_out.columns[RF_estimate_out.columns.str.startswith("BlankSub Mean ")].tolist()
            # Remove "BlankSub Mean" from sample names used for making concentration column names
            conc_col_names = [i[14:] for i in abun_cols]
            # Divide BlankSub Mean abundance by RF percentiles to get concentration estimates
            # Account for data shape of RF_estimate_out (if minimum and maximum of percentiles estimates across repetitions are present)
            if rep_range:
                RF_estimate_out[[f"{i}_ConcLCL" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[1][2]
                )
                RF_estimate_out[[f"{i}_ConcEst" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[1][1]
                )
                RF_estimate_out[[f"{i}_ConcUCL" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[1][0]
                )
            else:
                RF_estimate_out[[f"{i}_ConcLCL" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[2]
                )
                RF_estimate_out[[f"{i}_ConcEst" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[0]
                )
                RF_estimate_out[[f"{i}_ConcUCL" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[1]
                )
        # Account for data shape of RF_estimate_out in adding median RF percentiles as columns
        if rep_range:
            RF_estimate_out[["RF0.025", "RF0.5", "RF0.975"]] = [RF_percs[1][0], RF_percs[1][1], RF_percs[1][2]]
        else:
            RF_estimate_out[["RF0.025", "RF0.5", "RF0.975"]] = [RF_percs[0], RF_percs[1], RF_percs[2]]
        if long_form:
            RF_estimate_out = RF_estimate_out.loc[
                :, ["Feature ID", "Sample", "RF0.025", "RF0.5", "RF0.975", "ConcLCL", "ConcEst", "ConcUCL"]
            ]
            return RF_estimate_out
        else:
            # Reorder columns so that samples are grouped together
            column_order = [i + j for i in conc_col_names for j in ["_ConcLCL", "_ConcEst", "_ConcUCL"]]
            RF_estimate_out = RF_estimate_out.loc[:, ["Feature ID"] + ["RF0.025", "RF0.5", "RF0.975"] + column_order]
            return RF_estimate_out

    # Remove LOO argument? (If always including LOO columns). Consider also other forms of cross validation (k-fold)
    def RF_boot_validation(
        self,
        seed=1,
        reps=10000,
        alpha=0.05,
        rep_range=True,
        long_form=True,
        LOO=True,
        internal=False,
    ):
        """
        Method to make qNTA concentration estimates with confidence intervals and calculate performance metrics for accuracy (AQ, AAQ) and uncertainty (CLFR)

        Parameters
        ----------
        seed : int, optional
            Seed used for the random bootstrap sampling (np.random.choice()). The default is 1.
        reps : int, optional
            Number of bootstrap repetitions. The default is 10000.
        alpha : float, optional
            Alpha for the confidence level, determines the RF percentiles used. The default is 0.05.
        rep_range : Boolean, optional
            Provides minimum and maximum for each RF percentile across bootstrap repetitions, in addition to median. The default is True.
        long_form : Boolean, optional
            Return long form DataFrame (columns for "ConcLCL","ConcEst","ConcUCL", rows are unique chemical-sample combinations)
        LOO : Boolean, optional
            Include estimates made from leave-one-out (LOO) bootstrap percentiles, where the feature is excluded from its surrogate set. The default is True.
        internal : Boolean, optional
            Use internal qNTA surrogate calibration data to perform validation in a leave-one-out (LOO) manner. The default is False.

        Returns
        -------
        validation_out : pandas DataFrame
            DataFrame containing qNTA concentration estimates with calculated performance metrics (AQ, AAQ, CLFR) based on the validation data (internal or external)
        """
        # Get required attributes
        val = self.validation_data
        long_nz = self.surrogate_cal_data_long_nonzero.reset_index()
        chems = self.surrogate_cal_data_long_nonzero_chems
        occ = self.occurrence_data
        # Columns containing the targeted concentrations for validation
        prefixes = [
            "Feature",
            "Retention",
            "Ionization",
            "Mass",
            "m/z",
        ]
        # Get concentration columns
        conc_cols = [col for col in val.select_dtypes(include=np.number).columns if not any(x in col for x in prefixes)]
        # Calculate global bootstrap RF percentiles and use to make concentration estimates
        # NOTE: For internal, occurrence_data must contain columns with names that correspond to conc_cols
        global_out = self.RF_estimate_out
        # List of chemicals that overlap between qNTA surrogate set and validation data
        LOO_IDs = pd.Series(val["Feature ID"].values, index=val["Chemical Name"]).to_dict()
        logger.info("length LOO_IDs = {}".format(len(LOO_IDs)))
        # LOO_chems = {value: key for key, value in LOO_IDs.items()}
        LOO_IDs = [int(ID) for chem, ID in LOO_IDs.items() if not pd.isna(ID) and any(x in chem for x in chems)]
        # If chemicals overlap between qNTA surrogates and validation data and LOO is True
        if len(LOO_IDs) > 0 and LOO:
            # Get LOO RF bootstrap percentiles and concentration estimates
            logger.info(f"LOO_IDs: {LOO_IDs}")
            LOO_out = pd.concat(
                [
                    self.RF_boot_estimate(
                        long_nz[long_nz["Feature ID"] != i],
                        occ[occ["Feature ID"] == i],
                        seed,
                        reps,
                        alpha,
                        rep_range,
                        long_form,
                    )
                    for i in LOO_IDs
                ]
            )
            if long_form:
                # Change validation_data to long form
                # This is in order of sample and not chemical
                val = pd.melt(
                    val, id_vars="Feature ID", value_vars=conc_cols, var_name="Sample", value_name="ConcTargeted"
                )
                # Add _LOO suffix to column names (to distinguish LOO columns when
                # adding to global estimates DataFrame)
                LOO_out = LOO_out.rename(
                    columns={
                        c: c + "_LOO"
                        for c in LOO_out.columns
                        if not any(x in c for x in ["Feature", "Chemical", "Sample"])
                    }
                )
                # Ensure that correct ConcTargeted and ConcLCL, Est, UCL are compared
                LOO_out = pd.merge(LOO_out, val, on=["Feature ID", "Sample"], how="left")
                # Calculate qNTA performance metrics for accuracy and uncertainty
                LOO_out["AQ_LOO"] = LOO_out["ConcEst_LOO"] / LOO_out["ConcTargeted"]
                LOO_out["AAQ_LOO"] = 10 ** np.abs(np.log10(LOO_out["AQ_LOO"]))
                LOO_out["CLFR_LOO"] = LOO_out["ConcUCL_LOO"] / LOO_out["ConcLCL_LOO"]
                LOO_out = LOO_out.drop(columns=["ConcTargeted"])  # ConcTargeted will be merged again later
                # Left outer join keeps a row for all chemicals in validation data, with np.NaN (pd.NA?) for qNTA columns if not in global_out
                validation_out = pd.merge(global_out, val, on=["Feature ID", "Sample"], how="left")
                validation_out["AQ"] = validation_out["ConcEst"] / validation_out["ConcTargeted"]
                validation_out["AAQ"] = 10 ** np.abs(np.log10(validation_out["AQ"]))
                validation_out["CLFR"] = validation_out["ConcUCL"] / validation_out["ConcLCL"]
                # Merge on LOO_out
                validation_out = pd.merge(validation_out, LOO_out, on=["Feature ID", "Sample"], how="left")
                # Remove NaN rows from the ConcTargeted (the validation file) and ConcEst (occurrence file)
                validation_out = validation_out.loc[
                    ((validation_out["ConcTargeted"] > 0) & (validation_out["ConcEst"] > 0)), :
                ]
                # Return validation_out
                return validation_out
            else:
                # Get validation out by merging global_out and val
                validation_out = pd.merge(global_out, val, on="Feature ID", how="inner")
                # Add _LOO suffix to column names (to distinguish LOO columns when adding to global estimates DataFrame)
                LOO_out = LOO_out.rename(
                    columns={
                        c: c + "_LOO"
                        for c in LOO_out.columns
                        if not any(x in c for x in ["Feature", "Chemical", "Sample"])
                    }
                )
                # Rename columns prior to merge so that columns match
                if internal:
                    conc_cols = [c[5:] for c in conc_cols]
                # Add val data back on to LOO_out
                cols = ["Feature ID"] + conc_cols
                LOO_out = pd.merge(LOO_out, val[cols], on="Feature ID", how="left")
                # Iterate through conc_cols, calculate AQ, AAQ, and CLFR
                for i in conc_cols:
                    LOO_out[f"{i}_AQ_LOO"] = LOO_out.loc[:, f"{i}_ConcEst_LOO"] / LOO_out.loc[:, i]
                    LOO_out[f"{i}_AAQ_LOO"] = 10 ** np.abs(np.log10(LOO_out.loc[:, f"{i}_AQ_LOO"]))
                    LOO_out[f"{i}_CLFR_LOO"] = LOO_out.loc[:, f"{i}_ConcUCL_LOO"] / LOO_out.loc[:, f"{i}_ConcLCL_LOO"]
                # Replace Inf AAQ and NaN CLFR values?
                # Drop these columns, they are added when merging to global_out
                LOO_out = LOO_out.drop(columns=conc_cols)
                # Merge on LOO_out
                validation_out = pd.merge(validation_out, LOO_out, on="Feature ID", how="left")
                # Return validation_out
                return validation_out
        else:
            return None

    @staticmethod
    def validation_summary(
        validation_out,
        long_form=True,
        LOO=True,
    ):
        """
        Method to provide summary metrics (median) for qNTA performance metrics
        (AQ, AAQ, CLFR) and calculate reliability (ORP).

        Parameters
        ----------
        validation_out : pandas DataFrame
            DataFrame produced by method RF_boot_validation, containing columns for
            qNTA concentration estimates and qNTA performance metrics
        long_form : Boolean, optional
            DESCRIPTION. Specifies whether validation_out is long form (one column
            per performance metric) or wide form (one column per sample, per performance metric).
            The default is False.
        LOO : Boolean, optional
            DESCRIPTION. Specifies whether validation_out contains LOO columns
            (produced from leave-one-out qNTA bootstrap estimation). The default is True.

        Returns
        -------
        summary_out : pandas DataFrame
            DataFrame containing minimum, median, and maximum of qNTA performance
            metrics AQ, AAQ, CLFR and calculated reliability (ORP) for validation_out.

        """
        # Output summary statistics (minimum, median, and maximum for AQ, AAQ, and CLFR, and overall ORP (and per-chemical, per-sample ORP?))
        if long_form:
            # Limit to non-zero ConcEst to ensure AAQ is not Inf and CLFR is not NaN
            validation_out = validation_out[validation_out["ConcEst"] > 0]
            if LOO:
                summary_out = validation_out.loc[:, ["AQ", "AAQ", "CLFR", "AQ_LOO", "AAQ_LOO", "CLFR_LOO"]].agg(
                    ["min", "median", "max"]
                )
                summary_out["ORP"] = (
                    validation_out.loc[
                        (validation_out["ConcTargeted"] <= validation_out["ConcUCL"])
                        & (validation_out["ConcTargeted"] >= validation_out["ConcLCL"])
                    ].shape[0]
                    / validation_out.shape[0]
                )
                summary_out["ORP_LOO"] = (
                    validation_out.loc[
                        (validation_out["ConcTargeted"] <= validation_out["ConcUCL_LOO"])
                        & (validation_out["ConcTargeted"] >= validation_out["ConcLCL_LOO"])
                    ].shape[0]
                    / validation_out.shape[0]
                )
                return summary_out
            else:
                summary_out = validation_out.loc[:, ["AQ", "AAQ", "CLFR"]].agg(["min", "median", "max"])
                summary_out["ORP"] = (
                    validation_out.loc[
                        (validation_out["ConcTargeted"] <= validation_out["ConcUCL"])
                        & (validation_out["ConcTargeted"] >= validation_out["ConcLCL"])
                    ].shape[0]
                    / validation_out.shape[0]
                )
                return summary_out
        else:
            # Change from wide to long form (for performance metric columns)
            validation_long = pd.melt(
                validation_out,
                id_vars="Chemical Name",
                value_vars=validation_out.columns[validation_out.columns.str.contains("AQ|CLFR")].tolist(),
                var_name="SampleMetric",
                value_name="Metric",
            )
            validation_long[["Sample", "MetricName"]] = validation_long["SampleMetric"].str.split("__", expand=True)
            validation_long = validation_long.drop(columns="SampleMetric").pivot(
                index=["Chemical Name", "Sample"], columns="MetricName"
            )
            # Drop 'Metric' from column MultiIndex
            validation_long.columns = validation_long.columns.droplevel(0)

            """NEED TO ADD ANOTHER LOO LAYER OF LOGIC"""

            # Remove non-finite AAQ (also ensures that CLFR is not NaN)
            validation_long = validation_long[np.isfinite(validation_long["AAQ"])]
            # Change from wide to long (for concentration estimate columns)
            conc_long = pd.melt(
                validation_out,
                id_vars="Chemical Name",
                value_vars=validation_out.columns[validation_out.columns.str.contains("Conc")].tolist(),
                var_name="SampleConc",
                value_name="Conc",
            )
            conc_long[["Sample", "ConcName"]] = conc_long["SampleConc"].str.split("__", expand=True)
            conc_long = conc_long.drop(columns="SampleConc").pivot(
                index=["Chemical Name", "Sample"], columns="ConcName"
            )
            # Drop 'Conc' from column MultiIndex
            conc_long.columns = conc_long.columns.droplevel(0)
            # Join with validation_out to ensure that rows match between validation_long and conc_long when comparing
            sample_names = validation_out.columns[validation_out.columns.str.endswith("_")].tolist()
            conc_long["ConcTargeted"] = (
                pd.melt(
                    validation_out,
                    id_vars="Chemical Name",
                    value_vars=sample_names,
                    var_name="Sample",
                    value_name="ConcTargeted",
                )
                .replace(to_replace=r"_", value="", regex=True)
                .set_index(["Chemical Name", "Sample"])
            )
            # Remove ConcEst == 0 to ensure AAQ is not Inf and CLFR is not NaN
            conc_long = conc_long[conc_long["ConcEst"] > 0]
            if LOO:
                summary_out = validation_long.loc[:, ["AQ", "AAQ", "CLFR", "AQ_LOO", "AAQ_LOO", "CLFR_LOO"]].agg(
                    ["min", "median", "max"]
                )
                summary_out["ORP"] = (
                    conc_long.loc[
                        (conc_long["ConcTargeted"] <= conc_long["ConcUCL"])
                        & (conc_long["ConcTargeted"] >= conc_long["ConcLCL"])
                    ].shape[0]
                    / conc_long.shape[0]
                )
                summary_out["ORP_LOO"] = (
                    conc_long.loc[
                        (conc_long["ConcTargeted"] <= conc_long["ConcUCL_LOO"])
                        & (conc_long["ConcTargeted"] >= conc_long["ConcLCL_LOO"])
                    ].shape[0]
                    / conc_long.shape[0]
                )
            else:
                summary_out = validation_long.loc[:, ["AQ", "AAQ", "CLFR"]].agg(["min", "median", "max"])
                summary_out["ORP"] = (
                    conc_long.loc[
                        (conc_long["ConcTargeted"] <= conc_long["ConcUCL"])
                        & (conc_long["ConcTargeted"] >= conc_long["ConcLCL"])
                    ].shape[0]
                    / conc_long.shape[0]
                )
            return summary_out  # Note: order of columns is different here (AAQ, AQ instead of AQ, AAQ)
