import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

from matplotlib import pyplot as plt
import seaborn as sns
import re

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
        self.validation_out = None
        # Plot holders
        self.all_cal_models = None
        self.all_cal_plots = None
        self.cc_metrics = None
        self.bootstrap_array = None

    def execute(self):
        """Perform data manipulation functions"""
        self.check_parameters()
        self.check_occurrences()
        self.check_RF_input()
        self.check_validation_data()
        """Perform Calibration Curve Methods"""
        self.cal_curve_all_metrics()
        """Perform Bootstrap Methods"""
        # Calculate Response factor percentiles
        self.RF_percs = pd.DataFrame(
            self.RF_bootstrap(
                self.surrogate_cal_data_long_nonzero,
                seed=self.parameters["seed"],
                reps=self.parameters["reps"],
                alpha=self.parameters["alpha"],
                rep_range=self.parameters["rep_range"],
            ),
            index=pd.Index(self.percentiles, name="Response Factor Percentile Estimate"),
            columns=["Minimum", "Median", "Maximum"],
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

    """DATA MANIPULATION FUNCTIONS"""

    def check_parameters(self):
        # Check if parameters are selected by user
        if self.parameters is not None:
            # Calculate percentiles from alpha parameter
            self.percentiles = np.multiply([self.parameters["alpha"] / 2, 0.5, 1 - (self.parameters["alpha"] / 2)], 100)
        else:
            # Set parameters to default
            self.parameters = {
                "seed": 1,
                "reps": 10000,
                "alpha": 0.05,
                "rep_range": True,
                "long_form": True,
                "LOO": True,
                "internal": False,
            }
            # Calculate percentiles from alpha parameter
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
            # Identify rows with any zero
            rows_with_zero = (occ[back] == 0).any(axis=1)
            # Create subset limited to chemicals with ONLY non-zero occurrences, store
            self.occurrences_data_nonzero = occ[~rows_with_zero]
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
            # Identify rows with any zero
            rows_with_zero = (occ[back] == 0).any(axis=1)
            # Create subset limited to chemicals with ONLY non-zero occurrences, store
            self.occurrences_data_nonzero = occ[~rows_with_zero]

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
        # Get cols
        prefixes = ["Mean", "STD", "CV", "Detection Count", "Detection Percentage", "BlankSub Mean", "Conc", "RF"]
        cols = ["Feature ID", "Chemical Name", "Retention Time"] + [
            col for col in surr.columns if any(col.startswith(x) for x in prefixes)
        ]
        # Pivot surrogate_cal_data wide to long
        long = pd.wide_to_long(surr[cols], stubnames=prefixes, i="Feature ID", j="Cal Level", sep=" ", suffix="\\w+")
        # Change Conc column to numeric
        long["Conc"] = pd.to_numeric(long["Conc"])
        # Keep only BlankSub Mean abundances > 0 to avoid problems with log-10 transform
        # we also don't want to have RFs of 0 in the surrogate set
        long_nz = long.query("`BlankSub Mean` > 0")
        # Add log-10 transformed columns for BlankSub Mean Abundance and Concentration
        long_nz = long_nz.assign(LogAbun=np.log10(long_nz["BlankSub Mean"]), LogConc=np.log10(long_nz["Conc"]))
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
            # Get Conc col root names
            cols = [col[5:] for col in surr.columns if "Conc " in col if not any(x in col for x in blanks)]
            # Define regex pattern, use to extract vals from Conc col names
            re_pattern = "(\d+)"
            concs = [int(re.search(re_pattern, col).group()) for col in cols]
            # Set Chemical Name as ID column for future joins
            val = surr.loc[:, ["Chemical Name", "Feature ID"]]
            # Make test validation file using qNTA_cal_data_pos
            val[cols] = concs
            # Use copy to avoid overwriting original data
            self.validation_data = val.copy()

    """CALIBRATION CURVE METHODS"""

    def fit_cal_curve_model(
        self,
        chem,
    ):
        """
        Subsets qNTA surrogate calibration data (long form) to data from the chemical
        provided as input ('chem').
        Uses the atrribute 'surrogate_cal_data_long_nonzero'.
        Stores and returns a statsmodels object sm.OLS (ordinary least squares linear
        regression model) calibration curve (LogAbun vs LogConc).

        Parameters
        ----------
        chem : str
            Name of qNTA surrogate chemical used for calibration curve model

        Returns
        -------
        If the subset qNTA surrogate calibration data has fewer than three points,
        returns string with error message
        Else, return tuple with calibration curve model (statsmodels object) and
        qNTA surrogate chemical name

        """
        # Copy df
        surr = self.surrogate_cal_data_long_nonzero.copy()
        # Subset by chem
        cal_data = surr.loc[surr["Chemical Name"] == chem]
        # Check if there are more than 3 points
        if len(cal_data) < 3:
            # If no, return string
            return "Fewer than 3 calibration points"
        else:
            # If yes, generate calibration model object
            cal_model = sm.OLS(cal_data["LogAbun"], sm.add_constant(cal_data["LogConc"]))
            # Return tuple
            return (cal_model, chem)

    @staticmethod
    def plot_cal_curve(cal_model_named, storefig=False, savefig=False):
        """
        Parameters
        ----------
        cal_model_named : tuples
            Tuple with ordinary least squares linear model of log10-transformed BlankSub Mean abundance (LogAbun)
            and log10-transformed concentration (LogConc) and the str of the qNTA surrogate name
        storefig : Boolean, optional
            Store matplotlib figure and qNTA surrogate name as tuple. The default is False.
        savefig : Boolean, optional
            Save fig as .png file. The default is True.

        Returns
        -------
        If storefig, then return tuple with matplotlib figure and qNTA surrogate chemical name
        Else, None
        """
        # Separate out cal_model_named to the statsmodel OLS object and the qNTA surrogate chemical name
        cal_model = cal_model_named[0]
        chem = cal_model_named[1]
        # Use the calibration curve model to get fitted LogAbun values
        cal_model_results = cal_model.fit()
        cal_model_preds = cal_model_results.get_prediction()
        # Get lower and upper confidence limits for calibration curve
        cal_model_CI_lower = cal_model_preds.summary_frame()["obs_ci_lower"]
        cal_model_CI_upper = cal_model_preds.summary_frame()["obs_ci_upper"]
        # Create matplotlib pyplot of calibration curve with data points, fitted regression line, and  confidence intervals
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(cal_model.exog[:, 1], cal_model.endog, "o", label="Data")
        ax.plot(cal_model.exog[:, 1], cal_model_results.fittedvalues, "b-", label="Fit")
        ax.plot(cal_model.exog[:, 1], cal_model_CI_lower, "r--")
        ax.plot(cal_model.exog[:, 1], cal_model_CI_upper, "r--")
        ax.legend(loc="best")
        # Extract model parameters (slope and R-squared) to add to calibration curve figure as subtitle
        cal_model_params = cal_model_results.params.round(3)
        cal_model_equation = "LogAbun = " + str(cal_model_params["LogConc"]) + "LogConc"
        if "const" in cal_model_params.index:
            cal_model_equation = (
                "LogAbun = " + str(cal_model_params["const"]) + " + " + str(cal_model_params["LogConc"]) + "LogConc"
            )
        # Add qNTA surrogate chemical name as title and model equation and R-squared below title
        fig.suptitle(chem + " \n " + cal_model_equation + ", R-squared: " + str(cal_model_results.rsquared.round(3)))
        # Store chem, slope, and R2 in tuple
        cc_metrics = (chem, cal_model_params["LogConc"], cal_model_results.rsquared.round(3))
        if storefig:
            # Return fig/chem tuple, and cc_metrics tuple
            return (fig, chem), cc_metrics
        elif savefig:
            # Plot fig, return cc_metrics tuple
            plt.savefig(chem + "_Cal_Curve.png")
            return cc_metrics
        else:
            return cc_metrics

    def cal_curve_all_metrics(
        self,
    ):
        """
        Calculate calibration curve metrics from plots for all qNTA surrogate chemicals
        in the qNTA surrogate statistics DataFrame

        Parameters
        ----------
        storefig : Boolean, optional
            Store figures as attribute all_cal_plots, a list of tuples containing
                the matplotlib figure and the chemical name. The default is False.
        savefig : Boolean, optional
            Save the figure as a .png file. The default is True.

        Returns
        -------
        None.

        """
        # Call fit_cal_curve_model() on all unique chems in surrogate_cal_data_long_nonzero_chems
        self.all_cal_models = [self.fit_cal_curve_model(i) for i in self.surrogate_cal_data_long_nonzero_chems]
        # Create list of tuples from plot_cal_curve() on all items in all_call_models
        cc_tuples = [self.plot_cal_curve(i) for i in self.all_cal_models if "Fewer than 3 calibration points" not in i]
        # Generate and save dataframe
        self.cc_metrics = pd.DataFrame(cc_tuples, columns=["Chemical Name", "Slope", "R-squared"])

    def cal_curve_all(
        self,
        storefig=True,
        savefig=False,
    ):
        """
        Create calibration curve models and plots for all qNTA surrogate chemicals
        in the qNTA surrogate statistics DataFrame

        Parameters
        ----------
        storefig : Boolean, optional
            Store figures as attribute all_cal_plots, a list of tuples containing
                the matplotlib figure and the chemical name. The default is False.
        savefig : Boolean, optional
            Save the figure as a .png file. The default is True.

        Returns
        -------
        None.

        """
        # Call fit_cal_curve_model() on all unique chems in surrogate_cal_data_long_nonzero_chems
        self.all_cal_models = [self.fit_cal_curve_model(i) for i in self.surrogate_cal_data_long_nonzero_chems]
        # Check if storefig
        if storefig:
            # If yes, create list of tuples from plot_cal_curve() on all items in all_call_models
            cc_tuples = [
                self.plot_cal_curve(i, storefig, savefig)
                for i in self.all_cal_models
                if "Fewer than 3 calibration points" not in i
            ]
            # Get plots tuple from tuple
            self.all_cal_plots = [tup[0] for tup in cc_tuples]
            # Get metrics tuple from tuple
            cc_metrics = [tup[1] for tup in cc_tuples]
            self.cc_metrics = pd.DataFrame(cc_metrics, columns=["Chemical Name", "Slope", "R-squared"])
        else:
            # If no, call plot_cal_curve() on all items in all_call_models
            for i in self.all_cal_models:
                self.plot_cal_curve(i, storefig, savefig)

    """RESPONSE FACTOR BOOTSTRAP METHODS"""

    @staticmethod
    def RF_bootstrap(
        RF_data,
        seed=1,
        reps=10000,
        alpha=0.05,
        rep_range=True,
    ):
        """
        Performs hierarchical response factor bootstrap (choosing one chemical, then one of its RFs)

        Parameters
        ----------
        RF_data : pandas DataFrame
            DataFrame containing "Chemical Name" and "RF" columns
        seed : int, optional
            Seed used for the random bootstrap sampling (np.random.choice()). The default is 1.
        reps : int, optional
            Number of bootstrap repetitions. The default is 10000.
        alpha : float, optional
            Alpha value for confidence level, determines percentiles used. The default is 0.05.
        rep_range : Boolean, optional
            Output minimum and maximum across bootstrap repetitions for each percentile. The default is True.

        Returns
        -------
        If rep_range, numpy array with median, minimum, and maximum of percentiles across bootstrap repetitions
        Else, numpy array with median of percentiles across bootstrap replicates

        """
        # Get self.surrogate_cal_data_long_nonzero
        df = RF_data.copy()
        # Get unique chems from RF_data
        chems = pd.unique(df["Chemical Name"])
        # Set sample size of bootstrap resampling to number of unique chemicals in surrogate data (allow user to customize? Should always default to len(chems))
        sample_size = len(chems)
        # Store in a list each surrogate chemical's RFs in a separate list
        chem_RFs_list = [df[df["Chemical Name"] == i]["RF"].tolist() for i in chems]
        # Add lists of RFs to dictionary
        chem_RFs_list_dict = {}
        for chem, vals in zip(chems, chem_RFs_list):
            chem_RFs_list_dict[chem] = vals
        # Store number of RFs for each chemical for easy random sampling
        dict_len = [len(value) for key, value in chem_RFs_list_dict.items()]
        # Set seed for bootstrap random resampling
        np.random.seed(seed)
        # Sample a chemical's index from list
        chem_num_sampled = np.random.choice(range(len(chems)), size=sample_size * reps, replace=True)
        chems_sampled = [chems[i] for i in chem_num_sampled]
        # Resample random index from within range of each chemical's RFs
        RF_indices_sampled = [np.random.choice(range(dict_len[i])) for i in chem_num_sampled]
        # Get RF from the randomly sampled indices from chem_num_sampled
        RFs_sampled_by_index = [chem_RFs_list_dict.get(i)[j] for i, j in zip(chems_sampled, RF_indices_sampled)]
        RFs_sampled_by_index = np.split(np.array(RFs_sampled_by_index), reps)
        # Percentiles for RF bootstrap
        percentiles = np.multiply([alpha / 2, 0.5, 1 - (alpha / 2)], 100)
        # Calculate quantiles per sample
        quantile_per_sample = [np.percentile(i, percentiles) for i in RFs_sampled_by_index]
        # Transpose to change from list to np.array, with columns as resamples and rows as percentiles, to facilitate calculations
        quantiles_overall = np.transpose(quantile_per_sample)
        # Get the medians for each quantile across resamples
        RF_quantiles = [
            np.median(quantiles_overall[0]),
            np.median(quantiles_overall[1]),
            np.median(quantiles_overall[2]),
        ]
        # Save minimum and maximum across bootstrap replicates in addition to median
        if rep_range:
            RF_rep_min = [np.min(quantiles_overall[0]), np.min(quantiles_overall[1]), np.min(quantiles_overall[2])]
            RF_rep_max = [np.max(quantiles_overall[0]), np.max(quantiles_overall[1]), np.max(quantiles_overall[2])]
            return np.array([RF_rep_min, RF_quantiles, RF_rep_max])
        else:
            return np.array(RF_quantiles)

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
        # Get bootstrap percentile estimates
        RF_percs = self.RF_bootstrap(RF_data, seed, reps, alpha, rep_range)
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
        long_form=False,
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
        # LOO_chems = {value: key for key, value in LOO_IDs.items()}
        LOO_IDs = [int(ID) for chem, ID in LOO_IDs.items() if not pd.isna(ID) and any(x in chem for x in chems)]
        # If chemicals overlap between qNTA surrogates and validation data and LOO is True
        if len(LOO_IDs) > 0 and LOO:
            # Get LOO RF bootstrap percentiles and concentration estimates
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
                # Remove "Conc" from sample names coming from internal calibration data,
                # needed so that Sample columns match when merging
                if internal:
                    val["Sample"] = [s[5:] for s in val["Sample"]]
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
                # Remove NaN rows
                validation_out = validation_out.loc[~validation_out["ConcEst"].isna(), :]
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
