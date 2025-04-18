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
    def __init__(
        self,
        surrogate_cal_data,
        validation_input=None,
        occurrence_input=None,
    ):
        """
        Pivots surrogate_cal_data DataFrame from wide to long format, keeping only
            blank-subtracted means > 0
        Adds LogAbun and LogCon columns (log10-transformed)
        Stores unique surrogate chemicals in surrogate_cal_data_long_nonzero as
            surrogate_cal_data_long_nonzero_chems
        """
        # Required uploaded files
        # Replace NA values with 0 to avoid issues with mathematical operations on DataFrame
        self.surrogate_cal_data = surrogate_cal_data.fillna(0)
        self.validation_data = validation_input
        self.occurrence_data = occurrence_input
        # Dataframes to be calculated
        self.surrogate_cal_data_long_nonzero = None
        self.surrogate_cal_data_long_nonzero_chems = None
        self.occurrences_data_nonzero = None
        self.RF_estimate_out = None
        self.RF_percs = None
        # Plot holders
        self.all_cal_models = None
        self.all_cal_plots = None
        self.cc_metrics = None
        self.bootstrap_array = None

    def execute(self):
        """Perform data manipulation functions"""
        self.check_occurrences()
        self.check_RF_input()
        self.check_validation_data()
        """Perform Calibration Curve Methods"""
        self.cal_curve_all()
        # """Perform Bootstrap Methods"""
        # self.RF_percs = self.RF_bootstrap(self.surrogate_cal_data_long_nonzero)
        # self.RF_estimate_out = self.RF_boot_estimate(
        #     self.surrogate_cal_data_long_nonzero,
        #     self.occurrence_data,
        # )

        # self.RF_boot_validation()

    """DATA MANIPULATION FUNCTIONS"""

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
            logger.info("occ is not None")
            # Get cols (only take columns also in val; e.g., no Pool)
            front = [col for col in occ.columns if any(x in col for x in ["Feature", "Chemical", "Retention"])]
            if val is not None:
                logger.info("val is not None")
                back = [
                    col for col in occ.columns if col.startswith("BlankSub Mean") and any(x in col for x in val.columns)
                ]
            else:
                logger.info("val is None")
                back = [col for col in occ.columns if col.startswith("BlankSub Mean")]
            # Pare occ down to front + back
            self.occurrence_data = occ[front + back]
            logger.info("occurrence_data columns= {}".format(self.occurrence_data.columns.values))
            # Identify rows with any zero
            rows_with_zero = (occ[back] == 0).any(axis=1)
            # Create subset limited to chemicals with ONLY non-zero occurrences, store
            self.occurrences_data_nonzero = occ[~rows_with_zero]
        else:
            logger.info("occ is None")
            # Copy input
            surr = self.surrogate_cal_data.copy()
            # Get cols
            front = ["Feature ID", "Chemical Name", "Retention Time"]
            back = [col for col in surr.columns if col.startswith("BlankSub Mean")]
            cols = front + back
            # Subset columns from self.surrogate_cal_data, store
            occ = surr[cols]
            logger.info("occurrence_data columns= {}".format(self.occurrence_data.columns.values))
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
        logger.info("long_nz columns= {}".format(long_nz.columns.values))
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
            logger.info("val is not None")
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
            logger.info("val is None")
            # Define blanks, they may still be in column if no val
            logger.info("surr columns= {}".format(surr.columns.values))
            blanks = ["Blank", "blank", "BLANK", "MB", "Mb", "mb", "mB"]
            # Get Conc col root names
            cols = [col[5:] for col in surr.columns if "Conc " in col if not any(x in col for x in blanks)]
            logger.info("list comp columns= {}".format(cols))
            # Define regex pattern, use to extract vals from Conc col names
            re_pattern = "(\d+)"
            concs = [int(re.search(re_pattern, col).group()) for col in cols]
            # Set Chemical Name as ID column for future joins
            val = surr.loc[:, ["Chemical Name"]]
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
    def plot_cal_curve(cal_model_named, storefig=False, savefig=True):
        """
        Parameters
        ----------
        cal_model_named : tuples
            Tuple with ordinary least squares linear model of log10-transformed BlankSub Mean abundance (LogAbun) and log10-transformed concentration (LogConc) and the str of the qNTA surrogate name
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
        if savefig:
            # Plot fig, return cc_metrics tuple
            plt.savefig(chem + "_Cal_Curve.png")
            return cc_metrics

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
