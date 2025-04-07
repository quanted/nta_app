import pandas as pd
import numpy as np
import statsmodels.api as sm

from matplotlib import pyplot as plt
import seaborn as sns


# %%
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

    # This uses an intermediate output from INTERPRET NTA, which first takes in the Detection Matrix input file and qNTA surrogate input file
    # Pass in Ionization Mode as a str argument or determine it from the Ionization Mode column? (Currently separating the input data beforehand)
    def __init__(self, surrogate_cal_data):
        """
        Pivots surrogate_cal_data DataFrame from wide to long format, keeping only blank-subtracted means > 0
        Adds LogAbun and LogCon columns (log10-transformed)
        Stores unique surrogate chemicals in surrogate_cal_data_long_nonzero as surrogate_cal_data_long_nonzero_chems

        """
        # Replace NA values with 0 to avoid issues with mathematical operations on DataFrame
        self.surrogate_cal_data = surrogate_cal_data.fillna(
            0
        )  # Use np.ma.masked_invalid instead? Excludes NaNs and infs
        # Pivot data from wide to long format to facilitate mathematical operations
        self.surrogate_cal_data_long = pd.wide_to_long(
            self.surrogate_cal_data.loc[
                :,
                ["Feature ID", "Chemical Name", "Retention Time"]
                + self.surrogate_cal_data.columns[self.surrogate_cal_data.columns.str.endswith("_")].tolist(),
            ],
            stubnames=["Mean", "STD", "CV", "Detection Count", "Detection Percentage", "BlankSub Mean", "Conc", "RF"],
            i="Feature ID",
            j="Cal Level",
            sep=" ",
            suffix="\\w+",
        )
        # Change Conc column to numeric
        self.surrogate_cal_data_long["Conc"] = pd.to_numeric(self.surrogate_cal_data_long["Conc"])
        # Keep only BlankSub Mean abundances > 0 to avoid problems with log-10 transform, we also don't want to have RFs of 0 in the surrogate set
        self.surrogate_cal_data_long_nonzero = self.surrogate_cal_data_long.query("`BlankSub Mean` > 0")
        # Add log-10 transformed columns for BlankSub Mean Abundance and Concentration
        self.surrogate_cal_data_long_nonzero = self.surrogate_cal_data_long_nonzero.assign(
            LogAbun=np.log10(self.surrogate_cal_data_long_nonzero["BlankSub Mean"]),
            LogConc=np.log10(self.surrogate_cal_data_long_nonzero["Conc"]),
        )
        self.surrogate_cal_data_long_nonzero_chems = np.unique(self.surrogate_cal_data_long_nonzero["Chemical Name"])

    ### Calibration Curve Methods ###

    def fit_cal_curve_model(self, chem):
        """
        Subsets qNTA surrogate calibration data (long form) to data from the chemical provided as input ('chem').
        Uses the atrribute 'surrogate_cal_data_long_nonzero'.
        Stores and returns a statsmodels object sm.OLS (ordinary least squares linear regression model) calibration curve (LogAbun vs LogConc).

        Parameters
        ----------
        chem : str
            Name of qNTA surrogate chemical used for calibration curve model

        Returns
        -------
        If the subset qNTA surrogate calibration data has fewer than three points, returns string with error message
        Else, return tuple with calibration curve model (statsmodels object) and qNTA surrogate chemical name

        """

        cal_data = self.surrogate_cal_data_long_nonzero.loc[
            self.surrogate_cal_data_long_nonzero["Chemical Name"] == chem
        ]
        if len(cal_data) < 3:
            return "Fewer than 3 calibration points"
        else:
            cal_model = sm.OLS(cal_data["LogAbun"], sm.add_constant(cal_data["LogConc"]))
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
        cal_model_equation = "LogAbun = " + str(cal_model_params.iloc[0]) + "LogConc"
        if len(cal_model_params > 1):
            cal_model_equation = (
                "LogAbun = " + str(cal_model_params.iloc[0]) + " + " + str(cal_model_params.iloc[1]) + "LogConc"
            )
        # Add qNTA surrogate chemical name as title and model equation and R-squared below title
        fig.suptitle(chem + " \n " + cal_model_equation + ", R-squared: " + str(cal_model_results.rsquared.round(3)))

        if storefig:
            return (fig, chem)
        if savefig:
            plt.savefig(chem + "_Cal_Curve.png")

    def cal_curve_all(self, storefig=False, savefig=True):
        """
        Create calibration curve models and plots for all qNTA surrogate chemicals in the qNTA surrogate statistics DataFrame

        Parameters
        ----------
        storefig : Boolean, optional
            Store figures as attribute all_cal_plots, a list of tuples containing the matplotlib figure and the chemical name. The default is False.
        savefig : Boolean, optional
            Save the figure as a .png file. The default is True.

        Returns
        -------
        None.

        """
        self.all_cal_models = [self.fit_cal_curve_model(i) for i in self.surrogate_cal_data_long_nonzero_chems]
        if storefig:
            self.all_cal_plots = [self.plot_cal_curve(i, storefig, savefig) for i in self.all_cal_models]
        else:
            for i in self.all_cal_models:
                self.plot_cal_curve(i, storefig, savefig)

    ### Response Factor Bootstrap Methods ###

    @staticmethod
    def RF_bootstrap(RF_data, seed=1, reps=10000, alpha=0.05, rep_range=True):
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
        # Get unique chemicals from column 'Chemical Name'
        chems = pd.unique(RF_data["Chemical Name"])
        # Set sample size of bootstrap resampling to number of unique chemicals in surrogate data (allow user to customize? Should always default to len(chems))
        sample_size = len(chems)
        # Store in a list each surrogate chemical's RFs in a separate list
        chem_RFs_list = [RF_data[RF_data["Chemical Name"] == i]["RF"].tolist() for i in chems]
        # Add lists of RFs to dictionary
        chem_RFs_list_dict = {}
        for i in range(len(chems)):
            chem_RFs_list_dict[chems[i]] = chem_RFs_list[i]
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
            return np.array([RF_quantiles, RF_rep_min, RF_rep_max])
        else:
            return np.array(RF_quantiles)

    @staticmethod
    def plot_RF_strip(RF_data, order_by="RT", savefig=True):
        """
        Reads in surrogate calibration dataframe and saves Seaborn strip plot .png file, ordered using order_by (RT, LogRF, or Chemical Name)

        Parameters
        ----------
        RF_data : pandas DataFrame
            DataFrame containing "Chemical Name" and "RF" columns
        order_by : str, optional
            Order chemicals from top to bottom by increasing retention time ('RT'), median log10-transformed response factor ('RF'), or alphabeticcaly by chemical name ('Chemical Name'). The default is 'RT'.
        savefig : Boolean, optional
            Save figure as .png file. The default is True.
        Returns
        -------
        None.

        """
        # Log-10 transform RF data for plotting
        RF_data["LogRF"] = np.log10(RF_data["RF"])
        # Order from top to bottom by Retention Time, LogRF, or Chemical Name (alphabetical)
        if order_by == "RT":
            plot_order = RF_data.groupby("Chemical Name")["Retention Time"].median().sort_values().index.values
        if order_by == "LogRF":
            plot_order = RF_data.groupby("Chemical Name")["LogRF"].median().sort_values().index.values
        if order_by == "Chemical Name":
            plot_order = RF_data.groupby("Chemical Name")["Chemical Name"].unique().sort_values().index.values
        fig = plt.figure(figsize=(10, 16))  # Y-axis labels not shown fully
        sns.stripplot(
            data=RF_data, x="LogRF", y="Chemical Name", hue="Chemical Name", legend=False, jitter=True, order=plot_order
        )
        plt.tight_layout()
        if savefig:
            fig.savefig("RF_strip_plot.png")

    @staticmethod
    def plot_RF_hist(RF_data, RF_boot, num_bins=30, savefig=True):
        """
        Reads in surrogate calibration data and bootstrap percentiles (from RF_bootstrap function) and makes matplotlib histogram of LogRF with Log(Bootstrap RF percentiles) indicated

        Parameters
        ----------
        RF_data : pandas DataFrame
            DataFrame containing "Chemical Name" and "RF" columns
        RF_boot : numpy array
            Array containing summary statistics for percentiles across RF bootstrap repetitions.
        num_bins : int, optional
            Number of bins to use for RF histogram. The default is 30.
        save_fig : Boolean, optional
            Save figure as .png file. The default is True.

        Returns
        -------
        None.

        """
        fig, ax = plt.subplots()
        # Log-10 transform RFs for plotting
        RF_data["LogRF"] = np.log10(RF_data["RF"])
        # Create histogram of qNTA surrogate RFs
        n, bins, patches = ax.hist(RF_data["LogRF"], num_bins, density=True)
        # Log-10 transform bootstrap RFs for plotting
        LogRF_boot = np.array([np.log10(x) for x in RF_boot])
        # If RF_boot is 2-dimensional, it has median, minimum, and maximum for percentiles across bootstrap repetitions
        if len(np.shape(RF_boot)) == 2:
            # Shade region between minimum and maximum for each percentile estimate
            ax.fill_betweenx([0, 1], LogRF_boot[1][0], LogRF_boot[2][0], alpha=0.3, color="c")  # Min and max RF0.025
            ax.fill_betweenx([0, 1], LogRF_boot[1][1], LogRF_boot[2][1], alpha=0.3, color="c")  # Min and max RF0.5
            ax.fill_betweenx([0, 1], LogRF_boot[1][2], LogRF_boot[2][2], alpha=0.3, color="c")  # Min and max RF0.975
            # Indicate bootstrap RF percentiles with red lines
            ax.vlines(LogRF_boot[0][0], 0, 1, colors="r")  # RF0.025
            ax.vlines(LogRF_boot[0][1], 0, 1, colors="r")  # RF0.5
            ax.vlines(LogRF_boot[0][2], 0, 1, colors="r")  # RF0.975
        # If RF_boot is 1-dimensional, it only has median for percentiles across bootstrap repetitions
        else:
            # Indicate bootstrap RF percentiles with red lines
            ax.vlines(LogRF_boot[0], 0, 1, colors="r")  # RF0.025
            ax.vlines(LogRF_boot[1], 0, 1, colors="r")  # RF0.5
            ax.vlines(LogRF_boot[2], 0, 1, colors="r")  # RF0.975
        if savefig:
            fig.savefig("RF_histogram.png")

    def set_occurrences(self, occurrence_data):
        """
        Set attribute occurrence_data in qNTAClass object.

        Parameters
        ----------
        occurrence_data : pandas DataFrame
            DataFrame containing columns for "Chemical Name" and samples, with samples column names beginning with "BlankSub Mean " and values containing BlankSub Mean abundances.

        Returns
        -------
        None.

        """
        self.occurrence_data = occurrence_data

    def RF_boot_estimate(
        self, RF_data, occurrence_data, seed=1, reps=10000, alpha=0.05, rep_range=True, long_form=False
    ):
        """
        Performs qNTA concentration estimation on occurrence_data using RF bootstrap percentiles

        Parameters
        ----------
        RF_data : pandas DataFrame
            DESCRIPTION. DataFrame containing "Chemical Name" and "RF" columns for qNTA surrogates
        occurrence_data : pandas DataFrame
            DESCRIPTION. DataFrame containing "Chemincal Name" column and columns with BlankSub Mean abundances (named using "BlankSub Mean {Sample}")
        seed : int, optional
            DESCRIPTION. Seed used for the random bootstrap sampling (np.random.choice()). The default is 1.
        reps : int, optional
            DESCRIPTION. Number of bootstrap repetitions. The default is 10000.
        alpha : float, optional
            DESCRIPTION. Alpha for the confidence level, determines the RF percentiles used. The default is 0.05.
        rep_range : Boolean, optional
            DESCRIPTION. Provide minimum and maximum for each RF percentile across bootstrap repetitions, in addition to median. The default is True.
        long_form : Boolean, optional
            DESCRIPTION. Return long form DataFrame (columns for "ConcLCL","ConcEst","ConcUCL", rows are unique chemical-sample combinations)

        Returns
        -------
        RF_estimate_out : pandas DataFrame
            If long_form, contains "Chemical Name", "Sample", "ConcLCL", "ConcEst", "ConcEst" columns
            Else, contains "Chemical Name" column and "{Sample}_ConcLCL","{Sample}_ConcEst", "{Sample}_UCL" for all samples

        """
        RF_estimate_out = occurrence_data.copy()
        # Get bootstrap percentile estimates
        RF_percs = self.RF_bootstrap(RF_data, seed, reps, alpha, rep_range)
        if long_form:
            # Change data to long form
            RF_estimate_out = pd.melt(
                RF_estimate_out,
                id_vars=["Chemical Name"],
                value_vars=RF_estimate_out.columns[RF_estimate_out.columns.str.startswith("BlankSub Mean ")].tolist(),
                var_name="Sample",
                value_name="BlankSub Mean",
            )
            # Remove "BlankSub Mean " from sample names
            RF_estimate_out["Sample"] = [i[14:] for i in RF_estimate_out["Sample"]]
            # Divide BlankSub Mean abundance by RF percentiles to get concentration estimates
            # Account for data shape of RF_estimate_out (if minimum and maximum of percentiles estimates across repetitions are present)
            if rep_range:
                RF_estimate_out["ConcLCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[2][1]
                RF_estimate_out["ConcEst"] = RF_estimate_out["BlankSub Mean"] / RF_percs[0][1]
                RF_estimate_out["ConcUCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[1][1]
            else:
                RF_estimate_out["ConcLCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[2]
                RF_estimate_out["ConcEst"] = RF_estimate_out["BlankSub Mean"] / RF_percs[0]
                RF_estimate_out["ConcUCL"] = RF_estimate_out["BlankSub Mean"] / RF_percs[1]
        else:
            abun_cols = RF_estimate_out.columns[RF_estimate_out.columns.str.startswith("BlankSub Mean ")].tolist()
            # Remove "BlankSub Mean" from sample names used for making concentration column names
            conc_col_names = [i[14:] for i in abun_cols]
            # Divide BlankSub Mean abundance by RF percentiles to get concentration estimates
            # Account for data shape of RF_estimate_out (if minimum and maximum of percentiles estimates across repetitions are present)
            if rep_range:
                RF_estimate_out[[f"{i}_ConcLCL" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[2][1]
                )
                RF_estimate_out[[f"{i}_ConcEst" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[0][1]
                )
                RF_estimate_out[[f"{i}_ConcUCL" for i in conc_col_names]] = (
                    RF_estimate_out.loc[:, abun_cols] / RF_percs[1][1]
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
            RF_estimate_out[["RF0.025", "RF0.5", "RF0.975"]] = [
                RF_percs[1][1],
                RF_percs[0][1],
                RF_percs[2][1],
            ]  # Dynamically set names to account for user's alpha choice? Or use RFlower, RFmed, RFupper?
        else:
            RF_estimate_out[["RF0.025", "RF0.5", "RF0.975"]] = [RF_percs[1], RF_percs[0], RF_percs[2]]
        if long_form:
            RF_estimate_out = RF_estimate_out.loc[
                :, ["Chemical Name", "Sample", "RF0.025", "RF0.5", "RF0.975", "ConcLCL", "ConcEst", "ConcUCL"]
            ]
            return RF_estimate_out
        else:
            # Reorder columns so that samples are grouped together
            column_order = [i + j for i in conc_col_names for j in ["_ConcLCL", "_ConcEst", "_ConcUCL"]]
            RF_estimate_out = RF_estimate_out.loc[:, ["Chemical Name"] + ["RF0.025", "RF0.5", "RF0.975"] + column_order]
            return RF_estimate_out

    # Remove zeroes beforehand?
    def set_validation_data(self, validation_data):
        """
        Set attribute validation_data in qNTAClass object.

        Parameters
        ----------
        validation_data : pandas DataFrame
            DataFrame containing columns for samples, whose values are targeted concentrations; the units of concentration must be the same as those used to generate the qNTA surrogate RF data.

        Returns
        -------
        None.

        """
        # Use copy to avoid overwriting original data
        self.validation_data = validation_data.copy()
        # self.val_samples = validation_data.columns[validation_data.columns.str.endswith("_")].tolist()

    # Remove LOO argument? (If always including LOO columns). Consider also other forms of cross validation (k-fold)
    def RF_boot_validation(
        self, seed=1, reps=10000, alpha=0.05, rep_range=True, long_form=False, LOO=True, internal=False
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
        # Columns containing the targeted concentrations for validation
        conc_cols = []

        # Always do LOO if doing internal validation
        if internal:
            LOO = True
            conc_cols = self.surrogate_cal_data.columns[
                self.surrogate_cal_data.columns.str.startswith("Conc ")
            ].tolist()
            self.validation_data = self.surrogate_cal_data.loc[:, ["Chemical Name"] + conc_cols]
        else:
            conc_cols = self.validation_data.columns[~self.validation_data.columns.isin(["Chemical Name"])].tolist()

        # Calculate global bootstrap RF percentiles and use to make concentration estimates
        # NOTE: For internal, occurrence_data must contain columns with names that correspond to conc_cols
        global_out = self.RF_boot_estimate(
            self.surrogate_cal_data_long_nonzero, self.occurrence_data, seed, reps, alpha, rep_range, long_form
        )

        # List of chemicals that overlap between qNTA surrogate set and validation data
        LOO_chems = list(
            set(self.surrogate_cal_data_long_nonzero_chems) & set(pd.unique(self.validation_data["Chemical Name"]))
        )

        # If chemicals overlap between qNTA surrogates and validation data and LOO is True
        if len(LOO_chems) > 0 and LOO:
            # Get LOO RF bootstrap percentiles and concentration estimates
            LOO_out = pd.concat(
                [
                    self.RF_boot_estimate(
                        self.surrogate_cal_data_long_nonzero[
                            self.surrogate_cal_data_long_nonzero["Chemical Name"] != i
                        ],
                        self.occurrence_data[self.occurrence_data["Chemical Name"] == i],
                        seed,
                        reps,
                        alpha,
                        rep_range,
                        long_form,
                    )
                    for i in LOO_chems
                ]
            )
            if long_form:
                # Change validation_data to long form
                self.validation_data = pd.melt(
                    self.validation_data,
                    id_vars="Chemical Name",
                    value_vars=conc_cols,
                    var_name="Sample",
                    value_name="ConcTargeted",
                )  # This is in order of sample and not chemical
                # Add _LOO suffix to column names (to distinguish LOO columns when adding to global estimates DataFrame)
                LOO_out = LOO_out.rename(
                    columns={c: c + "_LOO" for c in LOO_out.columns if c not in ["Chemical Name", "Sample"]}
                )
                # Remove "Conc" from sample names coming from internal calibration data, needed so that Sample columns match when merging
                if internal:
                    self.validation_data["Sample"] = [s[5:] for s in self.validation_data["Sample"]]
                LOO_out = pd.merge(
                    LOO_out, self.validation_data, on=["Chemical Name", "Sample"], how="left"
                )  # Ensure that correct ConcTargeted and ConcLCL, Est, UCL are compared
                # Calculate qNTA performance metrics for accuracy and uncertainty
                LOO_out["AQ_LOO"] = LOO_out["ConcEst_LOO"] / LOO_out["ConcTargeted"]
                LOO_out["AAQ_LOO"] = 10 ** np.abs(np.log10(LOO_out["AQ_LOO"]))
                LOO_out["CLFR_LOO"] = LOO_out["ConcUCL_LOO"] / LOO_out["ConcLCL_LOO"]
                LOO_out = LOO_out.drop(columns=["ConcTargeted"])  # ConcTargeted will be merged again later
            else:
                # Add _LOO suffix to column names (to distinguish LOO columns when adding to global estimates DataFrame)
                LOO_out = LOO_out = LOO_out.rename(
                    columns={c: c + "_LOO" for c in LOO_out.columns if c not in ["Chemical Name", "Sample"]}
                )
                # Rename columns prior to merge so that columns match
                if internal:
                    conc_cols = [c[5:] for c in conc_cols]
                self.validation_data.columns = ["Chemical Name"] + conc_cols
                LOO_out = pd.merge(LOO_out, self.validation_data, on="Chemical Name", how="left")
                # AQ
                LOO_out[[f"{i}_AQ_LOO" for i in conc_cols]] = LOO_out.loc[
                    :, LOO_out.columns.str.endswith("_ConcEst_LOO")
                ].rename(columns=lambda x: x.replace("__ConcEst_LOO", "_AQ_LOO")) / LOO_out.loc[:, conc_cols].rename(
                    columns=lambda x: x.replace("_", "_AQ_LOO")
                )
                # AAQ
                LOO_out[[f"{i}_AAQ_LOO" for i in conc_cols]] = 10 ** np.abs(
                    np.log10(LOO_out.loc[:, LOO_out.columns.str.endswith("_AQ_LOO")])
                ).rename(columns=lambda x: x.replace("_AQ_LOO", "_AAQ_LOO"))
                # CLFR
                LOO_out[[f"{i}_CLFR_LOO" for i in conc_cols]] = LOO_out.loc[
                    :, LOO_out.columns.str.endswith("_ConcUCL_LOO")
                ].rename(columns=lambda x: x.replace("_ConcUCL_LOO", "_CLFR_LOO")) / LOO_out.loc[
                    :, LOO_out.columns.str.endswith("_ConcLCL_LOO")
                ].rename(
                    columns=lambda x: x.replace("_ConcLCL_LOO", "_CLFR_LOO")
                )
                # Replace Inf AAQ and NaN CLFR values?
                # Drop these columns, they are added when merging to global_out
                LOO_out = LOO_out.drop(columns=conc_cols)

        # RF bootstrap with global surrogates (always performed)
        if long_form:
            # Change validation data to long form if not done already
            if LOO is False:
                self.validation_data = pd.melt(
                    self.validation_data,
                    id_vars="Chemical Name",
                    value_vars=conc_cols,
                    var_name="Sample",
                    value_name="ConcTargeted",
                )
            # Remove "Conc " from Sample names if not done already
            if internal and long_form is False:
                self.validation_data["Sample"] = [s[5:] for s in self.validation_data["Sample"]]
            # Left outer join keeps a row for all chemicals in validation data, with np.NaN (pd.NA?) for qNTA columns if not in global_out
            validation_out = pd.merge(global_out, self.validation_data, on=["Chemical Name", "Sample"], how="left")
            validation_out["AQ"] = validation_out["ConcEst"] / validation_out["ConcTargeted"]
            validation_out["AAQ"] = 10 ** np.abs(np.log10(validation_out["AQ"]))
            validation_out["CLFR"] = validation_out["ConcUCL"] / validation_out["ConcLCL"]
            # Merging LOO estimates with global estimates
            if len(LOO_chems) > 0 and LOO:
                validation_out = pd.merge(validation_out, LOO_out, on=["Chemical Name", "Sample"], how="left")
            # Remove rows where ConcEst = 0 since it leads to Inf AAQ and NaN CLFR?
            return validation_out
        else:
            validation_out = pd.merge(global_out, self.validation_data, on="Chemical Name", how="left")
            # AQ
            validation_out[[f"{i}_AQ" for i in conc_cols]] = validation_out.loc[
                :, validation_out.columns.str.endswith("_ConcEst")
            ].rename(columns=lambda x: x.replace("__ConcEst", "_AQ")) / validation_out.loc[:, conc_cols].rename(
                columns=lambda x: x.replace("_", "_AQ")
            )
            # AAQ.reset_index(drop=True)
            validation_out[[f"{i}_AAQ" for i in conc_cols]] = 10 ** np.abs(
                np.log10(validation_out.loc[:, validation_out.columns.str.endswith("_AQ")])
            ).rename(columns=lambda x: x.replace("_AQ", "_AAQ"))
            # CLFR
            validation_out[[f"{i}_CLFR" for i in conc_cols]] = validation_out.loc[
                :, validation_out.columns.str.endswith("_ConcUCL")
            ].rename(columns=lambda x: x.replace("_ConcUCL", "_CLFR")) / validation_out.loc[
                :, validation_out.columns.str.endswith("_ConcLCL")
            ].rename(
                columns=lambda x: x.replace("_ConcLCL", "_CLFR")
            )
            # Merging LOO estimates with global estimates
            if len(LOO_chems) > 0 and LOO:
                validation_out = pd.merge(validation_out, LOO_out, on="Chemical Name", how="left")
            return validation_out

    # If long_form = False, use separate sheets for LOO and global in output file, and separate sheets for the concentrations estimates and performance metrics?

    @staticmethod
    def validation_summary(validation_out, long_form=False, LOO=True):
        """
        Method to provide summary metrics (median) for qNTA performance metrics (AQ, AAQ, CLFR) and calculate reliability (ORP).

        Parameters
        ----------
        validation_out : pandas DataFrame
            DataFrame produced by method RF_boot_validation, containing columns for qNTA concentration estimates and qNTA performance metrics
        long_form : Boolean, optional
            DESCRIPTION. Specifies whether validation_out is long form (one column per performance metric) or wide form (one column per sample, per performance metric). The default is False.
        LOO : Boolean, optional
            DESCRIPTION. Specifies whether validation_out contains LOO columns (produced from leave-one-out qNTA bootstrap estimation). The default is True.

        Returns
        -------
        summary_out : pandas DataFrame
            DataFrame containing minimum, median, and maximum of qNTA performance metrics AQ, AAQ, CLFR and calculated reliability (ORP) for validation_out.

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

    # qNTA point estimates with confidence intervals, with targeted concentrations from validation data plotted where available
    # Plot LOO if available?
    # Offer option to sort by Sample instead of Chemical Name?
    @staticmethod
    def qNTA_estimate_plot(estimates, long_form=False, LOO=True, sort_by="ConcTargeted", savefig=True):
        """
        Creates plot of qNTA concentration estimates and confidence intervals, with validation targeted estimates plotted when available.

        Parameters
        ----------
        estimates : pandas DataFrame
            DataFrame containing columns for qNTA estimated concentration per sample (wide form) or column with rows for qNTA estimates concentrations per sample, per chemical (long form)
        long_form : Boolean, optional
            Indicates whether estimates DataFrame is wide or long form. The default is False.
        LOO: Boolean, optional
            Indicates whether leave-one-out concentration estimates are present. The default is True.
        sort_by : str, optional
            How the x-axis for the plot will be ordered from left to right, by ascending value. Options are "ConcTargeted" (only usable if the column is present in the estimates DataFrame), "ConcUCL", "Chemical Name". The default is "ConcTargeted".
        savefig : Boolean, optional
            Save figure as .png file. The default is True.

        Returns
        -------
        None.

        """
        # Make a copy to avoid modifying in-place
        estimate_plot_data = estimates.copy()
        # Initialize plot
        fig, ax = plt.subplots(figsize=(20, 12), layout="constrained")
        # Change from wide form to long form
        if long_form is False:
            # Get columns that have concentration estimates
            est_cols = estimate_plot_data.columns[estimate_plot_data.columns.str.contains("Conc")].tolist()
            # Put estimates all in one column
            estimate_plot_data = pd.melt(
                estimate_plot_data,
                id_vars="Chemical Name",
                value_vars=est_cols,
                var_name="SampleConc",
                value_name="Conc",
            )
            # Separate out sample name and concentration estimate type (LCL, Est, UCL)
            # Note: Assumes sample name ends with _ and no _ are present anywhere else in the same name
            estimate_plot_data[["Sample", "ConcName"]] = estimate_plot_data["SampleConc"].str.split("__", expand=True)
            # Make separate columns for Sample,ConcLCL, ConcEst, ConcUCL
            estimate_plot_data = estimate_plot_data.drop(columns=["SampleConc"]).pivot(
                index=["Chemical Name", "Sample"], columns="ConcName"
            )
            # Drop 'ConcName' level from column MultiIndex
            estimate_plot_data.columns = estimate_plot_data.columns.droplevel(0)
            # Extract concentration columns to turn into ConcTargeted
            conc_cols = estimates.columns[estimates.columns.str.endswith("_")].tolist()
            if conc_cols is not []:
                conc_data = estimates.loc[:, ["Chemical Name"] + conc_cols]
                targeted_cols = pd.melt(
                    conc_data,
                    id_vars="Chemical Name",
                    value_vars=conc_cols,
                    var_name="Sample",
                    value_name="ConcTargeted",
                )
                # Remove _ from end to align with estimate_plot_data names
                targeted_cols["Sample"] = [s[:-1] for s in targeted_cols["Sample"]]
                estimate_plot_data = pd.merge(
                    estimate_plot_data.reset_index(), targeted_cols, on=["Chemical Name", "Sample"], how="left"
                )

        # Sorting the x-axis
        if "ConcTargeted" in estimate_plot_data.columns:
            if sort_by == "ConcTargeted":
                # Sort by increasing ConcTargeted (take max per chemical) and reindex
                estimate_plot_data = pd.merge(
                    estimate_plot_data,
                    estimate_plot_data.groupby("Chemical Name")["ConcTargeted"]
                    .max()
                    .rename("Max_Targeted")
                    .reset_index(),
                )  # If Max_Targeted all the same, is equivalent to sorting by Chemical Name
                estimate_plot_data = (
                    estimate_plot_data.sort_values(by=["Max_Targeted", "Chemical Name", "ConcTargeted"])
                    .reset_index()
                    .drop(columns=["index"])
                )
        if sort_by == "ConcUCL":
            # Sort by increasing ConcUCL (take max per chemical) and reindex
            estimate_plot_data = pd.merge(
                estimate_plot_data,
                estimate_plot_data.groupby("Chemical Name")["ConcUCL"].max().rename("Max_UCL").reset_index(),
            )
            estimate_plot_data = (
                estimate_plot_data.sort_values(by=["Max_UCL", "Chemical Name", "ConcTargeted"])
                .reset_index()
                .drop(columns=["index"])
            )
        if sort_by == "Chemical Name":
            # Sort alphabetically by Chemical Name
            estimate_plot_data = (
                estimate_plot_data.sort_values(by=["Chemical Name", "ConcTargeted"])
                .reset_index()
                .drop(columns=["index"])
            )

        # Dashed lines to divide different concentrations
        dividers = (
            estimate_plot_data.rename_axis("x").reset_index().groupby(["Chemical Name"], sort=False)["x"].max() + 0.5
        )
        # Just use row numbers as x-axis since estimate_plot_data is already ordered
        x_num = np.arange(0, len(estimate_plot_data))

        # Add confidence intervals
        if "ConcEst_LOO" in estimate_plot_data.columns:
            # Global estimates on left
            ax.errorbar(
                x=x_num - 0.2,
                y=estimate_plot_data["ConcEst"],
                yerr=np.transpose(estimate_plot_data.loc[:, ["ConcLCL", "ConcUCL"]]),
                ls="none",
                label="BRFglobal",
            )
            ax.scatter(x=x_num - 0.2, y=estimate_plot_data["ConcEst"], s=3, label="BRFglobal")
            # LOO estimates on right
            ax.errorbar(
                x=x_num + 0.2,
                y=estimate_plot_data["ConcEst_LOO"],
                yerr=np.transpose(estimate_plot_data.loc[:, ["ConcLCL_LOO", "ConcUCL_LOO"]]),
                ls="none",
                label="BRFLOO",
                c="purple",
            )
            ax.scatter(x=x_num + 0.2, y=estimate_plot_data["ConcEst_LOO"], s=3, label="BRFLOO", c="purple")
            # Targeted concentration
            if "ConcTargeted" in estimate_plot_data.columns:
                ax.scatter(x=x_num - 0.2, y=estimate_plot_data["ConcTargeted"], s=3, label="ConcTargeted", c="red")
                ax.scatter(x=x_num + 0.2, y=estimate_plot_data["ConcTargeted"], s=3, label="ConcTargeted", c="red")
        else:
            # Global estimates centered
            ax.errorbar(
                x=x_num,
                y=estimate_plot_data["ConcEst"],
                yerr=np.transpose(estimate_plot_data.loc[:, ["ConcLCL", "ConcUCL"]]),
                label="BRFglobal",
                ls="none",
            )
            ax.scatter(x=x_num, y=estimate_plot_data["ConcEst"], label="BRFglobal", s=3)
            if "ConcTargeted" in estimate_plot_data.columns:
                ax.scatter(x=x_num, y=estimate_plot_data["ConcTargeted"], s=3, label="ConcTargeted", c="red")
        for i in dividers:
            ax.axvline(i, c="grey", linestyle="--")

        # Add chemical names to x-axis, locating ticks at median of chemical's row numbers
        xlocs = estimate_plot_data.rename_axis("x").reset_index().groupby(["Chemical Name"], sort=False)["x"].median()
        ax.set_xticks(xlocs, pd.unique(estimate_plot_data["Chemical Name"]))
        ax.tick_params(axis="x", labelrotation=90)
        ax.legend()
        plt.ylabel("Concentration")  # Extract concentration unit from estimates DataFrame or provide as str argument?
        plt.yscale("log")
        if savefig:
            fig.savefig("qNTA_estimation_plot.png")

    ### Performance metrics ###
    # AQ plots
    @staticmethod
    def AQ_plots(validation_out, long_form=False, LOO=True, savefig=True):
        """
        Creates plots of the Accuracy Quotient (AQ) qNTA performance metric, including a boxplot of AQ values and a scatterplot of AQ vs. ConcTargeted.

        Parameters
        ----------
        validation_out : pandas DataFrame
            DataFrame output by method RF_boot_validation. Contains columns for AQ per sample (wide form) or one AQ column with per-sample, per-chemical values per row (long form).
        long_form : Boolean, optional
            Indicates whether the validation_out DataFrame is wide or long form. The default is False.
        LOO : Boolean, optional
            Indicates whether AQ values from leave-one-out qNTA estimates are present in the validation_out DataFrame. The default is True.
        savefig : Boolean, optional
            Save figure as .png file. The default is True.

        Returns
        -------
        None.

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
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
            # Boxplot
            ax1.boxplot([AQ_plot_data["AQ"], AQ_plot_data["AQ_LOO"]], tick_labels=["AQ", "AQ_LOO"])
            # Add points with x scatter
            ax1.scatter(np.random.normal(1, 0.01, len(AQ_plot_data)), AQ_plot_data["AQ"], label="AQ", alpha=0.6)
            ax1.scatter(
                np.random.normal(2, 0.01, len(AQ_plot_data)),
                AQ_plot_data["AQ_LOO"],
                c="purple",
                label="AQ LOO",
                alpha=0.6,
            )
            ax1.set_title("Boxplot of log10(AQ)")
            ax1.set_yscale("log")
            # Scatter plot
            ax2.scatter(AQ_plot_data["ConcTargeted"], AQ_plot_data["AQ"], label="AQ", alpha=0.5)
            ax2.scatter(AQ_plot_data["ConcTargeted"], AQ_plot_data["AQ_LOO"], label="AQ LOO", c="purple", alpha=0.5)
            ax2.set_title("Scatterplot of log10(AQ) vs. log10(ConcTargeted)")
            ax2.set_xscale("log")
            ax2.set_yscale("log")

            for ax in (ax1, ax2):
                ax.legend()
        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
            # Boxplot
            ax1.boxplot(AQ_plot_data["AQ"])
            # Add points with x scatter
            ax1.scatter(np.random.normal(1, 0.01, len(AQ_plot_data)), AQ_plot_data["AQ"], alpha=0.6)
            ax1.set_title("Boxplot of log10(AQ)")
            ax1.set_yscale("log")
            # Scatter plot
            ax2.scatter(AQ_plot_data["ConcTargeted"], AQ_plot_data["AQ"])
            ax2.set_title("Scatterplot of log10(AQ) vs. log10(ConcTargeted)")
            ax2.set_xscale("log")
            ax2.set_yscale("log")

        plt.tight_layout()

        if savefig:
            fig.savefig("AQ_plots.png")

    @staticmethod
    def ecdf(validation_out, long_form=False, LOO=True, savefig=True):
        """
        Creates empirical cumulative distribution function point plots for qNTA performance metrics (Absolute Accuracy Quotient (AAQ) and Confidence Limit Fold Range (CLFR)).

        Parameters
        ----------
        validation_out : pandas DataFrame
            A DataFrame output by method RF_boot_validation, containing columns for qNTA concentration estimates per sample (wide form) or rows for per-chemical, per-sample estimates (long form)
        long_form : Boolean, optional
            Indicates whether the validation_out DataFrame is wide or long form. The default is False.
        LOO : Boolean, optional
            Indicates whether qNTA performance metrics for leave-one-out estimates are present in the validation_out DataFrame. The default is True.
        savefig : Boolean, optional
            Save figure as .png file. The default is True.

        Returns
        -------
        None.

        """
        ecdf_data = validation_out.copy()
        # Removing columns with LOO
        if LOO:
            ecdf_data_LOO = ecdf_data.loc[
                :, ["Chemical Name"] + ecdf_data.columns[ecdf_data.columns.str.contains("LOO")].tolist()
            ]
            ecdf_data = ecdf_data.loc[:, ecdf_data.columns[~ecdf_data.columns.str.contains("LOO")].tolist()]
        if long_form:
            # Remove ConcEst = 0 because it means AAQ is Inf and CLFR is NaN
            ecdf_data = ecdf_data[ecdf_data["ConcEst"] > 0]
            # Sort by ascending value
            # Get y-axis value by dividing the cumulative sum of the metric by the total sum
            ecdf_data_AAQ = pd.DataFrame(data=np.sort(ecdf_data["AAQ"]), columns=["AAQ"])
            ecdf_data_AAQ = ecdf_data_AAQ.assign(
                AAQ_rank=np.cumsum(ecdf_data_AAQ["AAQ"]) / np.sum(ecdf_data_AAQ["AAQ"])
            )  # Should use sum, not length
            ecdf_data_CLFR = pd.DataFrame(np.sort(ecdf_data["CLFR"]), columns=["CLFR"])
            ecdf_data_CLFR = ecdf_data_CLFR.assign(
                CLFR_rank=np.cumsum(ecdf_data_CLFR["CLFR"]) / np.sum(ecdf_data_CLFR["CLFR"])
            )
            if LOO:
                # Limit to occurrences with AQ_LOO > 0 (ConcEst > 0)
                ecdf_data_LOO = ecdf_data_LOO[ecdf_data_LOO["AQ_LOO"] > 0]
                ecdf_data_AAQ_LOO = pd.DataFrame(data=np.sort(ecdf_data_LOO["AAQ_LOO"]), columns=["AAQ_LOO"])
                ecdf_data_AAQ_LOO = ecdf_data_AAQ_LOO.assign(
                    AAQ_LOO_rank=np.cumsum(ecdf_data_AAQ_LOO["AAQ_LOO"]) / np.sum(ecdf_data_AAQ_LOO["AAQ_LOO"])
                )
                ecdf_data_CLFR_LOO = pd.DataFrame(np.sort(ecdf_data_LOO["CLFR_LOO"]), columns=["CLFR_LOO"])
                ecdf_data_CLFR_LOO = ecdf_data_CLFR_LOO.assign(
                    CLFR_LOO_rank=np.cumsum(ecdf_data_CLFR_LOO["CLFR_LOO"]) / np.sum(ecdf_data_CLFR_LOO["CLFR_LOO"])
                )
        else:
            # Get targeted concentration columns and make ConcTargeted column
            conc_cols = ecdf_data.columns[ecdf_data.columns.str.endswith("_")].tolist()
            conc_data = ecdf_data.copy().loc[:, ["Chemical Name"] + conc_cols]
            targeted_cols = pd.melt(
                conc_data, id_vars="Chemical Name", value_vars=conc_cols, var_name="Sample", value_name="ConcTargeted"
            )
            targeted_cols["Sample"] = [s[:-1] for s in targeted_cols["Sample"]]
            # Change data from wide to long form
            ecdf_data = pd.melt(
                ecdf_data,
                id_vars="Chemical Name",
                value_vars=ecdf_data.columns[ecdf_data.columns.str.contains("AAQ|CLFR")].tolist(),
                var_name="SampleMetric",
                value_name="Metric",
            )
            ecdf_data[["Sample", "MetricName"]] = ecdf_data["SampleMetric"].str.split("__", expand=True)
            ecdf_data = ecdf_data.drop(columns=["SampleMetric"]).pivot(
                index=["Chemical Name", "Sample"], columns="MetricName"
            )
            # Drop 'MetricName' level from column MultiIndex
            ecdf_data.columns = ecdf_data.columns.droplevel(0)
            ecdf_data = ecdf_data.reset_index()
            # Add ConcTargeted column
            ecdf_data = pd.merge(ecdf_data, targeted_cols, on=["Chemical Name", "Sample"], how="left")
            # Removes AAQ Inf and CLFR is NaN
            ecdf_data = ecdf_data[np.isfinite(ecdf_data["AAQ"])]
            # Sort by ascending value
            # Get y-axis value by dividing the cumulative sum of the metric by the total sum
            ecdf_data_AAQ = pd.DataFrame(data=np.sort(ecdf_data["AAQ"]), columns=["AAQ"])
            ecdf_data_AAQ = ecdf_data_AAQ.assign(
                AAQ_rank=np.cumsum(ecdf_data_AAQ["AAQ"]) / np.sum(ecdf_data_AAQ["AAQ"])
            )
            ecdf_data_CLFR = pd.DataFrame(data=np.sort(ecdf_data["CLFR"]), columns=["CLFR"])
            ecdf_data_CLFR = ecdf_data_CLFR.assign(
                CLFR_rank=np.cumsum(ecdf_data_CLFR["CLFR"]) / np.sum(ecdf_data_CLFR["CLFR"])
            )
            if LOO:
                ecdf_data_LOO = pd.melt(
                    ecdf_data_LOO,
                    id_vars="Chemical Name",
                    value_vars=ecdf_data_LOO.columns[ecdf_data_LOO.columns.str.contains("AAQ|CLFR")].tolist(),
                    var_name="SampleMetric",
                    value_name="Metric",
                )
                ecdf_data_LOO[["Sample", "MetricName"]] = ecdf_data_LOO["SampleMetric"].str.split("__", expand=True)
                ecdf_data_LOO = ecdf_data_LOO.drop(columns=["SampleMetric"]).pivot(
                    index=["Chemical Name", "Sample"], columns="MetricName"
                )
                # Drop 'MetricName' level from column MultiIndex
                ecdf_data_LOO.columns = ecdf_data_LOO.columns.droplevel(0)
                ecdf_data_LOO = ecdf_data_LOO.reset_index()
                # Add ConcTargeted column
                ecdf_data_LOO = pd.merge(ecdf_data_LOO, targeted_cols, on=["Chemical Name", "Sample"], how="left")
                # Removes AAQ Inf and CLFR is NaN
                ecdf_data_LOO = ecdf_data_LOO[np.isfinite(ecdf_data_LOO["AAQ_LOO"])]
                # Sort by ascending value
                ecdf_data_AAQ_LOO = pd.DataFrame(data=np.sort(ecdf_data_LOO["AAQ_LOO"]), columns=["AAQ_LOO"])
                ecdf_data_AAQ_LOO = ecdf_data_AAQ_LOO.assign(
                    AAQ_LOO_rank=np.cumsum(ecdf_data_AAQ_LOO["AAQ_LOO"]) / np.sum(ecdf_data_AAQ_LOO["AAQ_LOO"])
                )
                ecdf_data_CLFR_LOO = pd.DataFrame(np.sort(ecdf_data_LOO["CLFR_LOO"]), columns=["CLFR_LOO"])
                ecdf_data_CLFR_LOO = ecdf_data_CLFR_LOO.assign(
                    CLFR_LOO_rank=np.cumsum(ecdf_data_CLFR_LOO["CLFR_LOO"]) / np.sum(ecdf_data_CLFR_LOO["CLFR_LOO"])
                )

        # Plot points
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
        ax1.scatter(ecdf_data_AAQ["AAQ"], ecdf_data_AAQ["AAQ_rank"])
        ax2.scatter(ecdf_data_CLFR["CLFR"], ecdf_data_CLFR["CLFR_rank"])
        ax1.set_xscale("log")
        # ax2.set_xscale('log') # Using logscale x-axis prevents CLFR from plotting when they are all the same value (for BRFglobal without BRFLOO)
        ax1.set_xlabel("AAQ")
        ax1.set_ylabel("Rank")
        ax1.set_title("Cumulative distribution of AAQ")
        ax2.set_xlabel("CLFR")
        ax2.set_ylabel("Rank")
        ax2.set_title("Cumulative distribution of CLFR")
        # Add LOO points
        if LOO:
            ax1.scatter(ecdf_data_AAQ_LOO["AAQ_LOO"], ecdf_data_AAQ_LOO["AAQ_LOO_rank"], c="purple")
            ax2.scatter(ecdf_data_CLFR_LOO["CLFR_LOO"], ecdf_data_CLFR_LOO["CLFR_LOO_rank"], c="purple")
            for ax in (ax1, ax2):
                ax.legend(["BRFGlobal", "BRFLOO"])
        if savefig:
            fig.savefig("Cumulative_Distributions_AAQ_CLFR.png")
