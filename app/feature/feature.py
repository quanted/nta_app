# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:01:38 2021

@author: MBOYCE
"""

import numpy as np
import pandas as pd
from .score_algo import SpectraScorer


class Feature:
    """
    Parent class for MS features. Provides a base class for the FeatureMS1
    and FeatureMS2 child classes to inherit from.

    :param feature_id: Unique identifier for the feature being constructed
    :type feature_id: int
    :param mass: this is a second param
    :type mass:
    :param mass_accuracy: Value used to set the mass accuracy (ppm)
    :type mass_accuracy: float, optional
    :param rt_accuracy: Value used to set the upper and lower range of the RT
    :type rt_accuracy: float, optional

    """

    def __init__(self, feature_id, mass, rt, mass_accuracy=10, rt_accuracy=0.2):
        """
        Constructor method
        """
        self.feature_id = feature_id
        self.mass = mass
        self.mass_accuracy = mass_accuracy
        self.rt = rt
        self.rt_accuracy = rt_accuracy
        self.classification = {}

    def annotate(self, value):
        """
        Set value of the annotation of a feature
        """
        self.annotation = value

    def __repr__(self):
        """
        Returns a string to the console when the feature object is called.
        """
        return f"\nID: {self.feature_id} - Mass: {self.mass} - RT: {self.rt}"


class Feature_MS1(Feature):
    """
    WIP

    :param feature_id: Unique identifier for the feature being constructed
    :type feature_id: int

    :returns:
    """

    def __init__(self, feature_id, data_dict, mass_accuracy, rt_accuracy):
        super(Feature_MS1, self).__init__(feature_id, data_dict["MASS"], data_dict["RT"])


class Feature_MS2(Feature):
    """
    Reads in a block of MGF data, which corresponds to MS2 spectrum of a single precursor ion
    Child class of Feature object

    Keyword arguments:
    :param feature_id: Identifier to a feature to track across different outputs
    :type feature_id: Int
    :param data_dict: Data input from data parser. Structered as a list of dictionaries in the following format -
        {'MASS': float, 'RT': float, 'CHARGE': str, 'FRAG_MASS':list, 'FRAG_INTENSITY':list}
    :type data_dict: list of dictionaries
    :param mass_is_netural: Bool to indicate if the input masses are neutral (True) or charged (False), (default is false)
    :type mass_is_neutral: Bool, optional
    :param POSMODE: Mode of ESI (defualt True)
    :type POSMODE: Bool, optional
    :param precursor_mass_accuracy: Tolerance of mass error in ppm (default 10 ppm)
    :type precursor_mass_accuracy: Int
    :param fragment_mass_accuracy: Tolerance of mass error in framgents (defaults 0.02 Da)
    :type fragment_mass_accuracy: float, optional
    :param rt_accuracy: Tolerance of retention time in min (default 0.2 min)
    :type rt_accuracy: float, optional
    """

    def __init__(
        self,
        feature_id,
        data_dict,
        mass_is_neutral=False,
        POSMODE=True,
        precursor_mass_accuracy=10,
        fragment_mass_accuracy=0.02,
        rt_accuracy=0.2,
    ):
        super(Feature_MS2, self).__init__(
            feature_id, data_dict["MASS"], data_dict["RT"], precursor_mass_accuracy, rt_accuracy
        )
        self.fragment_mass_accuracy = fragment_mass_accuracy
        self.mode = "POS" if POSMODE else "NEG"
        self.set_neutral_mass(mass_is_neutral)
        self.ms2_spectrum = MS2_Spectrum(data_dict["FRAG_MASS"], data_dict["FRAG_INTENSITY"], precursor_mass=self.mass)
        self.total_signal = sum(self.ms2_spectrum.frag_intensity)
        self.feature_data = {
            "ID": self.feature_id,
            "MASS_MGF": self.mass,
            "MASS_NEUTRAL": self.neutral_mass,
            "RT": self.rt,
        }

        self.reference_scores = {
            "ID": [],
            "MASS_MGF": [],
            "MASS_NEUTRAL": [],
            "RT": [],
            "DTXCID": [],
            "MASS": [],
            "FORMULA": [],
            "SINGLE_SCORES": [],
            "SUM_SCORE": [],
            "Q-SCORE": [],
            "PERCENTILE": [],
        }

    def set_neutral_mass(self, is_neutral):
        if is_neutral:
            self.neutral_mass = self.mass
            return
        # self.neutral_mass = self.mass + 1.0073 if self.mode == 'POS' else self.mass - 1.0073  #This is the opposite of what it should be, but gives correct resutls. Tracking down bug MWB
        self.neutral_mass = (
            self.mass - 1.0073 if self.mode == "POS" else self.mass + 1.0073
        )  # 2/23/2023 Flip this to correct this (in addition to adjusting save_data function in ms2_task to fix Jira issue NTAW-68)

    def merge(self, other):  # Implement later, consensus merge with fragment alignment
        return self

    def append_reference_similarity(self, dtxcid, formula, mass, scores):
        self.reference_scores["ID"].append(self.feature_id)
        self.reference_scores["MASS_MGF"].append(self.mass)
        self.reference_scores["MASS_NEUTRAL"].append(self.neutral_mass)
        self.reference_scores["RT"].append(self.rt)
        self.reference_scores["DTXCID"].append(dtxcid)
        self.reference_scores["FORMULA"].append(formula)
        self.reference_scores["MASS"].append(mass)
        self.reference_scores["SINGLE_SCORES"].append(scores)
        self.reference_scores["SUM_SCORE"].append(sum(scores))

    def dask_calc_similarity(self, spectra_dict):
        reference_scores = {
            "ID": [],
            "MASS_MGF": [],
            "MASS_NEUTRAL": [],
            "RT": [],
            "DTXCID": [],
            "MASS": [],
            "FORMULA": [],
            "SINGLE_SCORES": [],
            "SUM_SCORE": [],
            "Q-SCORE": [],
            "PERCENTILE": [],
        }

        for identifiers, spectra in spectra_dict.items():
            reference_scores["ID"].append(self.feature_id)
            reference_scores["MASS_MGF"].append(self.mass)
            reference_scores["MASS_NEUTRAL"].append(self.neutral_mass)
            reference_scores["RT"].append(self.rt)
            reference_scores["DTXCID"].append(identifiers[0])
            reference_scores["FORMULA"].append(identifiers[1])
            reference_scores["MASS"].append(identifiers[2])
            single_scores = [
                SpectraScorer.calc_score(self.ms2_spectrum, spectrum) for energy, spectrum in spectra.items()
            ]
            reference_scores["SINGLE_SCORES"].append(single_scores)
            reference_scores["SUM_SCORE"].append(sum(single_scores))

        return reference_scores

    def calc_similarity(self, spectra_dict):
        """
        Iterates through self.reference_spectra and calcualtes similarity
        scores betwen the parent's MS2_Spectrum and reference spectra captued in parent
        features reference_spectra dictionary

        As scores are calculated, a dictionary is updated with corresponding data
        to be used as an output.

        :param spectra_dict: Nested dictionary output from the utilities.dask_ms2_search_api search
        represented by the following structure:

            {(DTXCID, FORMULA, MASS):
                 {'ENERGY1': MS2_Spectrum,
                  'ENERGY2': MS2_Spectrum,
                  'ENERGY3': MS2_Spectrum}
                 },
            }

        :param spectra_scorer: SpectraScorer class object used to calcualte similarity (deafult None)
        :param type: SpectraScorer, optional
        """

        for identifiers, spectra in spectra_dict.items():
            self.reference_scores["ID"].append(self.feature_id)
            self.reference_scores["MASS_MGF"].append(self.mass)
            self.reference_scores["MASS_NEUTRAL"].append(self.neutral_mass)
            self.reference_scores["RT"].append(self.rt)
            self.reference_scores["DTXCID"].append(identifiers[0])
            self.reference_scores["FORMULA"].append(identifiers[1])
            self.reference_scores["MASS"].append(identifiers[2])
            single_scores = [
                SpectraScorer.calc_score(self.ms2_spectrum, spectrum) for energy, spectrum in spectra.items()
            ]
            self.reference_scores["SINGLE_SCORES"].append(single_scores)
            self.reference_scores["SUM_SCORE"].append(sum(single_scores))

    def __eq__(self, other):
        mass_equivalent = abs((other.mass - self.mass) / self.mass) * 1000000 < self.mass_accuracy
        rt_equivalent = abs(other.rt - self.rt) < self.rt_accuracy
        return mass_equivalent and rt_equivalent

    def __lt__(self, other):
        if self == other:
            return self.total_signal < other.total_signal
        return None

    def __le__(self, other):
        if self == other:
            return self.total_signal <= other.total_signal
        return None

    def __gt__(self, other):
        if self == other:
            return self.total_signal > other.total_signal
        return None

    def __ge__(self, other):
        if self == other:
            return self.total_signal >= other.total_signal
        return None


class FeatureList:
    """
    Object that captures a list of Features and provides users the ability to
    1) Modify features in the list (update_feature_list() and update_feature())
    2) Get a list of unique masses (get_masses())
    3) Get features using a lookup parameter (get_features())
    4) Export feature data into a dataframe

    As scores are calculated, a dictionary is updated with corresponding data
    to be used as an output.

    :param spectra_scorer: SpectraScorer class object used to calcualte similarity (deafult None)
    :param type: SpectraScorer, optional
    """

    def __init__(self, mgf_parse_result=None):
        self.feature_list = []
        self.update_feature_list(mgf_parse_result)

    def update_feature_list(self, mgf_parse_result, **kwargs):
        if mgf_parse_result:
            for datablock in mgf_parse_result:
                new_feature = Feature_MS2(len(self.feature_list), datablock, **kwargs)
                if new_feature in self.feature_list:
                    self.update_feature(new_feature)
                else:
                    self.feature_list += [new_feature]

    def update_feature(self, new_feature, mode="replace"):
        """
        Updates feature list if the list already has the feature to be added (based on mass and rt error).
        Compares the two features and replaces the old feature if the total signal of the new feature is greater
        """
        old_feature = self.feature_list[self.feature_list.index(new_feature)]
        if mode == "replace":
            if old_feature.total_signal < new_feature.total_signal:
                new_feature.feature_id = old_feature.feature_id
                self.feature_list.remove(old_feature)
                self.feature_list.append(new_feature)

        # Consensus comparison not implemented yet
        # if mode == 'consensus':
        #    old_feature.merge(new_feature)

    def get_masses(self, neutral=False):
        """
        Returns a list of unqiue precursor masses pulled from all features in self.feature_list

        :param netural: Bool used to decide if neutral masses should be resturned (defualt false)
        :type neutral: Bool, optional
        """
        if neutral:
            return list(set([feature.neutral_mass for feature in self.feature_list]))
        return list(set([feature.precursor_mass_mgf for feature in self.feature_list]))

    def get_features(self, lookup_value, by="mass"):
        """
        Returns a list of features that have the corresponding lookup_value

        :param lookup_value: Value to find features in the self.feature_list
        :type lookup_value: str or int
        :param by: String character used to define the attribute to pull from the features (deafult mass)
                                additonal values: 'rt', 'feature_id', 'mode', 'netural_mass'
        :param by: str, optional
        """
        return [feature for feature in self.feature_list if getattr(feature, by) == lookup_value]

    def to_df(self):
        """
        Exports data of all features in self.feature_list in a single dataframe
        Iterates through each feature, then each feature's reference spectra

        :returns: datframe constructed from Feature data
        """

        feature_dict = {
            "ID": [],
            "MASS_MGF": [],
            "MASS_NEUTRAL": [],
            "RT": [],
            "DTXCID": [],
            "MASS": [],
            "FORMULA": [],
            "SINGLE_SCORES": [],
            "SUM_SCORE": [],
            "Q-SCORE": [],
            "PERCENTILE": [],
        }

        for feature in self.feature_list:
            feature.reference_scores["Q-SCORE"] = [
                0
                if max(feature.reference_scores["SUM_SCORE"]) == 0
                else score / max(feature.reference_scores["SUM_SCORE"])
                for score in feature.reference_scores["SUM_SCORE"]
            ]

            # NTAW-537: Add percentile scores (doing this in a temporary dataframe for now, probably inefficient)
            df_temp = pd.DataFrame({"sum_scores": feature.reference_scores["SUM_SCORE"]})
            # NTAW-606: If all scores are zero, then make all percentiles zero
            if df_temp["sum_scores"].eq(0).all():
                df_temp["percentile"] = 0
            else:
                df_temp["percentile"] = df_temp["sum_scores"].rank(pct=True)  # Calculate percentile values
            feature.reference_scores["PERCENTILE"] = df_temp["percentile"].tolist()

            for key in feature_dict.keys():
                feature_dict[key].extend(feature.reference_scores.get(key, None))

        return pd.DataFrame.from_dict(feature_dict)

    def _remove_unnanotated_features(self):
        self.feature_list[:] = [feature for feature in self.feature_list if len(feature.reference_spectra) > 0]

    def calc_similarity(self):
        self._remove_unnanotated_features()
        for feature in self.feature_list:
            feature.calc_similarity_scores()

    def __len__(self):
        return len(self.feature_list)

    def __contains__(self, other):
        return sum(np.where(self.feature_list == other)) > 0

    def __repr__(self):
        return f"List of features: {self.feature_list}"


class MS2_Spectrum:
    """
    Reads MGF block data associated with a precursor ion, and standardizes the signal which:
        1) Removes any measured masses greater than the precursor mass
        2) Normalizes the signal after step 1
    """

    def __init__(self, fragment_mass, fragment_intensity, precursor_mass=None):
        self.precursor = precursor_mass
        self.frag_mass = fragment_mass
        self.frag_intensity = fragment_intensity
        self.reduced_fragment_mass = None
        self.reduced_signal = None
        self.normalized_signal = None
        self.spectrum_df = None
        self.standardize_spectrum()

    def standardize_spectrum(self):
        """
        Standardizes the the fragment_mass and signal inputs by:
            1) Zipping the fragment_mass and signal data together and removing any masses > the precusor
            2) Unzipping that merge into the reduced_fragment_mass and reduced_signal (this approach preserves the order of features)
            3) normalizing the signals using the max intensity of the reduced signal
            4) Return a dataframe of the fragment mass and normalized intensity values
        """
        # Commented code is prototyped to remove parent ion from comparison
        # merged_data = {(k,v) for k,v in zip(self.frag_mass, self.frag_intensity) if k <= self.precursor}
        merged_data = {(k, v) for k, v in zip(self.frag_mass, self.frag_intensity)}
        self.reduced_fragment_mass, self.reduced_signal = list(zip(*merged_data))
        self.normalize_signal()
        self.spectrum_df = pd.DataFrame(
            {"FRAGMENT_MASS": self.reduced_fragment_mass, "INTENSITY": self.normalized_signal}
        )

    def normalize_signal(self):
        self.normalized_signal = np.divide(self.reduced_signal, max(self.reduced_signal)) * 100

    def __add__(self, other):
        # TODO: Implement as part of consense MS merge
        selfDF = pd.DataFrame({"Frag": self.reduced_fragment_mass, "Signal_self": self.reduced_signal})
        otherDF = pd.DataFrame({"Frag": other.reduced_fragment_mass, "Signal_other": other.reduced_signal})
        mergeDF = pd.merge(selfDF, otherDF).fillna(0, inplace=True)

    def align(self, other_spectrum, frag_mass_accuracy=0.2, suffixes=["_self", "_other"]):
        """
        Aligns this object's fragment mass and intensity vlaues with the other spectrum's mass and intensity values.
        Mass values are aligned by:
            2) Binning masses into goups by rounding masses to the nearest whole number
            2) Constructing dataframe of self and other, then merging the two matricies on the 'bin_mass' column to
                    generate all possible combinations of masses in corresponding bins
            3) Calculate the absolute difference between the the fragment masses between the self and other spectrum and
                    assign a mass_match value if the difference is within the frag_mass_accuracy
            4) Remove ducpliate fragment masses by 1. sorting vlaues on the delta 2. subsetting on self to return features
                    that are either unqiue or nan AND subsetting on other with values that are unqiue OR nan.
            5)
            6) Return the subset matrix as the aligned maxtrix

        """
        df_self = pd.DataFrame(
            {
                "bin_mass": list(map(lambda x: round(x, 0), self.reduced_fragment_mass)),
                "frag_mass": self.reduced_fragment_mass,
                "intensity": self.normalized_signal,
            }
        )

        df_other = pd.DataFrame(
            {
                "bin_mass": list(map(lambda x: round(x, 0), other_spectrum.reduced_fragment_mass)),
                "frag_mass": other_spectrum.reduced_fragment_mass,
                "intensity": other_spectrum.normalized_signal,
            }
        )

        merged_df = pd.merge(df_self, df_other, how="outer", on="bin_mass", suffixes=suffixes)
        merged_df["mass_delta"] = abs(merged_df["frag_mass" + suffixes[0]] - merged_df["frag_mass" + suffixes[1]])

        if frag_mass_accuracy < 1:
            merged_df["mass_match"] = np.where(merged_df["mass_delta"] < frag_mass_accuracy, 1, 0)
        else:
            merged_df["mass_match"] = np.where(
                merged_df["mass_delta"] / merged_df["frag_mass" + suffixes[0]] * 1000000 < frag_mass_accuracy, 1, 0
            )

        merged_df["frag_mass" + suffixes[0]].fillna(merged_df["frag_mass" + suffixes[1]], inplace=True)
        merged_df["frag_mass" + suffixes[1]].fillna(merged_df["frag_mass" + suffixes[0]], inplace=True)
        merged_df.fillna(0, inplace=True)
        merged_df.sort_values(by="mass_delta", ascending=True, inplace=True)
        aligned_df = merged_df[
            (~merged_df.duplicated(subset="frag_mass" + suffixes[0], keep="first"))
            & (~merged_df.duplicated(subset="frag_mass" + suffixes[1], keep="first"))
        ].copy()
        aligned_df.drop(["bin_mass", "mass_delta"], axis=1, inplace=True)
        aligned_df.sort_values(by="frag_mass" + suffixes[0], inplace=True)
        aligned_df.reset_index(drop=True, inplace=True)

        return aligned_df

    def __repr__(self):
        return repr(self.spectrum_df)
