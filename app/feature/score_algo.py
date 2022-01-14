# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:57:54 2021

@author: MBOYCE
"""

import pandas as pd
import numpy as np

        
class SpectraScorer():
    """
    Scoring algorithm pulled from Stein, S; Scott, D. Optimization and testing of mass spectral library search algorithms for compound identification
    https://doi.org/10.1016/1044-0305(94)87009-8
    Current implementation allows two scoring methods:
        1) DOT-PRODUCT
        2) COMPOSITE
    Implmentations to code:
        1) SE (Spectral Entropy, Fiehn group)
    """
    def calc_score(self, measured_spectrum, reference_spectrum, equation = 'COMPOSITE', **kwargs):
        score = self._get_score_method(equation)
        aligned_df = self._get_align_df(measured_spectrum, reference_spectrum)
        return score(aligned_df, **kwargs)
    
    def _get_score_method(self, equation):
        if equation == 'DOT_PRODUCT':
            return self._calc_dot_product
        elif equation == 'COMPOSITE':
            return self._calc_composite
        else:
            raise ValueError(equation)

    def _get_align_df(self, measured_spectrum, reference_spectrum):
        return measured_spectrum.align(reference_spectrum, suffixes = ['_u', '_l'])
            
    def _calc_dot_product(self, aligned_df, m = 0.5, n = 0.5):
        
        aligned_df['weighted_u'] = (aligned_df['frag_mass_u']**n) * (aligned_df['intensity_u']**m)
        aligned_df['weighted_l'] = (aligned_df['frag_mass_l']**n) * (aligned_df['intensity_l']**m)
        numerator = sum(aligned_df['weighted_u'] * aligned_df['weighted_l'])**2
        denominator = sum(map(lambda x: x**2, aligned_df['weighted_u']))*sum(map(lambda x: x**2, aligned_df['weighted_l']))
        return numerator/denominator
    
    def _calc_composite(self, aligned_df, m = 0.5, n = 0.5):
        
        N_u = sum(aligned_df['intensity_u'] > 0)
        N_lu = sum(aligned_df['mass_match'] == 1)
        if N_lu == 0:   
            return 0    #Returns a 0 for the score if no matches are present in the data
        F_dot = self._calc_dot_product(aligned_df, m, n)
        F_ratio = self._calc_ratio_pairs(aligned_df, N_lu)
        return (N_u*F_dot + N_lu*F_ratio)/(N_u + N_lu)
    
    def _calc_ratio_pairs(self, aligned_df, N_lu):
        SUM = 0.0
        matched_df = aligned_df[aligned_df['mass_match'] == 1].reset_index(drop = True)
        for i in range(1, N_lu):
            numerator = matched_df['weighted_l'][i]*matched_df['weighted_u'][i-1]
            denominator = matched_df['weighted_l'][i-1]*matched_df['weighted_u'][i]
            l = 1 if numerator/denominator <= 1 else -1
            
            SUM += (numerator/denominator)**l
        f_r = 1/float(N_lu) * SUM
        return f_r
    