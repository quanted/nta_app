from __future__ import absolute_import
import os
import re
import pandas as pd
from operator import itemgetter
from itertools import groupby
from difflib import SequenceMatcher


class FileParser:
    def __init__(self):
        pass

    def run(self, file):
        ext = os.path.basename(file.name).split(".")[-1]
        parser = self._get_file_parser(ext)
        return parser(file)

    def _get_file_parser(self, parser: str):
        if parser == "csv":
            return self._parse_csv
        elif parser == "tsv":
            return self._parse_tsv
        elif parser == "xlsx":
            return self._parse_xlsx
        elif parser == "mgf":
            return self._parse_mgf
        else:
            raise ValueError(parser)

    def _parse_xlsx(self, file):
        xls = pd.ExcelFile(file)
        output_dict = {}
        for sheet in xls.sheet_names:
            output_dict[sheet] = pd.read_excel(file, sheet_name=sheet, index_col=None)
        return output_dict

    def _parse_csv(self, file):
        # df = pd.read_csv(file, comment='#', na_values=1 | 0)
        # NTAW-158: AC 6/10/2024
        df = pd.read_csv(file, comment="#", na_values=1 | 0, encoding="ISO-8859-1")
        self._update_headers(df)
        return df

    def _parse_tsv(self, file):
        df = pd.read_csv(file, sep="\t", comment="#", na_values=1 | 0)
        self._update_headers(df)
        return df

    def _update_headers(self, df):
        return df
