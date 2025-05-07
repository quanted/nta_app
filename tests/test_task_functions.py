import pandas as pd
import os
import pytest
from nta_app.tests.app_ms1_test_helpers import inputParameters
from nta_app.app.constants import EXAMPLE_NEG_FILENAME, EXAMPLE_POS_FILENAME
from nta_app.app.ms1.task_functions import duplicates, assign_feature_id, differences as count_string_differences, parse_headers, get_sample_and_blank_headers, passthrucol, window_size

data_dir = "input/ms1"
my_pos_df = pd.read_csv(os.path.join(data_dir, EXAMPLE_POS_FILENAME))
my_neg_df = pd.read_csv(os.path.join(data_dir, EXAMPLE_NEG_FILENAME))

def test__added_feature_id__new_column_for_feature_id():
    data = {
        "firstCol": [420, 380, 390],
        "secondCol": [50, 40, 45]
    }
    old_df = pd.DataFrame(data)
    assert "Feature ID" not in old_df.columns

    new_df = assign_feature_id(df_in=old_df, start=1)

    assert "Feature ID" in new_df.columns

def test__count_string_differences__for_basic_strings():
    assert count_string_differences(s1="ones", s2="one") == 2

def test__count_string_differences__for_strings_with_special_characters():
    assert count_string_differences(s1="o_nes", s2="o^nes") == 2

def test__parse_headers__returns_list_of_list_of_string(df=my_pos_df):
    result = parse_headers(df)
    assert len(result) == 23

def test__parse_headers__lists_contain_expected_items(df=my_pos_df):
    result = parse_headers(df)
    assert result[0][0] == "MB1"
    assert result[22][0] == "Ionization Mode"

def test__get_sample_and_blank_headers__returns_all_headers(pos_df=my_pos_df, neg_df=my_neg_df):
    assert len(get_sample_and_blank_headers((pos_df, neg_df))) == 3

def test__get_sample_and_blank_headers__fails_when_both_dfs_are_none(pos_df=my_pos_df, neg_df=my_neg_df):
    with pytest.raises(AttributeError):
        get_sample_and_blank_headers((None, None))

def test__get_sample_and_blank_headers__returns_correct_content(pos_df=my_pos_df, neg_df=my_neg_df):
    all_headers, blank_headers, sample_headers = get_sample_and_blank_headers((pos_df, neg_df))
    assert len(blank_headers[0]) == 5
    for sample_types in sample_headers:
        for sample in sample_types:
            assert not sample.startswith("MB")
    assert all_headers[-1][0] == "Ionization Mode"

def test__passthrucol__returns_passthrough_and_trimmed_df(pos_df=my_pos_df, neg_df=my_neg_df):
    pos_df = assign_feature_id(pos_df)
    all_headers = get_sample_and_blank_headers((pos_df, neg_df))[0]
    df_pt, df_trim = passthrucol(pos_df, all_headers)
    assert "m/z" in df_pt.columns.values
    assert "m/z" not in df_trim.columns.values

def test__window_size__default_mass_diff(df_in=my_pos_df):
    val = window_size(df_in)
    assert val == 1801

def test__window_size__supplied_mass_diff(df_in=my_pos_df):
    val = window_size(df_in, 100.00)
    assert val == 1597