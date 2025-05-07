from nta_app.tests.app_ms1_test_helpers import inputParameters
from nta_app.app.constants import EXAMPLE_POS_FILENAME

def test__ensure_parameters_are_complete_for_tests():
  assert inputParameters["pos_input"][1] == EXAMPLE_POS_FILENAME
  assert inputParameters["test_files"][1] == "yes"