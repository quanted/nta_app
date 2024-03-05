from django.urls import path
from django.conf.urls import include
from django.conf import settings
from nta_app.views.ms1 import input as ms1_input
from nta_app.views.ms1 import processing as ms1_processing
from nta_app.views.ms1 import output as ms1_output
from nta_app.views.ms1 import results_api as ms1_results_api
from nta_app.views.ms1 import formula_list as ms1_formulas
from nta_app.views.ms1 import ms1_test_files as ms1_test_files
from nta_app.views.ms1 import decision_tree as ms1_decision_tree
from nta_app.views.ms2 import ms2_test_files as ms2_test_files
from nta_app.views.ms2 import upload as ms2_upload
from nta_app.views.ms2 import processing as ms2_processing
from nta_app.views.ms2 import output as ms2_output
from nta_app.views.ms2 import results_api as ms2_results_api
from nta_app.views.merge import input as merge_input
from nta_app.views.merge import output as merge_output
from nta_app.views.merge import processing as merge_processing
from nta_app.views.merge import results_api as merge_results_api
from nta_app.views.data_handler import data_api as data_api
from nta_app.views.misc import github
import nta_app.login_middleware as login_middleware
import os

print("qed.nta_app.urls")

urlpatterns = [
    #
    # ms1 tool
    path("", ms1_input.input_page),
    path("ms1", ms1_input.input_page),
    path("ms1/", ms1_input.input_page),
    path("ms1/input/", ms1_input.input_page),
    path("ms1/processing/<slug:jobid>", ms1_processing.processing_page),
    path("ms1/output/<slug:jobid>", ms1_output.output_page),
    path("ms1/results/toxpi/<slug:jobid>", ms1_results_api.download_toxpi),
    path("ms1/results/all/<slug:jobid>", ms1_results_api.download_all),
    path("ms1/status/<slug:jobid>", ms1_results_api.check_status),
    path("ms1/formulas/", ms1_formulas.formula_list_page),
    path("ms1/ms1_test_files/", ms1_test_files.test_files_page),
    path("ms1/formulas/download", ms1_formulas.download_msready_formulas),
    path("ms1/ms1_test_files/download", ms1_test_files.download_test_files),
    path(
        "ms1/results/decision_tree/<slug:jobid>", ms1_decision_tree.decision_tree_page
    ),
    path(
        "ms1/results/decision_tree_data/<slug:jobid>",
        ms1_results_api.decision_tree_data,
    ),
    path("github/", github),
    #
    # ms2 tool
    path("ms2", ms2_upload.upload_page),
    path("ms2/", ms2_upload.upload_page),
    path("ms2/<slug:jobid>", ms2_upload.upload_page_job),
    path("ms2/upload", ms2_upload.upload_page),
    path("ms2/upload/", ms2_upload.upload_page),
    path("ms2/upload/<slug:jobid>", ms2_upload.upload_page_job),
    path("ms2/output/<slug:jobid>", ms2_output.output_page, name="ms2_results"),
    path("ms2/processing/<slug:jobid>", ms2_processing.processing_page),
    path("ms2/results/<slug:jobid>", ms2_results_api.download_all),
    path("ms2/status/<slug:jobid>", ms2_results_api.check_status),
    path("ms2/ms2_test_files/", ms2_test_files.test_files_page),
    path("ms2/ms2_test_files", ms2_test_files.test_files_page),
    path("ms2/ms2_test_files/download", ms2_test_files.download_test_files),
    #
    # merge tool
    path("merge", merge_input.input_page),
    path("merge/input/", merge_input.input_page),
    path("merge/output/<slug:jobid>", merge_output.output_page, name="merge_results"),
    path("merge/processing/<slug:jobid>", merge_processing.processing_page),
    path("merge/results/<slug:jobid>", merge_results_api.download_merged),
    path("merge/status/<slug:jobid>", merge_results_api.check_status),
    #
    # file uploads
    path("files", data_api.get_files),
    path("upload", data_api.upload_api),
    path("upload/", data_api.upload_api),
    path("delete/", data_api.delete_api),
]

# Login requirement set url
if settings.LOGIN_REQUIRED:
    urlpatterns.append(path("login/", login_middleware.login))

urlpatterns = [
    path("nta/", include(urlpatterns)),
    path("", ms1_input.input_page),
]  # this does not appear to be functional


# 404 Error view (file not found)
# handler404 = misc.file_not_found
# 500 Error view (server error)
# handler500 = misc.file_not_found
# 403 Error view (forbidden)
# handler403 = misc.file_not_found
