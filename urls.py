from django.urls import path
from django.conf.urls import include
from nta_app.views.ms1 import input as ms1_input
from nta_app.views.ms1 import processing as ms1_processing
from nta_app.views.ms1 import output as ms1_output
from nta_app.views.ms1 import results_api as ms1_results_api
from nta_app.views.ms1 import algorithms as ms1_algorithms
from nta_app.views.ms1 import qaqc as ms1_qaqc
from nta_app.views.ms1 import references as ms1_references
from nta_app.views.ms2 import input as ms2_input
from nta_app.views.ms2 import output as ms2_output
from nta_app.views.ms2 import algorithms as ms2_algorithms
from nta_app.views.ms2 import qaqc as ms2_qaqc
from nta_app.views.ms2 import references as ms2_references
from nta_app.views.ms2 import processing as ms2_processing
from nta_app.views.ms2 import results_api as ms2_results_api
from nta_app.views.merge import input as merge_input
from nta_app.views.merge import output as merge_output
from nta_app.views.merge import algorithms as merge_algorithms
from nta_app.views.merge import qaqc as merge_qaqc
from nta_app.views.merge import references as merge_references
from nta_app.views.merge import processing as merge_processing
from nta_app.views.merge import results_api as merge_results_api
from nta_app.views.misc import github

print('qed.nta_app.urls')

urlpatterns = [
    #
    # ms1 tool
    path('', ms1_input.input_page),
    path('ms1', ms1_input.input_page),
    path('ms1/input/', ms1_input.input_page),
    path('ms1/processing/<slug:jobid>', ms1_processing.processing_page),
    path('ms1/output/<slug:jobid>', ms1_output.output_page),
    path('ms1/results/toxpi/<slug:jobid>', ms1_results_api.download_toxpi),
    path('ms1/results/all/<slug:jobid>', ms1_results_api.download_all),
    path('ms1/status/<slug:jobid>', ms1_results_api.check_status),
    path('ms1/algorithms/', ms1_algorithms.algorithms_page),
    path('ms1/qaqc/', ms1_qaqc.qaqcd_page),
    path('ms1/references/', ms1_references.references_page),
    path('github/', github),
    #
    # ms2 tool
    path('ms2', ms2_input.input_page),
    path('ms2/input/', ms2_input.input_page),
    path('ms2/output/<slug:jobid>', ms2_output.output_page, name='ms2_results'),
    path('ms2/algorithms/', ms2_algorithms.algorithms_page),
    path('ms2/qaqc/', ms2_qaqc.qaqcd_page),
    path('ms2/references/', ms2_references.references_page),
    path('ms2/processing/<slug:jobid>', ms2_processing.processing_page),
    path('ms2/results/<slug:jobid>', ms2_results_api.download_all),
    path('ms2/status/<slug:jobid>', ms2_results_api.check_status),
    #
    # merge tool
    path('merge', merge_input.input_page),
    path('merge/input/', merge_input.input_page),
    path('merge/output/<slug:jobid>', merge_output.output_page, name='merge_results'),
    path('merge/algorithms/', merge_algorithms.algorithms_page),
    path('merge/qaqc/', merge_qaqc.qaqcd_page),
    path('merge/references/', merge_references.references_page),
    path('merge/processing/<slug:jobid>', merge_processing.processing_page),
    path('merge/results/<slug:jobid>', merge_results_api.download_all),
    path('merge/status/<slug:jobid>', merge_results_api.check_status)
]

urlpatterns = [path(r'^nta/', include(urlpatterns))]

# 404 Error view (file not found)
#handler404 = misc.file_not_found
# 500 Error view (server error)
#handler500 = misc.file_not_found
# 403 Error view (forbidden)
#handler403 = misc.file_not_found
