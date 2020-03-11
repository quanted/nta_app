from django.urls import path
from .views.ms1 import input as ms1_input
from .views.ms1 import processing as ms1_processing
from .views.ms1 import output as ms1_output
from .views.ms1 import results_api as ms1_results_api
from .views.ms1 import algorithms as ms1_algorithms
from .views.ms1 import qaqc as ms1_qaqc
from .views.ms1 import references as ms1_references
from .views.ms2 import input as ms2_input
from .views.ms2 import output as ms2_output
from .views.ms2 import algorithms as ms2_algorithms
from .views.ms2 import qaqc as ms2_qaqc
from .views.ms2 import references as ms2_references
from .views.ms2 import processing as ms2_processing
from .views.misc import github

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
    path('ms2/output/<slug:jobid>', ms2_output.output_page),
    path('ms2/algorithms/', ms2_algorithms.algorithms_page),
    path('ms2/qaqc/', ms2_qaqc.qaqcd_page),
    path('ms2/references/', ms2_references.references_page),
    path('ms2/processing/<slug:jobid>', ms2_processing.processing_page),
    path('ms2/results/<slug:jobid>', ms2_processing.processing_page, name='ms2_results'),

]

# 404 Error view (file not found)
#handler404 = misc.file_not_found
# 500 Error view (server error)
#handler500 = misc.file_not_found
# 403 Error view (forbidden)
#handler403 = misc.file_not_found
