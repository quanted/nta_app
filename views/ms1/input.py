import os
import string, random
import datetime

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri


from .. import links_left
from ...tools.ms1 import file_manager
from .input_form import NtaInputs
from ...app.ms1.nta_task import run_nta_dask

# hard-coded example file names for testing found in nta_app/input/ms1/
example_pos_filename = 'pooled_blood_pos_MPP.csv'
example_neg_filename = 'pooled_blood_neg_MPP.csv'
example_tracer_filename = 'pooled_blood_tracers.csv'
example_run_sequence_pos_filename = 'pooled_blood_run_sequence_pos.csv'
example_run_sequence_neg_filename = 'pooled_blood_run_sequence_neg.csv'

def input_page(request, form_data=None, form_files=None):

    model = 'ms1'
    header = "Run NTA MS1 Tool"
    page = 'run_model'

    # generate a timestamp with the current time and date
    current_datetime = datetime.datetime.now()

    # define inputParameters dictionary containing all the parameters and their attributes, labels, and initial values
    inputParameters = {'project_name': ['Project Name', None],
    'datetime': ['Date & Time', str(current_datetime)],
    'test_files': ['Run test files only (debugging)', None],
    'pos_input': ['Positive MPP file (csv)', None],
    'neg_input': ['Negative MPP file (csv)', None],
    'mass_accuracy_units': ['Adduct mass accuracy units', None],
    'mass_accuracy': ['Adduct mass accuracy', None],
    'rt_accuracy': ['Adduct retention time accuracy (mins)', None],
    'run_sequence_pos_file': ['Run sequence positive mode file (csv; optional)', None],
    'run_sequence_neg_file': ['Run sequence negative mode file (csv; optional)', None],
    'tracer_input': ['Tracer file (csv; optional)', None],
    'mass_accuracy_units_tr': ['Tracer mass accuracy units', None],
    'mass_accuracy_tr': ['Tracer mass accuracy', None],
    'rt_accuracy_tr': ['Tracer retention time accuracy (mins)', None],
    'min_replicate_hits': ['Min replicate hits', None],
    'min_replicate_hits_blanks': ['Min replicate hits in blanks', None],
    'max_replicate_cv': ['Max replicate CV', None],
    'parent_ion_mass_accuracy': ['Parent ion mass accuracy (ppm)', None],
    'minimum_rt': ['Discard features below this retention time (mins)', None],
    'search_dsstox': ['Search DSSTox for possible structures', None],
    'search_hcd': ['Search Hazard Comparison Dashboard for toxicity data', None],
    'search_mode': ['Search dashboard by', None],
    'top_result_only': ['Save top result only?', None],
    'api_batch_size': ['DSSTox search batch size (debugging)', None]
    }
    print("input_page: inputParameters: {} ".format(inputParameters))

    if (request.method == "POST"):
            print("POST: {}".format(request.POST))

            # the form data is sent as a combination of two types of data: POST data and FILES data. The 
            # POST data contains the form fields' values, while the FILES data contains any uploaded files.In 
            # order to handle both types of data, you need to pass them to the form's constructor. Django 
            # provides a convenient way to do this by using the request.POST and request.FILES dictionaries. 
            # By passing these dictionaries as parameters to the form's constructor, Django automatically 
            # populates the form fields with the submitted data.
            form = NtaInputs(request.POST, request.FILES)

            # if input 'test_files' is 'no', then the user has not selected to run the test files and at 
            # least one input file is required
            if request.POST['test_files'] == 'no':
                form.fields['pos_input'].required = True
                form.fields['neg_input'].required = True
                if 'pos_input' in request.FILES.keys():
                    # since the 'pos_input' file is present, the 'neg_input' file is not required
                    form.fields['neg_input'].required = False
                if 'neg_input' in request.FILES.keys():
                    # since the 'neg_input' file is present, the 'pos_input' file is not required
                    form.fields['pos_input'].required = False

            if (form.is_valid()):
                print("form is valid")

                # get parameters from the Request object. Note that the parameters are in the form of a QueryDict.
                parameters = request.POST
                print("1. parameters: {}".format(parameters))
                parameters = parameters.dict()
                print("2. parameters: {}".format(parameters))

                # get the uploaded files from the Request object. Note that the files are in the form of a
                # MultiValueDict. The MultiValueDict is a subclass of the standard Python dictionary that
                # provides a multiple values for the same key. This is necessary because some HTML form elements,
                # such as <select multiple>, pass multiple values for the same key.
                print("3. request.FILES.keys: {}".format(request.FILES.keys()))
                # loop through request.FILES and print out the keys and values
                for key, value in request.FILES.items():
                    print("key: {}".format(key))
                    print("value: {}".format(value))
                    # parameters[key] = value

                # save the Request parameters in the inputParameters dictionary [0] is the label, [1] is the value
                # This does not inclute the uploaded files, pos_input, neg_input, run_sequence_pos_file,
                # run_sequence_neg_file, and tracer_input, which are handled separately    
                inputParameters['project_name'][1] = parameters['project_name']
                inputParameters['test_files'][1] = parameters['test_files']
                inputParameters['mass_accuracy_units'][1] = parameters['mass_accuracy_units']
                inputParameters['mass_accuracy'][1] = parameters['mass_accuracy']
                inputParameters['rt_accuracy'][1] = parameters['rt_accuracy']
                inputParameters['mass_accuracy_units_tr'][1] = parameters['mass_accuracy_units_tr']
                inputParameters['mass_accuracy_tr'][1] = parameters['mass_accuracy_tr']
                inputParameters['rt_accuracy_tr'][1] = parameters['rt_accuracy_tr']
                inputParameters['min_replicate_hits'][1] = parameters['min_replicate_hits']
                inputParameters['min_replicate_hits_blanks'][1] = parameters['min_replicate_hits_blanks']
                inputParameters['max_replicate_cv'][1] = parameters['max_replicate_cv']
                inputParameters['parent_ion_mass_accuracy'][1] = parameters['parent_ion_mass_accuracy']
                inputParameters['minimum_rt'][1] = parameters['minimum_rt']
                inputParameters['search_dsstox'][1] = parameters['search_dsstox']
                inputParameters['search_hcd'][1] = parameters['search_hcd']
                inputParameters['search_mode'][1] = parameters['search_mode']
                inputParameters['top_result_only'][1] = parameters['top_result_only']
                inputParameters['api_batch_size'][1] = parameters['api_batch_size']

                # two basic scenarios are possible: 1) the user has selected to run the test files, or 2) the user
                # has not selected to run the test files. If the user has selected to run the test files, then the
                # test files will be used as the input files. If the user has not selected to run the test files,
                # then the user must upload the desired input files. In this case, the uploaded files found in
                # request.FILES will be used as the input files.
                if parameters['test_files'] == 'yes':
                    # handle case 1: the user has selected to run the test files
                    # get the path and filename of the test files
                    example_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','input/ms1')
                    pos_input = os.path.join(example_data_dir, example_pos_filename)
                    neg_input = os.path.join(example_data_dir, example_neg_filename)
                    tracer_file = os.path.join(example_data_dir, example_tracer_filename)
                    run_sequence_pos_file = os.path.join(example_data_dir, example_run_sequence_pos_filename)
                    run_sequence_neg_file = os.path.join(example_data_dir, example_run_sequence_neg_filename)
                    # save the name of the files to the inputParameters dictionary
                    inputParameters['pos_input'][1] = pos_input
                    inputParameters['neg_input'][1] = neg_input
                    inputParameters['tracer_input'][1] = tracer_file
                    inputParameters['run_sequence_pos_file'][1] = run_sequence_pos_file
                    inputParameters['run_sequence_neg_file'][1] = run_sequence_neg_file
                    # read the test files into pandas dataframes. Note: pos_input and neg_input are loaded later
                    # in the code
                    tracer_df = file_manager.tracer_handler(tracer_file)
                    run_sequence_pos_df = file_manager.tracer_handler(run_sequence_pos_file)
                    run_sequence_neg_df = file_manager.tracer_handler(run_sequence_neg_file)
                else:
                    # handle case 2: the user has not selected to run the test files
                    if 'pos_input' in request.FILES.keys():
                        pos_input = request.FILES["pos_input"]

                        # save the name of the file to the inputParameters dictionary
                        inputParameters['pos_input'][1] = pos_input.name
                    else:
                        pos_input = None

                    if 'neg_input' in request.FILES.keys():
                        neg_input = request.FILES["neg_input"]
                         # save the name of the file to the inputParameters dictionary
                        inputParameters['neg_input'][1] = neg_input.name
                    else:
                        neg_input = None

                    try:
                        tracer_file = request.FILES["tracer_input"]
                        tracer_df = file_manager.tracer_handler(tracer_file)
                        # save the name of the file to the inputParameters dictionary
                        inputParameters['tracer_input'][1] = tracer_file.name
                    except Exception:
                        tracer_df = None

                    try:
                        run_sequence_pos_file = request.FILES["run_sequence_pos_file"]
                        run_sequence_pos_df = file_manager.tracer_handler(run_sequence_pos_file)
                        # save the name of the file to the inputParameters dictionary
                        inputParameters['run_sequence_pos_file'][1] = run_sequence_pos_file.name
                    except Exception:
                        run_sequence_pos_df = None

                    try:
                        run_sequence_neg_file = request.FILES["run_sequence_neg_file"]
                        run_sequence_neg_df = file_manager.tracer_handler(run_sequence_neg_file)
                        # save the name of the file to the inputParameters dictionary
                        inputParameters['run_sequence_neg_file'][1] = run_sequence_neg_file.name
                    except Exception:
                        run_sequence_neg_df = None

                # create a list of the input files
                inputs = [pos_input, neg_input]
                print("len(inputs)= ", len(inputs) )
                print("inputs: {} ".format(inputs))
                input_dfs = []
                for index, df in enumerate(inputs) :
                    print('indx=',index)
                    if df is not None:
                        input_dfs.append(file_manager.input_handler(df, index))
                    else:
                        input_dfs.append(None)
                # input_dfs = [file_manager.input_handler(df, index) for index, df in enumerate(inputs) if df is not None]
                print("len(input_dfs)= ", len(input_dfs) )
                print("input_page: inputParameters: {} ".format(inputParameters))

                # create a job ID
                job_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                print("job ID: " + job_id)

                run_nta_dask(inputParameters, input_dfs, tracer_df, run_sequence_pos_df, run_sequence_neg_df, job_id)
                return redirect('/nta/ms1/processing/'+job_id, permanent=True)
            else:
                print("form is NOT valid")
                form_data = request.POST
                form_files = request.FILES

    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    # function name example: 'sip_input_page'
    html += render_to_string('ms1/nta_input_scripts.html')
    html += render_to_string('ms1/nta_input_css.html')
    html += render_to_string('ms1/nta_input_start_drupal.html', {
        'MODEL': model,
        'TITLE': header},
         request=request)

    html += str(NtaInputs(form_data, form_files))
    html += "**Disclaimer: This tool is being provided for internal testing purposes and is not yet approved by EPA-ORD. Please do not publish any outputs.**"
    html += "**Disclaimer: Input data is stored within this application. Be sure to remove all PII (Personal Identifiable Information) before submitting data.**"
    html += render_to_string('04uberinput_end_drupal.html', {})
    html += render_to_string('04ubertext_end_drupal.html', {})

    html += links_left.ordered_list(model, page)

    # css and scripts
    html += render_to_string('09epa_drupal_pram_css.html', {})
    html += render_to_string('09epa_drupal_pram_scripts.html', {})

    # epa template footer
    html += render_to_string('10epa_drupal_footer.html', {})

    response = HttpResponse()
    response.write(html)
    return response



class HttpResponseTemporaryRedirect(HttpResponse):
    status_code = 307

    def __init__(self, redirect_to):
        HttpResponse.__init__(self)
        self['Location'] = iri_to_uri(redirect_to)

