import os
import string, random
import datetime

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri

from .input_form import MS2ParametersInput
from ...app.ms2.ms2_task import run_ms2_dask


from .. import links_left

example_pos_filename_1 = 'EntactEnv_Pos_MS1_Dust1IDA_01_Debug.mgf'
example_neg_filename_1 = 'EntactEnv_Neg_MS1_Dust1IDA_01_Debug.mgf'


def upload_page(request, form_data=None, form_files=None):
    print("Entered upload_page")

    model = 'ms2'
    header = "Run MS2 CFMID Tool"
    page = 'run_model'

    # generate a timestamp with the current time and date
    current_datetime = datetime.datetime.now()

    # define inputParameters dictionary containing all the parameters and their attributes, labels, and initial values
    inputParameters = {'project_name': ['Project Name', None],
    'datetime': ['Date & Time', str(current_datetime)],
    'csrfmiddlewaretoken': ['csrfmiddlewaretoken', None],
    'jobID': ['jobID', None],
    'fileUpload': ['fileUpload', []],
    'test_files': ['test_files', None],
    'precursor_mass_accuracy': ['precursor_mass_accuracy', None],
    'fragment_mass_accuracy': ['fragment_mass_accuracy', None],
    'classification_file': ['classification_file', None]
    }
    print("views/ms2/upload.py: inputParameters: {} ".format(inputParameters))

    if (request.method == "POST"):
        form = MS2ParametersInput(request.POST)
        # if request.POST['test_files'] == 'no':
        #     print('no test files')
        # else:
        #     print('test files Selected')

        if form.is_valid():
            if request.POST['test_files'] != 'no':
                print('test files Selected')
            else:
                print('test files NOT Selected')


            parameters = request.POST
            # print('1. parameters: {}'.format(parameters))
            parameters = parameters.dict()
            # print('2. parameters: {}'.format(parameters))
            # print("3. request.FILES.keys: {}".format(request.FILES.keys()))
            # print("4. request.FILES: {}".format(request.FILES))
            # Assuming request.FILES['fileUpload'] is a list of InMemoryUploadedFile objects
            # uploadedFiles = []
            file_uploads = request.FILES.getlist('fileUpload')
            for uploaded_file in file_uploads:
                file_name = uploaded_file.name
                # uploadedFiles.append(file_name)
                inputParameters['fileUpload'][1].append(file_name)
                # print(f"File Name: {file_name}")

            # print("views/ms2/upload.py: inputParameters: {} ".format(inputParameters))

            # print("===>")
            # loop over FILES.keys() and print each file name
            # for key in request.FILES.keys():
            #     print("key: {}".format(key))
            #     print("request.FILES[key]: {}".format(request.FILES[key]))
            #     print("request.FILES[key].name: {}".format(request.FILES[key].name))
                
            job_id = parameters['jobID']
            inputParameters['jobID'][1] = job_id
            inputParameters['project_name'][1] = parameters['project_name']
            inputParameters['precursor_mass_accuracy'][1] = parameters['precursor_mass_accuracy']
            inputParameters['fragment_mass_accuracy'][1] = parameters['fragment_mass_accuracy']
            inputParameters['classification_file'][1] = parameters['classification_file']
            inputParameters['test_files'][1] = parameters['test_files']
            inputParameters['csrfmiddlewaretoken'][1] = parameters['csrfmiddlewaretoken']
            parameters['inputParameters'] = inputParameters
            print('Final parameters: {}'.format(parameters))
            run_ms2_dask(parameters, job_id)
            return redirect('/nta/ms2/processing/'+job_id, permanent=True)
        else:
            form_data = request.POST

    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    # function name example: 'sip_input_page'
    html += render_to_string('ms2/nta_input_scripts.html')
    html += render_to_string('ms2/nta_input_css.html')
    html += render_to_string('ms2/ms2_upload_start_drupal.html', {
        'MODEL': model,
        'TITLE': header},
         request=request)
    
    html += render_to_string('ms2/ms2_file_upload.html',{})
    html += str(MS2ParametersInput(form_data))
    html += render_to_string('nta_input_form_end.html', {})
    
    html += links_left.ordered_list(model, page)

    # css and scripts
    html += render_to_string('nta_scripts_css.html', {})

    # epa template footer
    html += render_to_string('10epa_drupal_footer.html', {})

    response = HttpResponse()
    response.write(html)
    return response

def upload_page_job(request, jobid = "000000", form_data=None, form_files=None):
    
    model = 'ms2'
    header = "Run MS2 CFMID Tool"
    page = 'run_model'
    if (request.method == "POST"):
        parameters = request.POST
        parameters = parameters.dict()
        job_id = parameters['jobID']
        return redirect('/nta/ms2/parameters/'+job_id, permanent=True)

    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    # function name example: 'sip_input_page'
    html += render_to_string('ms2/nta_input_scripts.html')
    html += render_to_string('ms2/nta_input_css.html')

    html += render_to_string('ms2/ms2_upload_start_drupal.html', {
        'MODEL': model,
        'TITLE': "Upload MS2 Data"},
         request=request)    
    
    html += render_to_string('ms2/ms2_file_upload.html',{})
    html += str(MS2ParametersInput(form_data))
    html += render_to_string('04uberinput_end_drupal.html', {})
    html += render_to_string('04ubertext_end_drupal.html', {})    
    
    html += links_left.ordered_list(model, page)

    # css and scripts
    html += render_to_string('nta_scripts_css.html', {})

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

