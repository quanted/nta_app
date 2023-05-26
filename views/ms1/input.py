import os
import string, random

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri


from .. import links_left
from ...tools.ms1 import file_manager
from .input_form import NtaInputs
from ...app.ms1.nta_task import run_nta_dask

example_pos_filename = 'pooled_blood_pos_MPP.csv'
example_neg_filename = 'pooled_blood_neg_MPP.csv'
example_tracer_filename = 'pooled_blood_tracers.csv'
example_run_sequence_pos_filename = 'pooled_blood_run_sequence_pos.csv'
example_run_sequence_neg_filename = 'pooled_blood_run_sequence_neg.csv'

def input_page(request, form_data=None, form_files=None):

    model = 'ms1'
    header = "Run NTA MS1 Tool"
    page = 'run_model'
    if (request.method == "POST"):
            form = NtaInputs(request.POST, request.FILES)
            inputs_required = request.POST['test_files'] == 'no'
            form.fields['pos_input'].required = inputs_required
            form.fields['neg_input'].required = inputs_required
            print("Inputs required: {}".format(inputs_required))
            if inputs_required:
 
                form.fields['pos_input'].required = False
                form.fields['neg_input'].required = False
                if 'pos_input' in request.FILES.keys():
                    form.fields['pos_input'].required = True
                if 'neg_input' in request.FILES.keys():
                    form.fields['neg_input'].required = True

            if (form.is_valid()):
                print("form is valid")
                parameters = request.POST
                parameters = parameters.dict()
                job_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
                print("job ID: " + job_id)
                if parameters['test_files'] == 'yes':
                    example_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','input/ms1')
                    pos_input = os.path.join(example_data_dir, example_pos_filename)
                    neg_input = os.path.join(example_data_dir, example_neg_filename)
                    tracer_file = os.path.join(example_data_dir, example_tracer_filename)
                    tracer_df = file_manager.tracer_handler(tracer_file)
                    run_sequence_pos_file = os.path.join(example_data_dir, example_run_sequence_pos_filename)
                    run_sequence_pos_df = file_manager.tracer_handler(run_sequence_pos_file)
                    run_sequence_neg_file = os.path.join(example_data_dir, example_run_sequence_neg_filename)
                    run_sequence_neg_df = file_manager.tracer_handler(run_sequence_neg_file)
                else:
                    if 'pos_input' in request.FILES.keys():
                        pos_input = request.FILES["pos_input"]
                    else:
                        pos_input = None
                    if 'neg_input' in request.FILES.keys():
                        neg_input = request.FILES["neg_input"]
                    else:
                        neg_input = None
                    try:
                        tracer_file = request.FILES["tracer_input"]
                        tracer_df = file_manager.tracer_handler(tracer_file)
                    except Exception:
                        tracer_df = None
                    try:
                        run_sequence_pos_file = request.FILES["run_sequence_pos_file"]
                        run_sequence_pos_df = file_manager.tracer_handler(run_sequence_pos_file)
                    except Exception:
                        run_sequence_pos_df = None
                    try:
                        run_sequence_neg_file = request.FILES["run_sequence_neg_file"]
                        run_sequence_neg_df = file_manager.tracer_handler(run_sequence_neg_file)
                    except Exception:
                        run_sequence_neg_df = None
                inputs = [pos_input, neg_input]
                print("len(inputs)= ", len(inputs) )
                input_dfs = []
                for index, df in enumerate(inputs) :
                    print('indx=',index)
                    if df is not None:
                        input_dfs.append(file_manager.input_handler(df, index))
                    else:
                        input_dfs.append(None)
                # input_dfs = [file_manager.input_handler(df, index) for index, df in enumerate(inputs) if df is not None]
                print("len(input_dfs)= ", len(input_dfs) )
                run_nta_dask(parameters, input_dfs, tracer_df, run_sequence_pos_df, run_sequence_neg_df, job_id)
                return redirect('/nta/ms1/processing/'+job_id, permanent=True)
            else:
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

