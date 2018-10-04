import importlib
import os
import string, random

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri
from dask.distributed import Client


from . import links_left, processing
from ..tools import file_manager
from .input_form import NtaInputs
from ..app.nta_task import run_nta_dask

def input_page(request, form_data=None, form_files=None):

    model = 'nta'
    header = "Run NTA"
    page = 'run_model'
    if (request.method == "POST"):
        form = NtaInputs(request.POST, request.FILES)
        if (form.is_valid()):
            print("form is valid")
            parameters = request.POST
            job_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            print("job ID: " + job_id)
            pos_input = request.FILES["pos_input"]
            neg_input = request.FILES["neg_input"]
            try:
                tracer_file = request.FILES["tracer_input"]
                tracer_df = file_manager.tracer_handler(tracer_file)
            except:
                tracer_df = None
            inputs = [pos_input, neg_input]
            input_dfs = [file_manager.input_handler(df, index) for index, df in enumerate(inputs)]
            #print(input_dfs[0])
            run_nta_dask(parameters, input_dfs, tracer_df, job_id)
            #nta_run = NtaRun(parameters, input_dfs, tracer_df, job_id)
            #nta_run.execute()
            return redirect('/nta/processing/'+job_id, permanent=True)
            #return HttpResponseTemporaryRedirect('/nta/output/'+job_id)
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
    html += render_to_string('04uberinput_jquery.html', {'model': model})
    html += render_to_string('nta_input_scripts.html')
    html += render_to_string('nta_input_css.html')
    html += render_to_string('nta_input_start_drupal.html', {
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

