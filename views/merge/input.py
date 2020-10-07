import os
import string, random

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri


from .. import links_left
from ...tools.ms2 import file_manager
from .input_form import MS2Inputs
from ...app.ms2.ms2_task import run_ms2_dask

def input_page(request, form_data=None, form_files=None):

    model = 'MS2'
    header = "Run MS2 CFMID Tool"
    page = 'run_model'
    if (request.method == "POST"):
        form = MS2Inputs(request.POST, request.FILES)
        if (form.is_valid()):
            print("form is valid")
            parameters = request.POST
            parameters = parameters.dict()
            job_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            print("job ID: " + job_id)
            pos_input = request.FILES.getlist("pos_inputs")
            neg_input = request.FILES.getlist("neg_inputs")
            pos_input_list = pos_input if type(pos_input) in [list, tuple] else [pos_input]
            neg_input_list = neg_input if type(neg_input) in [list, tuple] else [neg_input]
            input_dfs = [None,None]
            input_dfs[0] = [file_manager.parse_mgf(csv_file) for csv_file in pos_input_list if csv_file]
            input_dfs[1] = [file_manager.parse_mgf(csv_file) for csv_file in neg_input_list if csv_file]
            run_ms2_dask(parameters, input_dfs, job_id)
            return redirect('/nta/ms2/processing/'+job_id, permanent=True)
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
    html += render_to_string('ms2/nta_input_scripts.html')
    html += render_to_string('ms2/nta_input_css.html')
    html += render_to_string('ms2/ms2_input_start_drupal.html', {
        'MODEL': model,
        'TITLE': header},
         request=request)

    html += str(MS2Inputs(form_data, form_files))
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

