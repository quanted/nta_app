import importlib
import os

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri

from . import links_left
from .input_form import NtaInputs

def input_page(request, form_data=None):

    model = 'nta'
    header = "Run NTA"
    page = 'run_model'
    if (request.method == "POST"):
        form = NtaInputs(request.POST)
        if (form.is_valid()):
            print("form is valid")
            return HttpResponseTemporaryRedirect('output/')
        else:
            form_data = request.POST

    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    # function name example: 'sip_input_page'
    html += render_to_string('04uberinput_jquery.html', {'model': model})
    html += render_to_string('nta_input_start_drupal.html', {
        'MODEL': model,
        'TITLE': header},
         request=request)

    html += str(NtaInputs(form_data))
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

