import importlib
import os

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri
from django.views.decorators.http import require_POST

from . import links_left
from .input_form import NtaInputs


#@require_POST
def output_page(request, model='nta', header='NTA', jobid='00000000'):
    header = "NTA"
    model = "nta"
    model_output_html = "Processing... (3-5 minutes)" #this is where the func to generate output html will be called
    html = output_page_html(header, model, model_output_html)
    response = HttpResponse()
    response.write(html)
    #print(html)
    return response


def output_page_html(header, model, tables_html):
    """Generates HTML to fill '.articles_output' div on output page"""

    #epa template header
    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    #main body
    html += render_to_string('06ubertext_start_index_drupal.html', {
        'TITLE': header + ' Output',
        'TEXT_PARAGRAPH': tables_html
    })
    html += render_to_string('07ubertext_end_drupal.html', {})
    html += links_left.ordered_list(model)

    #css and scripts
    html += render_to_string('09epa_drupal_pram_css.html', {})
    #html += render_to_string('09epa_drupal_pram_scripts.html', {})

    #epa template footer
    html += render_to_string('10epa_drupal_footer.html', {})
    return html
