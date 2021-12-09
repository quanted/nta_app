import os
import logging
import requests
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string
from .. import links_left

DSSTOX_API = os.environ.get('UBERTOOL_REST_SERVER')

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.views")
logger.setLevel(logging.INFO)

#@require_POST
def formula_list_page(request, model='ms1', header='Download MS-ready formula list', jobid='00000000'):
    model_html = """<div id="Download button"><input type="button" value="Download MS-ready formulas" onclick="window.open('download')">
</div>"""
    html = formula_list_html(header, model, model_html)
    response = HttpResponse()
    response.write(html)
    #print(html)
    return response


def formula_list_html(header, model, tables_html):
    """Generates HTML to fill '.articles_output' div on output page"""
    page = 'formula_list'
    #epa template header
    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    #main body
    html += render_to_string('06ubertext_start_index_drupal.html', {
        'TITLE': header + ' References',
        'TEXT_PARAGRAPH': tables_html
    })
    html += render_to_string('07ubertext_end_drupal.html', {})
    html += links_left.ordered_list(model, page)

    #css and scripts
    html += render_to_string('09epa_drupal_pram_css.html', {})
    html += render_to_string('09epa_drupal_pram_scripts.html', {})
    #html += render_to_string('09epa_drupal_pram_scripts.html', {})

    #epa template footer
    html += render_to_string('10epa_drupal_footer.html', {})
    return html

def download_msready_formulas(request):
    logger.info("=========== calling DSSTOX REST API for formula list")
    api_url = '{}/rest/ms1/list'.format(DSSTOX_API)
    logger.info(api_url)
    #http_headers = {'Content-Type': 'application/json'}
    response = requests.get(api_url)
    response_json = response.json()
    return JsonResponse(response_json)