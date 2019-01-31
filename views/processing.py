import os

from django.http import HttpResponse
from django.template.loader import render_to_string

from . import links_left


#@require_POST
def processing_page(request, model='nta', header='NTA', jobid='00000000'):
    header = "NTA"
    model = "nta"
    model_output_html = '<div id="wait_gif"><img src="/static_qed/nta/loading1.gif" alt="Loading..."></div>'
    model_output_html += '<div id="jobid"> Job ID: {}</div>'.format(jobid)
    model_output_html += '<div id="status"> Processing... please wait. </div>' #this is where the func to generate output html will be called
    model_output_html += '<div id="except_info"></div>' #if there is an error, exception info will be placed here by the js script

    html = processing_page_html(header, model, model_output_html)
    response = HttpResponse()
    response.write(html)
    #print(html)
    return response


def processing_page_html(header, model, tables_html):
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
    html += render_to_string('09epa_drupal_pram_scripts.html', {})
    html += render_to_string('nta_processing_scripts.html')
    #html += render_to_string('09epa_drupal_pram_scripts.html', {})

    #epa template footer
    html += render_to_string('10epa_drupal_footer.html', {})
    return html
