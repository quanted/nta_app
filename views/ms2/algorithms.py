import os
from django.http import HttpResponse
from django.template.loader import render_to_string
from .. import links_left

#@require_POST
def algorithms_page(request, model='nta', header='NTA', jobid='00000000'):
    header = "NTA"
    model = "ms2"
    model_html = '<div id="soon">Coming soon...</div>'
    html = algorithms_page_html(header, model, model_html)
    response = HttpResponse()
    response.write(html)
    #print(html)
    return response


def algorithms_page_html(header, model, tables_html):

    page = 'algorithms'

    #epa template header
    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    #main body
    html += render_to_string('06ubertext_start_index_drupal.html', {
        'TITLE': header + ' Algorithms',
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
