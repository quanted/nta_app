import os
from django.http import HttpResponse
from django.template.loader import render_to_string
from .. import links_left

#@require_POST
def references_page(request, model='ms2', header='NTA', jobid='00000000'):
    header = "NTA"
    model = "ms2"
    model_html = '<div id="soon">Coming soon...</div>'
    html = references_page_html(header, model, model_html)
    response = HttpResponse()
    response.write(html)
    #print(html)
    return response


def references_page_html(header, model, tables_html):
    """Generates HTML to fill '.articles_output' div on output page"""
    page = 'references'
    #epa template header
    html = render_to_string('01epa_drupal_header.html', {
        'SITE_SKIN': os.environ['SITE_SKIN'],
        'TITLE': u"\u00FCbertool"
    })
    html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    html += render_to_string('epa_drupal_section_title_nta.html', {})

    #main body
    html += render_to_string('nta_main_content.html', {
        'TITLE': header + ' References',
        'TEXT_PARAGRAPH': tables_html
    })
    html += links_left.ordered_list(model, page)

    #css and scripts
    html += render_to_string('09epa_drupal_pram_css.html', {})
    html += render_to_string('09epa_drupal_pram_scripts.html', {})
    #html += render_to_string('09epa_drupal_pram_scripts.html', {})

    #epa template footer
    html += render_to_string('10epa_drupal_footer.html', {})
    return html
