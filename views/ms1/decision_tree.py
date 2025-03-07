import os

from django.http import HttpResponse
from django.template.loader import render_to_string

from .. import links_left


# @require_POST
def decision_tree_page(request, model="nta", header="NTA", jobid="00000000"):
    header = "NTA"
    model = "ms1"
    # decision_tree_content_html = "<h3> Job ID: " + jobid + "</h3> <br>"
    # decision_tree_content_html += "<div class=decision_tree> </div>"
    decision_tree_content_html = "<div class=decision_tree> </div>"
    # decision_tree_content_html += {{ jobid|json_script:'jobid' }} # pass job id to the javascript
    decision_tree_content_html += render_to_string(
        "ms1/nta_decision_tree.html", {"jobid": jobid}
    )  # call the decision tree template
    # this is where the func to generate page html will be called
    html = decision_tree_page_html(header, model, decision_tree_content_html)
    response = HttpResponse()
    response.write(html)
    # print(html)
    return response


def decision_tree_page_html(header, model, decision_tree_html):
    """Generates HTML to fill '.articles_output' div on decision tree page"""

    # NTAW-561 : Turn off additional html for decision tree page

    # #epa template header
    # html = render_to_string('01epa_drupal_header.html', {
    #     'SITE_SKIN': os.environ['SITE_SKIN'],
    #     'TITLE': u"\u00FCbertool"
    # })
    # html += render_to_string('02epa_drupal_header_bluestripe_onesidebar.html', {})
    # html += render_to_string('epa_drupal_section_title_nta.html', {})

    # #main body
    # html += render_to_string('nta_main_content.html', {
    #     'TITLE': header + ' Output',
    #     'TEXT_PARAGRAPH': decision_tree_html
    # })
    # html += links_left.ordered_list(model)

    # #css and scripts
    # html += render_to_string('nta_scripts_css.html', {})
    # html += render_to_string('ms1/nta_output_scripts.html', {})

    # #epa template footer
    # html += render_to_string('10epa_drupal_footer.html', {})

    # NTAW-561: Sole code for loading decision tree html
    html = render_to_string(
        # "nta_main_content.html", {"TITLE": header + " Output", "TEXT_PARAGRAPH": decision_tree_html}
        "nta_main_content.html",
        {"TEXT_PARAGRAPH": decision_tree_html},
    )

    return html
