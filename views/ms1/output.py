import os

from django.http import HttpResponse
from django.template.loader import render_to_string

from .. import links_left


# @require_POST
def output_page(request, model="nta", header="NTA", jobid="00000000"):
    header = "NTA"
    model = "ms1"
    model_output_html = "<h3> Job ID: " + jobid + "</h3> <br>"

    # this is where the func to generate output html will be called
    model_output_html += file_download_buttons(jobid)
    model_output_html += (
        '<div id="except_info"></div>'  # if there is an error, exception info will be placed here by the js script
    )
    html = output_page_html(header, model, model_output_html)
    response = HttpResponse()
    response.write(html)
    # print(html)
    return response


def output_page_html(header, model, tables_html):
    """Generates HTML to fill '.articles_output' div on output page"""

    # epa template header
    html = render_to_string(
        "01epa_drupal_header.html", {"SITE_SKIN": os.environ["SITE_SKIN"], "TITLE": "\u00FCbertool"}
    )
    html += render_to_string("02epa_drupal_header_bluestripe_onesidebar.html", {})
    html += render_to_string("epa_drupal_section_title_nta.html", {})

    # main body
    html += render_to_string("nta_main_content.html", {"TITLE": header + " Output", "TEXT_PARAGRAPH": tables_html})
    html += links_left.ordered_list(model)

    # css and scripts
    html += render_to_string("nta_scripts_css.html", {})
    html += render_to_string("ms1/nta_output_scripts.html", {})

    # epa template footer
    html += render_to_string("10epa_drupal_footer.html", {})
    return html


def file_download_buttons(jobid):
    html = """
    <div id="download_area">
        <H3 id="section1">Download results:</H3>
        <div class="buttons">
            <input type="button" value="Final Results" onclick="window.open('/nta/ms1/results/toxpi/{jobid}')">
        </div>
    </div>
    <div id="decision_tree_area">
        <H3 id="section1">View plots:</H3>
        <div class="buttons">
            <input type="button" value="Filtering Decision Tree" onclick="window.open('/nta/ms1/results/decision_tree/{jobid}')">
        </div>
    </div>
    """
    return html.format(jobid=jobid)
