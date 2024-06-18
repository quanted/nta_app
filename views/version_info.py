import os
import logging
from io import BytesIO
from django.http import HttpResponse
from django.template.loader import render_to_string
from . import links_left


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.views")
logger.setLevel(logging.INFO)


def version_info_page(request, model="none", header="EPA NTA WebApp Version Info", jobid="00000000"):
    page_html = """
    <embed src="/nta/static/docs/NTA_WebApp_Version_History.txt" width="600" height="1000" type="text/plain">
    """
    html = version_info_html(header, model, page_html)
    response = HttpResponse()
    response.write(html)
    # print(html)
    return response


def version_info_html(header, model, tables_html):
    """Generates HTML to fill '.articles_output' div on output page"""
    page = "version_info"
    # epa template header
    html = render_to_string(
        "01epa_drupal_header.html",
        {"SITE_SKIN": os.environ["SITE_SKIN"], "TITLE": "\u00FCbertool"},
    )
    html += render_to_string("02epa_drupal_header_bluestripe_onesidebar.html", {})
    html += render_to_string("epa_drupal_section_title_nta.html", {})

    # main body
    html += render_to_string(
        "nta_main_content.html",
        {"TITLE": header, "TEXT_PARAGRAPH": tables_html},
    )
    html += links_left.ordered_list(model, page)

    # css and scripts
    html += render_to_string("nta_scripts_css.html", {})
    # html += render_to_string('09epa_drupal_pram_scripts.html', {})

    # epa template footer
    html += render_to_string("10epa_drupal_footer.html", {})
    return html
