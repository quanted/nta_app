import os
import string, random

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri


from .. import links_left
from ...tools.merge import file_manager
from .input_form import MergeInputs
from ...app.merge.merge_task import run_merge_dask


def input_page(request, form_data=None, form_files=None):
    model = "Merge"
    header = "MS1 and MS2 merge workflow"
    page = "run_model"
    if request.method == "POST":
        form = MergeInputs(request.POST, request.FILES)
        if form.is_valid():
            print("form is valid")
            parameters = request.POST
            parameters = parameters.dict()
            job_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            print("job ID: " + job_id)
            fileParser = file_manager.FileParser()
            input_data = {"MS1": None, "MS2_pos": [], "MS2_neg": []}
            ms1_input = request.FILES.getlist("ms1_inputs")

            # NTAW-158 = AC 6/10/2024: Update parser to handle xlsx files. An xlsx file will create a dictionary with all the results sheets; only grab the chemical results sheet if it is a dict
            ms1_input_temp = fileParser.run(ms1_input[0])
            if isinstance(ms1_input_temp, dict):
                input_data["MS1"] = ms1_input_temp["chemical_results"]
            else:
                input_data["MS1"] = ms1_input_temp

            if bool(request.FILES.get("ms2_pos_inputs", False)) == True:
                ms2_pos_input = request.FILES.getlist("ms2_pos_inputs")
                pos_input_list = ms2_pos_input if type(ms2_pos_input) in [list, tuple] else [ms2_pos_input]
                input_data["MS2_pos"] = [
                    {"file_name": file.name, "file_df": fileParser.run(file)} for file in pos_input_list if file
                ]

            if bool(request.FILES.get("ms2_neg_inputs", False)) == True:
                ms2_neg_input = request.FILES.getlist("ms2_neg_inputs")
                neg_input_list = ms2_neg_input if type(ms2_neg_input) in [list, tuple] else [ms2_neg_input]
                input_data["MS2_neg"] = [
                    {"file_name": file.name, "file_df": fileParser.run(file)} for file in neg_input_list if file
                ]

            run_merge_dask(parameters, input_data, job_id)
            return redirect("/nta/merge/processing/" + job_id, permanent=True)
        else:
            form_data = request.POST
            form_files = request.FILES

    html = render_to_string(
        "01epa_drupal_header.html", {"SITE_SKIN": os.environ["SITE_SKIN"], "TITLE": "\u00FCbertool"}
    )
    html += render_to_string("02epa_drupal_header_bluestripe_onesidebar.html", {})
    html += render_to_string("epa_drupal_section_title_nta.html", {})

    # function name example: 'sip_input_page'
    html += render_to_string("merge/nta_input_scripts.html")
    html += render_to_string("merge/nta_input_css.html")
    html += render_to_string("merge/merge_input_start_drupal.html", {"MODEL": model, "TITLE": header}, request=request)

    html += str(MergeInputs(form_data, form_files))
    html += render_to_string("nta_input_form_end.html", {})

    html += links_left.ordered_list(model, page)

    # css and scripts
    html += render_to_string("nta_scripts_css.html", {})

    # epa template footer
    html += render_to_string("10epa_drupal_footer.html", {})

    response = HttpResponse()
    response.write(html)
    return response


class HttpResponseTemporaryRedirect(HttpResponse):
    status_code = 307

    def __init__(self, redirect_to):
        HttpResponse.__init__(self)
        self["Location"] = iri_to_uri(redirect_to)
