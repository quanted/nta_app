import os
import string, random
import datetime
import logging

from django.http import HttpResponse
from django.shortcuts import redirect
from django.template.loader import render_to_string
from django.utils.encoding import iri_to_uri
from django.views.decorators.csrf import csrf_exempt


from .. import links_left
from ...tools.ms1 import file_manager
from .input_form import NtaInputs
from ...app.ms1.nta_task import run_nta_dask

# set up logging
logger = logging.getLogger("nta_app.views.ms1")
if os.getenv("DEPLOY_ENV", "kube-dev") == "kube-prod":
    logger.setLevel(logging.WARNING)

# hard-coded example file names for testing found in nta_app/input/ms1/
example_pos_filename = "1a_MZmine3_pos.csv"
example_neg_filename = "1b_MZmine3_neg.csv"
example_tracer_filename = "WW2DW_Tracers_Amenable.csv"
example_run_sequence_pos_filename = "WW2DW_sequence_cal.csv"
example_run_sequence_neg_filename = "WW2DW_sequence_cal.csv"


@csrf_exempt
def input_page(request, form_data=None, form_files=None):
    model = "ms1"
    header = "Run NTA MS1 workflow"
    page = "run_model"

    # generate a timestamp with the current time and date
    current_datetime = datetime.datetime.now()

    # manually define current version of the WebApp
    current_version = "0.3.6"

    # define inputParameters dictionary containing all the parameters and their attributes, labels, and initial values
    inputParameters = {
        "project_name": ["Project name", None],
        "version": ["WebApp Version", current_version],
        "datetime": ["Date & time", str(current_datetime)],
        "test_files": ["Run test files only (debugging)", None],
        "pos_input": ["Positive mode file", None],
        "neg_input": ["Negative mode file", None],
        "pos_adducts": ["Positive mode adducts", None],
        "neg_adducts": ["Negative mode adducts", None],
        "neutral_losses": ["Neutral losses (both modes)", None],
        "mass_accuracy_units": ["Adduct / duplicate mass accuracy units", None],
        "mass_accuracy": ["Adduct / duplicate mass accuracy", None],
        "rt_accuracy": ["Adduct / duplicate retention time accuracy (mins)", None],
        "run_sequence_pos_file": [
            "Run sequence positive mode file",
            None,
        ],
        "run_sequence_neg_file": [
            "Run sequence negative mode file",
            None,
        ],
        "tracer_input": ["Tracer file", None],
        "mass_accuracy_units_tr": ["Tracer mass accuracy units", None],
        "mass_accuracy_tr": ["Tracer mass accuracy", None],
        "rt_accuracy_tr": ["Tracer retention time accuracy (mins)", None],
        "tracer_plot_yaxis_format": ["Tracer plot y-axis scaling", None],
        "tracer_plot_trendline": ["Tracer plot trendlines shown", None],
        "min_replicate_hits": ["Min replicate hits (%)", None],
        "min_replicate_hits_blanks": ["Min replicate hits in blanks (%)", None],
        "max_replicate_cv": ["Max replicate CV", None],
        "mrl_std_multiplier": ["MRL standard deviation multiplier", None],
        "parent_ion_mass_accuracy": ["Parent ion mass accuracy (ppm)", None],
        "minimum_rt": ["Discard features below this retention time (mins)", None],
        "search_dsstox": ["Search DSSTox for possible structures", None],
        "search_hcd": ["Search Cheminformatics Hazard Module for toxicity data", None],
        "search_mode": ["Search dashboard by", None],
    }
    logger.debug("input_page: inputParameters: {} ".format(inputParameters))

    if request.method == "POST":
        logger.debug("POST: {}".format(request.POST))

        # the form data is sent as a combination of two types of data: POST data and FILES data. The
        # POST data contains the form fields' values, while the FILES data contains any uploaded files.In
        # order to handle both types of data, you need to pass them to the form's constructor. Django
        # provides a convenient way to do this by using the request.POST and request.FILES dictionaries.
        # By passing these dictionaries as parameters to the form's constructor, Django automatically
        # populates the form fields with the submitted data.
        form = NtaInputs(request.POST, request.FILES)

        # if input 'test_files' is 'no', then the user has not selected to run the test files and at
        # least one input file is required
        if request.POST["test_files"] == "no":
            # Set requirement status to True
            form.fields["pos_input"].required = True
            form.fields["neg_input"].required = True
            # If 'pos_input' file is present, the 'neg_input' file is not required
            if "pos_input" in request.FILES.keys():
                form.fields["neg_input"].required = False
            # If 'neg_input' file is present, the 'pos_input' file is not required
            if "neg_input" in request.FILES.keys():
                form.fields["pos_input"].required = False

        if form.is_valid():
            logger.info("form is valid")

            # get parameters from the Request object. Note that the parameters are in the form of a QueryDict.
            parameters = request.POST
            parameters = parameters.dict()
            logger.info("parameters dict items: {}".format(parameters.items()))

            # get the uploaded files from the Request object. Note that the files are in the form of a
            # MultiValueDict. The MultiValueDict is a subclass of the standard Python dictionary that
            # provides a multiple values for the same key. This is necessary because some HTML form elements,
            # such as <select multiple>, pass multiple values for the same key.
            logger.debug("3. request.FILES.keys: {}".format(request.FILES.keys()))
            # loop through request.FILES and print out the keys and values
            for key, value in request.FILES.items():
                logger.debug("key: {}".format(key))
                logger.debug("value: {}".format(value))
                # parameters[key] = value

            # save the Request parameters in the inputParameters dictionary [0] is the label, [1] is the value
            # This does not include the uploaded files, pos_input, neg_input, run_sequence_pos_file,
            # run_sequence_neg_file, and tracer_input, which are handled separately
            inputParameters["project_name"][1] = parameters["project_name"]
            inputParameters["test_files"][1] = parameters["test_files"]
            inputParameters["mass_accuracy_units"][1] = parameters["mass_accuracy_units"]
            inputParameters["mass_accuracy"][1] = parameters["mass_accuracy"]
            inputParameters["rt_accuracy"][1] = parameters["rt_accuracy"]
            inputParameters["mass_accuracy_units_tr"][1] = parameters["mass_accuracy_units_tr"]
            inputParameters["mass_accuracy_tr"][1] = parameters["mass_accuracy_tr"]
            inputParameters["rt_accuracy_tr"][1] = parameters["rt_accuracy_tr"]
            inputParameters["tracer_plot_yaxis_format"][1] = parameters["tracer_plot_yaxis_format"]
            inputParameters["tracer_plot_trendline"][1] = parameters["tracer_plot_trendline"]
            inputParameters["min_replicate_hits"][1] = parameters["min_replicate_hits"]
            inputParameters["min_replicate_hits_blanks"][1] = parameters["min_replicate_hits_blanks"]
            inputParameters["max_replicate_cv"][1] = parameters["max_replicate_cv"]
            inputParameters["mrl_std_multiplier"][1] = parameters["mrl_std_multiplier"]
            inputParameters["parent_ion_mass_accuracy"][1] = parameters["parent_ion_mass_accuracy"]
            inputParameters["minimum_rt"][1] = parameters["minimum_rt"]
            inputParameters["search_dsstox"][1] = parameters["search_dsstox"]
            inputParameters["search_hcd"][1] = parameters["search_hcd"]
            inputParameters["search_mode"][1] = parameters["search_mode"]

            # Get user-selected adducts via POST.getlist()
            # inputParameters["pos_adducts"][1] = parameters["pos_adducts"]
            # logger.info("pos adducts list v2: {}".format(inputParameters["pos_adducts"][1]))
            inputParameters["pos_adducts"][1] = request.POST.getlist("pos_adducts")
            inputParameters["neg_adducts"][1] = request.POST.getlist("neg_adducts")
            inputParameters["neutral_losses"][1] = request.POST.getlist("neutral_losses")
            logger.info("pos adducts list: {}".format(inputParameters["pos_adducts"][1]))
            logger.info("neg adducts list: {}".format(inputParameters["neg_adducts"][1]))
            logger.info("neutral adducts list: {}".format(inputParameters["neutral_losses"][1]))

            # two basic scenarios are possible: 1) the user has selected to run the test files, or 2) the user
            # has not selected to run the test files. If the user has selected to run the test files, then the
            # test files will be used as the input files. If the user has not selected to run the test files,
            # then the user must upload the desired input files. In this case, the uploaded files found in
            # request.FILES will be used as the input files.
            if parameters["test_files"] == "yes":
                # handle case 1: the user has selected to run the test files
                # get the path and filename of the test files
                example_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "input/ms1")
                pos_input = os.path.join(example_data_dir, example_pos_filename)
                neg_input = os.path.join(example_data_dir, example_neg_filename)
                tracer_file = os.path.join(example_data_dir, example_tracer_filename)
                run_sequence_pos_file = os.path.join(example_data_dir, example_run_sequence_pos_filename)
                run_sequence_neg_file = os.path.join(example_data_dir, example_run_sequence_neg_filename)
                # save the name of the files to the inputParameters dictionary
                inputParameters["pos_input"][1] = pos_input
                inputParameters["neg_input"][1] = neg_input
                inputParameters["tracer_input"][1] = tracer_file
                inputParameters["run_sequence_pos_file"][1] = run_sequence_pos_file
                inputParameters["run_sequence_neg_file"][1] = run_sequence_neg_file
                # read the test files into pandas dataframes. Note: pos_input and neg_input are loaded later
                # in the code
                tracer_df = file_manager.tracer_handler(tracer_file)
                run_sequence_pos_df = file_manager.tracer_handler(run_sequence_pos_file)
                run_sequence_neg_df = file_manager.tracer_handler(run_sequence_neg_file)
            else:
                # handle case 2: the user has not selected to run the test files
                if "pos_input" in request.FILES.keys():
                    pos_input = request.FILES["pos_input"]
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["pos_input"][1] = pos_input.name
                else:
                    pos_input = None

                if "neg_input" in request.FILES.keys():
                    neg_input = request.FILES["neg_input"]
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["neg_input"][1] = neg_input.name
                else:
                    neg_input = None

                try:
                    tracer_file = request.FILES["tracer_input"]
                    tracer_df = file_manager.tracer_handler(tracer_file)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["tracer_input"][1] = tracer_file.name
                except Exception:
                    tracer_df = None
                # Check for either (or both run sequence files)
                try:
                    run_sequence_pos_file = request.FILES["run_sequence_pos_file"]
                    run_sequence_pos_df = file_manager.tracer_handler(run_sequence_pos_file)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["run_sequence_pos_file"][1] = run_sequence_pos_file.name
                except Exception:
                    run_sequence_pos_df = None

                try:
                    run_sequence_neg_file = request.FILES["run_sequence_neg_file"]
                    run_sequence_neg_df = file_manager.tracer_handler(run_sequence_neg_file)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["run_sequence_neg_file"][1] = run_sequence_neg_file.name
                except Exception:
                    run_sequence_neg_df = None

            # create a list of the input files
            inputs = [pos_input, neg_input]
            logger.info("Input Files: {} ".format(inputs))

            input_dfs = []
            # Get user-input non-detect value, pass to file_manager.input_handler
            # Try to convert to float if a number, if not store string
            try:
                na_value = float(parameters["na_val"])
            except ValueError:
                na_value = parameters["na_val"]
            # Iterate through inputs, format, and append to input_dfs
            for index, df in enumerate(inputs):
                if df is not None:
                    input_dfs.append(file_manager.input_handler(df, index, na_value))
                else:
                    input_dfs.append(None)

            # create a job ID
            job_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            logger.info("job ID: " + job_id)

            # log the submission
            logger.warn("MS1 Job {} Submitted. Parameters: {} ".format(job_id, inputParameters))

            run_nta_dask(
                inputParameters,
                input_dfs,
                tracer_df,
                run_sequence_pos_df,
                run_sequence_neg_df,
                job_id,
            )
            return redirect("/nta/ms1/processing/" + job_id, permanent=True)
        else:
            logger.info("form is NOT valid")
            form_data = request.POST
            form_files = request.FILES

    html = render_to_string(
        "01epa_drupal_header.html",
        {"SITE_SKIN": os.environ["SITE_SKIN"], "TITLE": "\u00FCbertool"},
    )
    html += render_to_string("02epa_drupal_header_bluestripe_onesidebar.html", {})
    html += render_to_string("epa_drupal_section_title_nta.html", {})

    # function name example: 'sip_input_page'
    html += render_to_string("ms1/nta_input_scripts.html")
    html += render_to_string("ms1/nta_input_css.html")
    if "/external/" in request.path:  # adding this switch as a short-term fix to connect AMOS front end
        input_start_form = "ms1/nta_input_start_drupal_nologin.html"
    else:
        input_start_form = "ms1/nta_input_start_drupal.html"
    html += render_to_string(
        input_start_form,
        {"MODEL": model, "TITLE": header},
        request=request,
    )

    html += str(NtaInputs(form_data, form_files))
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
