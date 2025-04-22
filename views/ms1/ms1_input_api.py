# decorators.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.core.exceptions import ValidationError
import os
import string, random
import datetime
import logging
from ...app.ms1.nta_task import run_nta_dask
from ...tools.ms1 import file_manager
from ..views_dectorators import api_key_required
from ...data.atom_ranges import atom_ranges

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
example_surrogate_filename = "qNTA_Surrogate_Input_File_WW2DW.csv"


@api_key_required
@csrf_exempt
def ms1_run_api(request):
    """
    The API to trigger an MS1 task
    """
    if request.method == "POST":
        logger.info("POST received")
        try:
            data = request.POST
            logger.info("POST: {}".format(request.POST))
            # generate a timestamp with the current time and date
            current_datetime = datetime.datetime.now()

            # manually define current version of the WebApp
            current_version = "0.3.6"

            # Initialize parameters dictionary from the POST data (but not files). Second argument gives
            # the defualt value if the parameter is not passed in the POST request data.
            parameters = {
                "project_name": data.get("project_name", "Example nta"),
                "version": ["WebApp Version", current_version],
                "datetime": ["Date & time", str(current_datetime)],
                "test_files": data.get("test_files", "no"),
                "pos_adducts": data.getlist("pos_adducts[]", ["Na", "K", "NH4"]),
                "neg_adducts": data.getlist("neg_adducts[]", ["Cl", "HCO2", "CH3CO2", "FA"]),
                "neutral_losses": data.getlist("neutral_losses[]", ["H2O", "CO2"]),
                "mass_accuracy_units": data.get("mass_accuracy_units", "ppm"),
                "mass_accuracy": data.get("mass_accuracy", 10),
                "rt_accuracy": data.get("rt_accuracy", 0.05),
                "mass_accuracy_units_tr": data.get("mass_accuracy_units_tr", "ppm"),
                "mass_accuracy_tr": data.get("mass_accuracy_tr", 5),
                "rt_accuracy_tr": data.get("rt_accuracy_tr", 0.1),
                "tracer_plot_yaxis_format": data.get("tracer_plot_yaxis_format", "log"),
                "tracer_plot_trendline": data.get("tracer_plot_trendline", "yes"),
                "min_replicate_hits": data.get("min_replicate_hits", 66),
                "min_replicate_hits_blanks": data.get("min_replicate_hits_blanks", 66),
                "max_replicate_cv": data.get("max_replicate_cv", 0.8),
                "mrl_std_multiplier": data.get("mrl_std_multiplier", "3"),
                "parent_ion_mass_accuracy": data.get("parent_ion_mass_accuracy", 5),
                "minimum_rt": data.get("minimum_rt", 0.00),
                "search_dsstox": data.get("search_dsstox", "yes"),
                "search_hcd": data.get("search_hcd", "no"),
                "search_mode": data.get("search_mode", "mass"),
                "do_qnta": data.get("do_qnta", "no"),
                "do_atom_filtering": data.get("do_atom_filtering", "no"),
                "atom_ranges": data.get("atom_ranges", None),
                "na_val": data.get("na_val", ""),
            }
            # Validate numerical fields here
            MinValueValidator(0)(float(parameters["mass_accuracy"]))
            MinValueValidator(0)(float(parameters["rt_accuracy"]))
            MinValueValidator(0)(float(parameters["mass_accuracy_tr"]))
            MinValueValidator(0)(float(parameters["rt_accuracy_tr"]))
            MinValueValidator(0)(float(parameters["min_replicate_hits"]))
            MinValueValidator(0)(float(parameters["min_replicate_hits_blanks"]))
            MinValueValidator(0)(float(parameters["max_replicate_cv"]))
            MinValueValidator(0)(float(parameters["parent_ion_mass_accuracy"]))
            MinValueValidator(0)(float(parameters["minimum_rt"]))

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

            # generate a timestamp with the current time and date
            current_datetime = datetime.datetime.now()

            # define inputParameters dictionary containing all the parameters and their attributes, labels, and initial values
            inputParameters = {
                "project_name": ["Project name", None],
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
                "do_qnta": ["Perform qNTA?", None],
                "qnta_input": ["qNTA Surrogate input file", None],
                "do_atom_filtering": ["Do atom filtering?", None],
                "atom_ranges": ["Atom filtering ranges", None],
            }

            # save the Request parameters in the inputParameters dictionary [0] is the label, [1] is the value
            # This does not include the uploaded files, pos_input, neg_input, run_sequence_pos_file,
            # run_sequence_neg_file, and tracer_input, which are handled separately
            # save the Request parameters in the inputParameters dictionary [0] is the label, [1] is the value
            # This does not include the uploaded files, pos_input, neg_input, run_sequence_pos_file,
            # run_sequence_neg_file, and tracer_input, which are handled separately
            inputParameters["project_name"][1] = parameters["project_name"]
            inputParameters["test_files"][1] = parameters["test_files"]
            inputParameters["pos_adducts"][1] = parameters["pos_adducts"]
            inputParameters["neg_adducts"][1] = parameters["neg_adducts"]
            inputParameters["neutral_losses"][1] = parameters["neutral_losses"]
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
            inputParameters["do_qnta"][1] = parameters["do_qnta"]
            inputParameters["do_atom_filtering"][1] = parameters["do_atom_filtering"]
            # Check if 'do_atom_filtering'
            if inputParameters["do_atom_filtering"][1] == "yes":
                # Get user-submitted atom dict li
                us_atom_dict_li = request.POST.getlist("atom_ranges")
                logger.info("parameters atom_ranges: {} ".format(us_atom_dict_li))
                # Update atom filtering dictionary if present
                if us_atom_dict_li is not None:
                    # Log POST.request
                    logger.info("Atom filtering is yes, and user submitted atom_ranges")
                    # Store in temporary variable for updates
                    atom_dict_li = atom_ranges.copy()
                    # Iterate through list items and update matches
                    for item1 in atom_dict_li:
                        for item2 in us_atom_dict_li:
                            if item1["element"] == item2["element"]:
                                item1["min"] = item2["min"]
                                item1["max"] = item2["max"]
                                break

                    logger.info("updated atom_ranges: {} ".format(atom_dict_li))
                    # Store updated dictionary list
                    inputParameters["atom_ranges"][1] = atom_dict_li
                # Else, use default dictionary of atom_ranges
                else:
                    inputParameters["atom_ranges"][1] = atom_ranges
            # else set atom_ranges to None
            else:
                inputParameters["atom_ranges"][1] = None

            # Print selected adducts to logger
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
                if parameters["do_qnta"] == "yes":
                    qnta_file = os.path.join(example_data_dir, example_surrogate_filename)
                    inputParameters["qnta_input"][1] = qnta_file
                    qnta_df = file_manager.tracer_handler(qnta_file)
                else:
                    inputParameters["qnta_input"][1] = None
                    qnta_df = None
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

                # function to validate file extensions using Django's FileExtensionValidator
                file_validator = FileExtensionValidator(allowed_extensions=["csv"])

                if "pos_input" in request.FILES.keys():
                    pos_input = request.FILES["pos_input"]
                    file_validator(pos_input)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["pos_input"][1] = pos_input.name
                else:
                    pos_input = None

                if "neg_input" in request.FILES.keys():
                    neg_input = request.FILES["neg_input"]
                    file_validator(neg_input)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["neg_input"][1] = neg_input.name
                else:
                    neg_input = None

                try:
                    tracer_file = request.FILES["tracer_input"]
                    file_validator(tracer_file)
                    tracer_df = file_manager.tracer_handler(tracer_file)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["tracer_input"][1] = tracer_file.name
                except Exception:
                    tracer_df = None

                try:
                    run_sequence_pos_file = request.FILES["run_sequence_pos_file"]
                    file_validator(run_sequence_pos_file)
                    run_sequence_pos_df = file_manager.tracer_handler(run_sequence_pos_file)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["run_sequence_pos_file"][1] = run_sequence_pos_file.name
                except Exception:
                    run_sequence_pos_df = None

                try:
                    run_sequence_neg_file = request.FILES["run_sequence_neg_file"]
                    file_validator(run_sequence_neg_file)
                    run_sequence_neg_df = file_manager.tracer_handler(run_sequence_neg_file)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["run_sequence_neg_file"][1] = run_sequence_neg_file.name
                except Exception:
                    run_sequence_neg_df = None
                try:
                    qnta_file = request.FILES["qnta_input"]
                    qnta_df = file_manager.tracer_handler(qnta_file)
                    # save the name of the file to the inputParameters dictionary
                    inputParameters["qnta_input"][1] = qnta_file.name
                except Exception:
                    qnta_df = None

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
            # Use test_file_input_handler for test files
            if parameters["test_files"] == "yes":
                for index, df in enumerate(inputs):
                    if df is not None:
                        input_dfs.append(file_manager.test_file_input_handler(df, index, na_value))
                    else:
                        input_dfs.append(None)
            # Use input_handler for user-submitted files
            else:
                for index, df in enumerate(inputs):
                    if df is not None:
                        input_dfs.append(file_manager.input_handler(df, index, na_value))
                    else:
                        input_dfs.append(None)

            # create a job ID
            job_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

            # log the submission
            logger.warning("API - MS1 Job {} Submitted. Parameters: {} ".format(job_id, inputParameters))

            run_nta_dask(
                inputParameters,
                input_dfs,
                tracer_df,
                run_sequence_pos_df,
                run_sequence_neg_df,
                qnta_df,
                job_id,
            )
            # return redirect("/nta/ms1/processing/" + job_id, permanent=True)
            processing_url = "/nta/ms1/api/status/" + job_id
            return JsonResponse({"status": "success", "job_id": job_id, "status_url": processing_url}, status=200)

        except ValidationError as e:
            logger.warning("API - MS1 Job {} is NOT valid. Parameters: {} ".format(job_id, inputParameters))
            return JsonResponse({"status": "Input Validation Error", "message": str(e)}, status=400)
        except json.JSONDecodeError:
            logger.info("Invalid JSON")
            return JsonResponse({"status": "Error", "message": "Invalid JSON"}, status=400)
    return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)
