import os
import logging
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from django.http import HttpResponse
from django.template.loader import render_to_string
from .. import links_left


example_pos_filename = 'pooled_blood_pos_MPP.csv'
example_neg_filename = 'pooled_blood_neg_MPP.csv'
example_tracer_filename = 'pooled_blood_tracers.csv'
example_run_sequence_pos_filename = 'pooled_blood_run_sequence_pos.csv'
example_run_sequence_neg_filename = 'pooled_blood_run_sequence_neg.csv'

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("nta_app.views")
logger.setLevel(logging.INFO)

#@require_POST
def test_files_page(request, model='ms1', header='Download MS1 test files', jobid='00000000'):
    model_html = """<div id="Download button"><input type="button" value="Download MS1 Test Files" onclick="window.open('download')">
</div>"""
    html = formula_list_html(header, model, model_html)
    response = HttpResponse()
    response.write(html)
    #print(html)
    return response


def formula_list_html(header, model, tables_html):
    """Generates HTML to fill '.articles_output' div on output page"""
    page = 'ms1_test_files'
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
    html += render_to_string('nta_scripts_css.html', {})
    #html += render_to_string('09epa_drupal_pram_scripts.html', {})

    #epa template footer
    html += render_to_string('10epa_drupal_footer.html', {})
    return html

def download_test_files(request):
    """
    Downloads MS1 test files from the code directory input/ms1 and returns them as a zip file in the response.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object containing the zip file.

    """

    # Log the start of the function
    logger.info("=========== returns ms1 test files from code directory input/ms1")

    # create an absolute path to the 'example_data_dir' containing the test data files, then create
    # absolute paths to each test data file. Note the test data files are located in this code base.
    example_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','input/ms1')
    pos_input = os.path.join(example_data_dir, example_pos_filename)
    neg_input = os.path.join(example_data_dir, example_neg_filename)
    tracer_file = os.path.join(example_data_dir, example_tracer_filename)
    run_sequence_pos_file = os.path.join(example_data_dir, example_run_sequence_pos_filename)
    run_sequence_neg_file = os.path.join(example_data_dir, example_run_sequence_neg_filename)

    # create filenames
    filename1 = 'ms1_pos_input_test_data.csv'
    filename2 = 'ms1_neg_input_test_data.csv'
    filename3 = 'ms1_tracer_test_data.csv'
    filename4 = 'ms1_run_sequence_pos_test_data.csv'
    filename5 = 'ms1_run_sequence_neg_test_data.csv'

    # List of files to be zipped
    files_to_zip = {filename1: pos_input, filename2: neg_input, filename3: tracer_file, filename4: run_sequence_pos_file, filename5: run_sequence_neg_file}

    # Create an in-memory zip file
    in_memory_zip = BytesIO()
    with ZipFile(in_memory_zip, 'w', ZIP_DEFLATED) as zipf:
        # Add each file to the zipfile
        for filename in files_to_zip:
            logger.info('filename: {}'.format(filename))
            file_path = files_to_zip[filename]
            with open(file_path, 'rb') as file:
                file_content = file.read()
                zipf.writestr(filename, file_content)
        # The ZipFile object is automatically closed when exiting the 'with' block

    zip_filename = "ms1_test_data_files.zip"
    # Create an HTTP response with the zip file attached for download
    response = HttpResponse(in_memory_zip.getvalue(),content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename=' + zip_filename
    response['Content-length'] = in_memory_zip.tell()

    # Return the HTTP response
    return response