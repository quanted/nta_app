from ...app.data_handler.data_task import delete_data, get_filenames, get_grid_db, handle_uploaded_file
from django import forms
from django.http import HttpResponse
from django.utils.encoding import iri_to_uri



class FileUploadForm(forms.Form):
    file_source = forms.FileField()
    
class JobForm(forms.Form):
    job_name = forms.CharField()

def upload_api(request, form_data=None, form_files=None):
    if request.method == 'POST':
        #Add form validation for file post
        for key in request.FILES.keys():
            file_id = handle_uploaded_file(request.FILES[key], 
                                           request.POST['filename'], 
                                           request.POST['filetype'],
                                           request.POST['ms'], 
                                           request.POST['mode'], 
                                           request.POST['jobid'])
        response = HttpResponse(file_id)
    else:
        response = HttpResponse('Upload failed')
    return response
            
def delete_api(request, form_data=None):
    if request.method == 'POST':
        #form = JobForm(data=request.POST)
        #if form.is_valid():
        delete_data(request.POST['filename'], request.POST['jobid'], request.POST['ms'])
        response=HttpResponse('File Removed')
    else:
        response=HttpResponse('Removal Failed')
    return response

def get_files(request, form_data=None):
    response = HttpResponse('No files found')
    if request.method == 'POST':
        files = get_filenames(request.POST['jobid'], request.POST['ms'])
        response = HttpResponse(files)
    return response

class HttpResponseTemporaryRedirect(HttpResponse):
    status_code = 307

    def __init__(self, redirect_to):
        HttpResponse.__init__(self)
        self['Location'] = iri_to_uri(redirect_to)

