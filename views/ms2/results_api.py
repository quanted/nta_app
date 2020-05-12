from ...tools.ms2.output_access import OutputServer


def download_all(request, jobid = None):
    server = OutputServer(jobid)
    response = server.all_files()
    return response

def check_status(request, jobid= None):
    server = OutputServer(jobid)
    response = server.status()
    return response