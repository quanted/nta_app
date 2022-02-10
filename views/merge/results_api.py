from ...tools.merge.output_access import OutputServer


def download_merged(request, jobid = None):
    server = OutputServer(jobid)
    response = server.final_result()
    return response

def check_status(request, jobid= None):
    server = OutputServer(jobid)
    response = server.status()
    return response
