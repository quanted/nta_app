from ...tools.ms1.output_access import OutputServer


def download_toxpi(request, jobid=None):
    server = OutputServer(jobid)
    response = server.final_result()
    return response


def download_all(request, jobid=None):
    server = OutputServer(jobid)
    response = server.all_files()
    return response


def check_status(request, jobid=None):
    server = OutputServer(jobid)
    response = server.status()
    return response


def decision_tree_data(request, jobid=None):
    server = OutputServer(jobid)
    response = server.decision_tree()
    return response


def decision_tree_analysis_parameters(request, jobid=None):
    server = OutputServer(jobid)
    response = server.decision_tree_parameters()
    return response
