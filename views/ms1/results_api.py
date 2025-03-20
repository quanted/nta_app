from ...tools.ms1.output_access import OutputServer
from ..views_dectorators import api_key_required
from django.views.decorators.csrf import csrf_exempt

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

# Below are API-key versions that wrap the above functions, intended for integration with AMOS or other apps / users

@csrf_exempt
@api_key_required
def check_status_api_key(request, jobid=None):
    return check_status(request, jobid)

@csrf_exempt
@api_key_required
def get_output_api_key(request, jobid=None):
    return download_toxpi(request, jobid)

@csrf_exempt
@api_key_required
def decision_tree_data_api_key(request, jobid=None):
    """not currently assigned to a URL"""
    return decision_tree_data(request, jobid)

@csrf_exempt
@api_key_required
def decision_tree_parameters_api_key(request, jobid=None):
    """not currently assigned to a URL"""
    return decision_tree_analysis_parameters(request, jobid)