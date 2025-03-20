# File to hold custom decorators to use in views functions (web pages and APIs)
from functools import wraps
from django.http import JsonResponse
import os


def api_key_required(view_func):
    """ Require API key listed in the kubernetes secret 'django-api-keys"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key not in os.getenv("DJANGO_API_KEYS"): 
            return JsonResponse({'status': 'error', 'message': 'Invalid or missing API key'}, status=401)
        return view_func(request, *args, **kwargs)
    return _wrapped_view
