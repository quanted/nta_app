# File to hold custom decorators to use in views functions (web pages and APIs)
from functools import wraps
from django.http import JsonResponse
import os


def api_key_required(view_func):
    """ Require API key listed in the kubernetes secret 'django-api-keys"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        api_key = request.headers.get('x-api-key')
        keyfile_lines = os.getenv("DJANGO_API_KEYS").splitlines() # get each line of the key file
        api_keys_list = []
        for line in keyfile_lines:
            key = line.split('#', 1)[0].strip()  # Split at '#' and take the part before it
            if key: 
                api_keys_list.append(key)  # Add to the list if it's not empty
        if api_key not in api_keys_list: 
            return JsonResponse({'status': 'error', 'message': 'Invalid or missing API key'}, status=401)
        return view_func(request, *args, **kwargs)
    return _wrapped_view
