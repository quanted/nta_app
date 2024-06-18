from django.shortcuts import redirect
from django import forms


def github(request):
    return redirect("https://github.com/quanted/nta_app/")


# allow multiple file inputs to a django file input field
class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


# Field to allow multiple file inputs to a django file input field
class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result
