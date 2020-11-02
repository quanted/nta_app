from django import forms
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.forms.widgets import NumberInput


class RangeInput(NumberInput):
    input_type = 'range'

class MS2Inputs(forms.Form):

    project_name = forms.CharField(
        widget=forms.Textarea(attrs={'cols': 30, 'rows': 1}),
        initial='Example NTA merge (test 2)',
        required=True)
    ms1_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        #label = 'Positive mode MS2 files (mgf)',
        label = 'NTA MS1 results file',
        validators= [FileExtensionValidator(['csv'])],
        required=False)
    ms2_neg_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='NTA MS2 results file (negative mode)',
        validators= [FileExtensionValidator(['csv'])],
        required=False)
    ms2_pos_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='NTA MS2 results file (positive mode)',
        validators= [FileExtensionValidator(['csv'])],
        required=False)
    pcdl_neg_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='PCDL MS2 results file (negative mode)',
        validators= [FileExtensionValidator(['csv'])],
        required=False)
    pcdl_pos_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='PCDL MS2 results file (negative mode)',
        validators= [FileExtensionValidator(['csv'])],
        required=False)
    #mass_accuracy_units = forms.ChoiceField(
    #    choices=(('ppm', 'ppm'), ('Da', 'Da'),),
    #    label = 'Adduct mass accuracy units',
    #    initial = 'ppm')
    mass_accuracy_tolerance = forms.FloatField(
        label='Mass acccuracy tolerance (ppm)',
        initial=10,
        validators=[MinValueValidator(0)])
    rt_tolerance = forms.FloatField(
        widget=forms.NumberInput(attrs={'step': 0.01}),
        label='Retention time tolerance (min)',
        initial=0.3,
        validators=[MinValueValidator(0)])

