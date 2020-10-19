from django import forms
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.forms.widgets import NumberInput


class RangeInput(NumberInput):
    input_type = 'range'

class MS2Inputs(forms.Form):

    project_name = forms.CharField(
        widget=forms.Textarea(attrs={'cols': 30, 'rows': 1}),
        initial='Example NTA merge',
        required=True)
    pos_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        #label = 'Positive mode MS2 files (mgf)',
        label = 'NTA MS1 results file',
        validators= [FileExtensionValidator(['mgf'])],
        required=False)
    neg_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='NTA MS2 results file',
        validators= [FileExtensionValidator(['mgf'])],
        required=False)
    #mass_accuracy_units = forms.ChoiceField(
    #    choices=(('ppm', 'ppm'), ('Da', 'Da'),),
    #    label = 'Adduct mass accuracy units',
    #    initial = 'ppm')
    precursor_mass_accuracy = forms.FloatField(
        label='Precursor mass accuracy (ppm)',
        initial=10,
        validators=[MinValueValidator(0)])
    fragment_mass_accuracy = forms.FloatField(
        widget=forms.NumberInput(attrs={'step': 0.01}),
        label='Fragment mass accuracy (Da)',
        initial=0.02,
        validators=[MinValueValidator(0)])

