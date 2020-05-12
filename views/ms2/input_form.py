from django import forms
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.forms.widgets import NumberInput


class RangeInput(NumberInput):
    input_type = 'range'

class MS2Inputs(forms.Form):

    project_name = forms.CharField(
        widget=forms.Textarea(attrs={'cols': 30, 'rows': 1}),
        initial='Example ms2 nta',
        required=True)
    results_email = forms.EmailField(
        label='Email for results link (optional):',
        initial='',
        required=False)
    pos_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label = 'Positive MS2 files (mgf)',
        validators= [FileExtensionValidator(['mgf'])])
    neg_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='Negative MS2 files (mgf)',
        validators= [FileExtensionValidator(['mgf'])])
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

