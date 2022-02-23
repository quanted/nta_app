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
    test_files = forms.ChoiceField(
        label='Run test files only (debugging)',
        choices=(('no', 'no'),('yes', 'yes')),
        initial='no')
    pos_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label = 'Positive mode MS2 files (mgf)',
        validators= [FileExtensionValidator(['mgf'])],
        required=False)
    neg_inputs = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'multiple': True}),
        label='Negative mode MS2 files (mgf)',
        validators= [FileExtensionValidator(['mgf'])],
        required=False)
    #mass_accuracy_units = forms.ChoiceField(
    #    choices=(('ppm', 'ppm'), ('Da', 'Da'),),
    #    label = 'Adduct mass accuracy units',
    #    initial = 'ppm')+
    precursor_mass_accuracy = forms.FloatField(
        label='Precursor mass accuracy (ppm)',
        initial=10,
        validators=[MinValueValidator(0)])
    fragment_mass_accuracy = forms.FloatField(
        widget=forms.NumberInput(attrs={'step': 0.01}),
        label='Fragment mass accuracy (Da)',
        initial=0.02,
        validators=[MinValueValidator(0)])
        
    def clean(self):
        ms2_neg_clean = self.cleaned_data.get("neg_inputs")
        ms2_pos_clean = self.cleaned_data.get("pos_inputs")
        test_selected = self.cleaned_data.get("test_files")
        if not ms2_neg_clean and not ms2_pos_clean and test_selected == 'no':
            raise forms.ValidationError("No file entered for MS2 data")
        return self.cleaned_data

