from django import forms
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.forms.widgets import NumberInput


class RangeInput(NumberInput):
    input_type = 'range'

class NtaInputs(forms.Form):

    project_name = forms.CharField(
        widget=forms.Textarea(attrs={'cols': 30, 'rows': 1}),
        initial='Example nta',
        required=True)
    test_files = forms.ChoiceField(
        label='Run test files only (debugging)',
        choices=(('no', 'no'),('yes', 'yes')),
        initial='no')
    pos_input = forms.FileField(
        label = 'Positive MPP file (csv)',
        validators= [FileExtensionValidator(['csv'])])
    neg_input = forms.FileField(
        label='Negative MPP file (csv)',
        validators= [FileExtensionValidator(['csv'])])
    mass_accuracy_units = forms.ChoiceField(
        choices=(('ppm', 'ppm'), ('Da', 'Da'),),
        label = 'Adduct mass accuracy units',
        initial = 'ppm')
    mass_accuracy = forms.FloatField(
        label='Adduct mass accuracy',
        initial=10,
        validators=[MinValueValidator(0)])
    rt_accuracy = forms.FloatField(
        widget=forms.NumberInput(attrs={'step': 0.01}),
        label='Adduct retention time accuracy (mins)',
        initial=0.05,
        validators=[MinValueValidator(0)])
    tracer_input = forms.FileField(
        label='Tracer file (csv; optional)',
        required=False,
        validators= [FileExtensionValidator(['csv'])])
    mass_accuracy_units_tr = forms.ChoiceField(
        choices=(('ppm', 'ppm'), ('Da', 'Da'),),
        label='Tracer mass accuracy units',
        initial='ppm')
    mass_accuracy_tr = forms.FloatField(
        label='Tracer mass accuracy',
        initial=5,
        validators=[MinValueValidator(0)])
    rt_accuracy_tr = forms.DecimalField(
        widget=forms.NumberInput(attrs={'step': 0.1}),
        label='Tracer retention time accuracy (mins)',
        initial=0.1,
        validators=[MinValueValidator(0)])
    sample_to_blank = forms.FloatField(
        widget=forms.NumberInput(attrs={'step': 0.5}),
        label='Min sample:blank cutoff',
        initial=3,
        validators=[MinValueValidator(0)])
    min_replicate_hits = forms.IntegerField(
        widget = RangeInput(attrs={'max': '20', 'min':'1', 'class': 'slider_bar'}),
        label='Min replicate hits',
        initial=2,
        validators=[MinValueValidator(0)])
    max_replicate_cv = forms.DecimalField(
        widget=forms.NumberInput(attrs={'step': 0.1}),
        label='Max replicate CV',
        initial=0.8,
        validators=[MinValueValidator(0)])
    parent_ion_mass_accuracy = forms.IntegerField(
        widget = RangeInput(attrs={'max': '10', 'min':'1', 'class': 'slider_bar'}),
        label='Parent ion mass accuracy (ppm)',
        initial=5,
        validators=[MinValueValidator(0)])
    minimum_rt = forms.FloatField(
        widget=forms.NumberInput(attrs={'step': 0.1}),
        label='Discard features below this retention time (mins)',
        initial=0.00,
        validators=[MinValueValidator(0)])
    search_dsstox = forms.ChoiceField(
        label='Search DSSTox for possible structures',
        choices=(('yes','yes'),('no','no'),),
        initial='yes')
    search_hcd = forms.ChoiceField(
        label='Search Hazard Comparison Dashboard for toxicity data',
        choices=(('yes','yes'),('no','no'),),
        initial='no')
    search_mode = forms.ChoiceField(
        label='Search dashboard by',
        choices=(('mass', 'mass'), ('formula', 'formula'),),
        initial='mass')
    top_result_only = forms.ChoiceField(
        label='Save top result only?',
        choices=(('yes', 'yes'), ('no', 'no'),),
        initial='no')
    api_batch_size = forms.IntegerField(
        widget=forms.NumberInput(attrs={'step': 1}),
        label='DSSTox search batch size (debugging)',
        initial=150)
