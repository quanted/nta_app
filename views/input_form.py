from django import forms
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.utils.safestring import mark_safe
from django.forms.widgets import NumberInput

from nta_app.models.forms import validation


class RangeInput(NumberInput):
    input_type = 'range'

class NtaInputs(forms.Form):

    project_name = forms.CharField(
        widget=forms.Textarea(attrs={'cols': 30, 'rows': 1}),
        initial='Example nta',
        required=True)
    pos_input = forms.FileField(
        label = 'Positive MPP file (csv)',
        validators= [FileExtensionValidator(['csv'])])
    neg_input = forms.FileField(
        label='Negative MPP file (csv)',
        validators= [FileExtensionValidator(['csv'])])
    entact = forms.ChoiceField(
        label = 'ENTACT files?',
        choices = (('yes', 'yes'), ('no', 'no'),),
        initial = 'no')
    mass_accuracy_units = forms.ChoiceField(
        choices=(('ppm', 'ppm'), ('Da', 'Da'),),
        label = 'Adduct mass accuracy units',
        initial = 'ppm')
    mass_accuracy = forms.FloatField(
        label='Adduct mass accuracy',
        initial=10,
        validators=[MinValueValidator(0)])
    rt_accuracy = forms.FloatField(
        label='Adduct retention time accuracy (min)',
        initial=1,
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
        initial=10,
        validators=[MinValueValidator(0)])
    rt_accuracy_tr = forms.FloatField(
        label='Tracer retention time accuracy (min)',
        initial=1,
        validators=[MinValueValidator(0)])
    sample_to_blank = forms.FloatField(
        label = 'Min sample:blank cutoff',
        initial=3,
        validators=[MinValueValidator(0)])
    min_replicate_hits = forms.IntegerField(
        widget = RangeInput(attrs={'max': '3', 'min':'1', 'class': 'slider_bar'}),
        label='Min replicate hits',
        initial=1,
        validators=[MinValueValidator(0)])
    max_replicate_cv = forms.FloatField(
        label='Max replicate CV',
        initial=0.8,
        validators=[MinValueValidator(0)])
    parent_ion_mass_accuracy = forms.IntegerField(
        widget = RangeInput(attrs={'max': '10', 'min':'1', 'class': 'slider_bar'}),
        label='Parent ion mass accuracy (ppm)',
        initial=5,
        validators=[MinValueValidator(0)])
    search_mode = forms.ChoiceField(
        label='Search dashboard by',
        choices=(('mass', 'mass'), ('formula', 'formula'),),
        initial='mass')
    top_result_only = forms.ChoiceField(
        label='Save top result only?',
        choices=(('yes', 'yes'), ('no', 'no'),),
        initial='no')
