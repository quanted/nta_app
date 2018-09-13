from django import forms
from django.core.validators import MinValueValidator
from django.utils.safestring import mark_safe

from pram_app.models.forms import validation


class NtaInputs(forms.Form):

    version = forms.ChoiceField(
        choices=(('1.0', '1.0'),),
        label='Version',
        initial='1.0')
    project_name = forms.CharField(
        widget=forms.Textarea(attrs={'cols': 30, 'rows': 1}),
        initial='Example nta',
        required=True)
    pos_input = forms.FileField(
        label = 'Positive MPP file (csv)')
    neg_input = forms.FileField(
        label='Negative MPP file (csv)')
    mass_accuracy_units = forms.ChoiceField(
        choices=(('ppm', 'ppm'), ('Da', 'Da'),),
        label = 'Mass accuracy units',
        initial = 'ppm')
    mass_accuracy = forms.FloatField(
        label='Mass accuracy',
        initial=20,
        validators=[MinValueValidator(0)])
    rt_accuracy = forms.FloatField(
        label='Retention time accuracy (min)',
        initial=1,
        validators=[MinValueValidator(0)])