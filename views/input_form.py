from django import forms
from django.utils.safestring import mark_safe

from pram_app.models.forms import validation

application_method_CHOICES = (('soil application', 'soil application'), ('tree trunk', 'tree trunk'),
                              ('foliar spray', 'foliar spray'), ('seed treatment', 'seed treatment'))
empirical_residue_CHOICES = (('yes', 'yes'), ('no', 'no'))


class NtaInputs(forms.Form):

    version = forms.ChoiceField(
        choices=(('1.0', '1.0'),),
        label='Version',
        initial='1.0')
    project_name = forms.CharField(
        widget=forms.Textarea(attrs={'cols': 30, 'rows': 1}),
        initial='Example nta',
        required=True)
