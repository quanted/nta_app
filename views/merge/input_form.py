from django import forms
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.forms.widgets import NumberInput
from ..misc import MultipleFileField


class RangeInput(NumberInput):
    input_type = "range"


class MergeInputs(forms.Form):
    project_name = forms.CharField(
        widget=forms.Textarea(attrs={"cols": 30, "rows": 1}),
        initial="Example NTA merge (test 2)",
        required=True,
    )
    ms1_inputs = MultipleFileField(
        label="NTA MS1 results file",
        validators=[FileExtensionValidator(["csv", "tsv", "xlsx"])],
        required=True,
    )
    ms2_neg_inputs = MultipleFileField(
        label="NTA MS2 results file (negative mode)",
        validators=[FileExtensionValidator(["csv"])],
        required=False,
    )
    ms2_pos_inputs = MultipleFileField(
        label="NTA MS2 results file (positive mode)",
        validators=[FileExtensionValidator(["csv"])],
        required=False,
    )
    mass_accuracy_tolerance = forms.FloatField(
        label="Mass acccuracy tolerance (ppm)",
        initial=10,
        validators=[MinValueValidator(0)],
    )
    rt_tolerance = forms.FloatField(
        widget=forms.NumberInput(attrs={"step": 0.01}),
        label="Retention time tolerance (min)",
        initial=0.3,
        validators=[MinValueValidator(0)],
    )

    def clean(self):
        ms2_neg_clean = self.cleaned_data.get("ms2_neg_inputs")
        ms2_pos_clean = self.cleaned_data.get("ms2_pos_inputs")
        if not ms2_neg_clean and not ms2_pos_clean:
            raise forms.ValidationError("No file entered for MS2 data")
        return self.cleaned_data
