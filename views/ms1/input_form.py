from django import forms
from django.core.validators import MinValueValidator, FileExtensionValidator
from django.forms.widgets import NumberInput
from django.utils.safestring import mark_safe


class RangeInput(NumberInput):
    input_type = "range"


class NtaInputs(forms.Form):
    project_name = forms.CharField(
        widget=forms.Textarea(attrs={"cols": 30, "rows": 1}),
        initial="Example nta",
        required=True,
    )
    test_files = forms.ChoiceField(
        label="Run test files only (debugging)",
        choices=(("no", "no"), ("yes", "yes")),
        initial="no",
    )
    pos_input = forms.FileField(
        label="Positive mode file (csv)",
        required=False,
        validators=[FileExtensionValidator(["csv"])],
    )
    neg_input = forms.FileField(
        label="Negative mode file (csv)",
        required=False,
        validators=[FileExtensionValidator(["csv"])],
    )
    na_val = forms.CharField(
        max_length=10,
        min_length=1,
        empty_value="",
        label="Input matrix non-detect value",
        initial="Please enter your data's non-detect number or character",
        required=True,
    )
    pos_adducts = forms.MultipleChoiceField(
        label="Positive mode adducts",
        widget=forms.CheckboxSelectMultiple(attrs={"class": "two"}),
        choices=(
            ("Na", "[M+Na]+"),
            ("K", "[M+K]+"),
            ("NH4", mark_safe("[M+NH<sub>4</sub>]+")),
        ),
        initial=["Na", "K", "NH4"],
        required=False,
    )
    neg_adducts = forms.MultipleChoiceField(
        label="Negative mode adducts",
        widget=forms.CheckboxSelectMultiple(attrs={"class": "two"}),
        choices=(
            ("Cl", "[M+Cl]-"),
            ("Br", "[M+Br]-"),
            ("HCO2", mark_safe("[M+HCO<sub>2</sub>]-")),
            ("CH3CO2", mark_safe("[M+CH<sub>3</sub>CO<sub>2</sub>]-")),
            ("CF3CO2", mark_safe("[M+CF<sub>3</sub>CO<sub>2</sub>]-")),
        ),
        initial=[
            "Cl",
            "HCO2",
            "CH3CO2",
        ],
        required=False,
    )
    neutral_losses = forms.MultipleChoiceField(
        label="Neutral losses (both modes)",
        widget=forms.CheckboxSelectMultiple(attrs={"class": "two"}),
        choices=(
            ("H2O", mark_safe("[M-H<sub>2</sub>O]")),
            ("2H2O", mark_safe("[M-2H<sub>2</sub>O]")),
            ("3H2O", mark_safe("[M-3H<sub>2</sub>O]")),
            ("4H2O", mark_safe("[M-4H<sub>2</sub>O]")),
            ("5H2O", mark_safe("[M-5H<sub>2</sub>O]")),
            ("NH3", mark_safe("[M-NH<sub>3</sub>]")),
            ("O", "[M+O]"),
            ("CO", "[M-CO]"),
            ("CO2", mark_safe("[M-CO<sub>2</sub>]")),
            ("C2H4", mark_safe("[M-C<sub>2</sub>H<sub>4</sub>]")),
            ("CH2O2", mark_safe("[M+CH<sub>2</sub>O<sub>2</sub>]")),
            ("CH3COOH", mark_safe("[M+CH<sub>3</sub>COOH]")),
            ("CH3OH", mark_safe("[M+CH<sub>3</sub>OH]")),
            ("CH3CN", mark_safe("[M+CH<sub>3</sub>CN]")),
            ("(CH3)2CHOH", mark_safe("[M+(CH<sub>3</sub>)<sub>2</sub>CHOH]")),
        ),
        initial=["H2O", "CO2"],
        required=False,
    )
    mass_accuracy_units = forms.ChoiceField(
        choices=(
            ("ppm", "ppm"),
            ("Da", "Da"),
        ),
        label="Adduct / duplicate mass accuracy units",
        initial="ppm",
    )
    mass_accuracy = forms.FloatField(
        label="Adduct / duplicate mass accuracy",
        initial=10,
        validators=[MinValueValidator(0)],
    )
    rt_accuracy = forms.FloatField(
        widget=forms.NumberInput(attrs={"step": 0.01}),
        label="Adduct / duplicate retention time accuracy (mins)",
        initial=0.05,
        validators=[MinValueValidator(0)],
    )
    run_sequence_pos_file = forms.FileField(
        label="Run sequence positive mode file (csv; optional)",
        required=False,
        validators=[FileExtensionValidator(["csv"])],
    )
    run_sequence_neg_file = forms.FileField(
        label="Run sequence negative mode file (csv; optional)",
        required=False,
        validators=[FileExtensionValidator(["csv"])],
    )
    tracer_input = forms.FileField(
        label="Tracer file (csv; optional)",
        required=False,
        validators=[FileExtensionValidator(["csv"])],
    )
    mass_accuracy_units_tr = forms.ChoiceField(
        choices=(
            ("ppm", "ppm"),
            ("Da", "Da"),
        ),
        label="Tracer mass accuracy units",
        initial="ppm",
    )
    mass_accuracy_tr = forms.FloatField(label="Tracer mass accuracy", initial=5, validators=[MinValueValidator(0)])
    rt_accuracy_tr = forms.DecimalField(
        widget=forms.NumberInput(attrs={"step": 0.1}),
        label="Tracer retention time accuracy (mins)",
        initial=0.1,
        validators=[MinValueValidator(0)],
    )
    tracer_plot_yaxis_format = forms.ChoiceField(
        choices=(
            ("log", "log"),
            ("linear", "linear"),
        ),
        label="Tracer plot y-axis scaling",
        initial="log",
    )
    tracer_plot_trendline = forms.ChoiceField(
        choices=(
            ("yes", "yes"),
            ("no", "no"),
        ),
        label="Tracer plot trendlines shown",
        initial="yes",
    )
    min_replicate_hits = forms.IntegerField(
        widget=RangeInput(attrs={"max": "100", "min": "1", "class": "slider_bar"}),
        label="Min replicate hits (%)",
        initial=66,
        validators=[MinValueValidator(0)],
    )
    min_replicate_hits_blanks = forms.IntegerField(
        widget=RangeInput(attrs={"max": "100", "min": "1", "class": "slider_bar"}),
        label="Min replicate hits in blanks (%)",
        initial=66,
        validators=[MinValueValidator(0)],
    )
    max_replicate_cv = forms.DecimalField(
        widget=forms.NumberInput(attrs={"step": 0.1}),
        label="Max replicate CV",
        initial=0.8,
        validators=[MinValueValidator(0)],
    )
    mrl_std_multiplier = forms.ChoiceField(
        choices=(
            ("3", 3),
            ("5", 5),
            ("10", 10),
        ),
        label="MRL standard deviation multiplier",
        initial="3",
    )
    parent_ion_mass_accuracy = forms.IntegerField(
        widget=RangeInput(attrs={"max": "10", "min": "1", "class": "slider_bar"}),
        label="Parent ion mass accuracy (ppm)",
        initial=5,
        validators=[MinValueValidator(0)],
    )
    minimum_rt = forms.FloatField(
        widget=forms.NumberInput(attrs={"step": 0.1}),
        label="Discard features below this retention time (mins)",
        initial=0.00,
        validators=[MinValueValidator(0)],
    )
    search_dsstox = forms.ChoiceField(
        label="Search DSSTox for possible structures",
        choices=(
            ("yes", "yes"),
            ("no", "no"),
        ),
        initial="yes",
    )
    search_hcd = forms.ChoiceField(
        label="Search Cheminformatics Hazard Module for toxicity data",
        choices=(
            ("yes", "yes"),
            ("no", "no"),
        ),
        initial="no",
    )
    search_mode = forms.ChoiceField(
        label="Search DSSTox by",
        choices=(
            ("mass", "mass"),
            ("formula", "formula"),
        ),
        initial="mass",
    )

    def __init__(self):
        super().__init__()
        # Check for pos input file
        if "pos_input" in self.FILES:
            try:
                pos_input = self.FILES["pos_input"]
            except Exception:
                pos_input = None
        # Check for neg input file
        if "neg_input" in self.FILES:
            try:
                neg_input = self.FILES["neg_input"]
            except Exception:
                neg_input = None
        # Check for tracer input file
        if "tracer_input" in self.FILES:
            try:
                tracer_input = self.FILES["tracer_input"]
            except Exception:
                tracer_input = None
        # If tracer input present, require run sequence files corresponding to data modes
        if tracer_input:
            if pos_input:
                self.fields["run_sequence_pos_file"].required = True
            if neg_input:
                self.fields["run_sequence_neg_file"].required = True
