from django.template.loader import render_to_string
from collections import OrderedDict


def ordered_list(model=None, page=None):
    link_dict = OrderedDict(
        [
            (
                "Tools",
                OrderedDict(
                    [
                        ("MS1 Tool", "ms1"),
                        ("MS2 CFMID Tool", "ms2"),
                        ("Merge MS1 and MS2 Data", "merge"),
                    ]
                ),
            ),
            (
                "Documentation",
                OrderedDict(
                    [
                        # ('API Documentation', '/qedinternal.epa.gov/pisces/rest'),
                        ("Source Code", "github"),
                        (
                            "NTA Informatics Toolkit User Guide v0.1",
                            "static/docs/NTA-WebApp-user-guide_v0.1.docx",
                        ),
                        ("Version updates / Work in development", "version_info"),
                    ]
                ),
            ),
        ]
    )

    return render_to_string(
        "nta_links_left_drupal.html",
        {"LINK_DICT": link_dict, "MODEL": model, "PAGE": page},
    )
