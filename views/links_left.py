from django.template.loader import render_to_string
from collections import OrderedDict


def ordered_list(model=None, page=None):

    link_dict = OrderedDict([
        ('Model', OrderedDict([
                ('NTA', 'nta'),
            ])
         ),
        ('Documentation', OrderedDict([
                # ('API Documentation', '/qedinternal.epa.gov/pisces/rest'),
                ('Source Code', '/github.com/quanted/nta_app'),
            ])
         )
    ])

    return render_to_string('nta_links_left_drupal.html', {
        'LINK_DICT': link_dict,
        'MODEL': model,
        'PAGE': page
    })
