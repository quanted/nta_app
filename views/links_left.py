from django.template.loader import render_to_string
from collections import OrderedDict


def ordered_list(model=None, page=None):

    link_dict = OrderedDict([
        ('Tools', OrderedDict([
                ('MS1 Tool', 'ms1'),
                ('MS2 Tool', 'ms2')
            ])
         ),
        ('Documentation', OrderedDict([
                # ('API Documentation', '/qedinternal.epa.gov/pisces/rest'),
                ('Source Code', 'github'),
            ])
         )
    ])

    return render_to_string('nta_links_left_drupal.html', {
        'LINK_DICT': link_dict,
        'MODEL': model,
        'PAGE': page
    })
