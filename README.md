# nta_app
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ee7652f237d44dbab5e3a96d76165505)](https://www.codacy.com/app/puruckertom/nta_app?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=quanted/nta_app&amp;utm_campaign=Badge_Grade)

App for non-targeted analysis of mass spectrometry

For running a development version locally:
*  Requires a mongoDB instance running locally on default port (27017)
*  Run this Django app from manage.py with DJANGO_SETTINGS_MODULE = "settings_local"
*  Run the flask database api (flask_qed repo) using flask_qed/flask_cgi.py
