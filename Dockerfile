FROM python:3.9

RUN apt-get update --allow-releaseinfo-change -y
RUN apt-get upgrade --fix-missing -y
RUN apt-get install -y --fix-missing --no-install-recommends \
    python3-pip software-properties-common build-essential \
    cmake sqlite3 gfortran python-dev && \
    pip install -U pip

WORKDIR /src/nta_app
COPY . /src/nta_app

RUN pip install -r /src/nta_app/requirements.txt
RUN pip install uwsgi
RUN python --version

ENV PATH "/src:/src/nta_app":${PATH}
ENV PYTHONPATH "/src:/src/nta_app":${PYTHONPATH}
ENV DJANGO_SETTINGS_MODULE "settings"
EXPOSE 8080

COPY uwsgi.ini /etc/uwsgi/

RUN chmod 755 /src/nta_app/start_django.sh
CMD ["sh", "/src/nta_app/start_django.sh"]