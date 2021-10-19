FROM python:3.9

RUN apt-get update --allow-releaseinfo-change -y
RUN apt-get upgrade --fix-missing -y
RUN apt-get install -y --fix-missing --no-install-recommends \
    python3-pip software-properties-common build-essential \
    cmake sqlite3 gfortran python-dev && \
    pip install -U pip

WORKDIR /src/

RUN pip install -r requirements.txt
RUN python --version

ENV PATH "/src":${PATH}
ENV PYTHONPATH "/src":${PYTHONPATH}
ENV DJANGO_SETTINGS_MODULE = "settings"
EXPOSE 8080

COPY uwsgi.ini /etc/uwsgi/

COPY . /src/
RUN chmod 755 /src/start_django.sh
CMD ["sh", "/src/start_django.sh"]