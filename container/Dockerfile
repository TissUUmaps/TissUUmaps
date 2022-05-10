FROM python:3.8

# Install libvips and openslide
RUN apt-get update && apt-get install bash nano openslide-tools libvips -y

COPY ./container/requirements.txt /requirements.txt
COPY ./container/tissuumaps.cfg /tissuumaps.cfg

RUN pip3 install -r /requirements.txt
RUN pip3 install gunicorn gevent

COPY ./tissuumaps/ /app/tissuumaps
WORKDIR /app/
ENV PYTHONPATH /app

ENV GUNICORN_CMD_ARGS "--bind=0.0.0.0:80 --workers=8 --thread=8 --worker-class=gevent --forwarded-allow-ips='*' -"

ENV TISSUUMAPS_CONF /tissuumaps.cfg

CMD ["gunicorn", "tissuumaps:app"]
