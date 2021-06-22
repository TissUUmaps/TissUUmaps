FROM python:3.8

# Install libvips and openslide
RUN apt-get update && apt-get install bash nano openslide-tools libvips -y

COPY ./container/requirements.txt /requirements.txt

RUN pip3 install -r /requirements.txt
RUN pip3 install gunicorn

COPY ./tissuumaps/ /app/tissuumaps
WORKDIR /app/
ENV PYTHONPATH /app

ENV GUNICORN_CMD_ARGS "--bind=0.0.0.0:80 --workers=4 --thread=4 --worker-class=gthread --forwarded-allow-ips='*' --access-logfile -"

CMD ["gunicorn", "tissuumaps:app"]