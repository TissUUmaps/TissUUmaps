FROM tiangolo/uwsgi-nginx-flask:python3.7
COPY ./ /app
RUN apt-get update && apt-get install bash nano openslide-tools libvips -y
ENV STATIC_URL /static
ENV STATIC_PATH /app/static
COPY ./requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt
