FROM python:3.10-alpine

COPY ./container/requirements.txt /requirements.txt

# Install libvips

RUN apk update \
	&& apk upgrade
RUN apk add vips hdf5 hdf5-dev

RUN apk add --no-cache openblas-dev && \
    pip3 install --upgrade pip setuptools wheel && \
    apk add --no-cache --virtual .build-deps gfortran gcc g++ zlib-dev make python3-dev jpeg-dev && \
    pip3 install -r /requirements.txt && \
    apk del .build-deps

# Install openslide

WORKDIR /usr/local/src

ARG OPENSLIDE_VERSION=3.4.1
ARG OPENSLIDE_URL=https://github.com/openslide/openslide/releases/download

COPY container/openslide-init.patch /usr/local/src

RUN apk add --no-cache --virtual .build-deps2 build-base \
	autoconf automake libtool bc zlib-dev libxml2-dev jpeg-dev \
	openjpeg-dev tiff-dev glib-dev gdk-pixbuf-dev sqlite-dev libjpeg-turbo-dev \
	libexif-dev lcms2-dev fftw-dev giflib-dev libpng-dev libwebp-dev orc-dev \
	poppler-dev librsvg-dev libgsf-dev openexr-dev gtk-doc \
    && wget ${OPENSLIDE_URL}/v${OPENSLIDE_VERSION}/openslide-${OPENSLIDE_VERSION}.tar.gz \
	&& tar xf openslide-${OPENSLIDE_VERSION}.tar.gz \
	&& patch -p0 <openslide-init.patch \
    && cd openslide-${OPENSLIDE_VERSION} \
	&& ./configure \
	&& make \
	&& make install \
    && apk del .build-deps2

RUN pip3 install gunicorn gevent

COPY ./container/tissuumaps.cfg /tissuumaps.cfg
COPY ./tissuumaps/ /app/tissuumaps
WORKDIR /app/
ENV PYTHONPATH /app

ENV GUNICORN_CMD_ARGS "--bind=0.0.0.0:80 --workers=8 --thread=8 --worker-class=gevent --forwarded-allow-ips='*' -"

ENV TISSUUMAPS_CONF /tissuumaps.cfg

CMD ["gunicorn", "tissuumaps:app"]
