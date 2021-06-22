#!/bin/bash
app="tissuumaps"
docker build -t ${app} -f container/Dockerfile .
docker run -p 56733:80 \
  --name=${app} \
  -v /home/chrav452/shared:/mnt/data ${app}
