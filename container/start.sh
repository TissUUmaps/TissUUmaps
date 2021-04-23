#!/bin/bash
app="tissuumaps"
docker build -t ${app} .
docker run -d -p 56733:80 \
  --name=${app} \
  -v /mnt/c/Users/chrav452/Documents/datatest_tissuumaps:/mnt/data ${app}
