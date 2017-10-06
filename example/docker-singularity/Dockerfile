FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
  git \
  build-essential
RUN apt-get install -y autotools-dev libtool autoconf python

ADD install_singularity.sh install_singularity.sh


RUN ./install_singularity.sh
