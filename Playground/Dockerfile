FROM ubuntu:18.04

WORKDIR /playground

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get -y --no-install-recommends install \
    build-essential \
    clang \
    cmake \
    gdb \
    wget

CMD cd src && make clean && make all && /bin/bash && tmux