FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install cmake libhiredis-dev
