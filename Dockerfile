FROM python:3.8.10-slim-buster
WORKDIR /code

RUN pip install tensorflow
RUN pip install numpy
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install Pillow