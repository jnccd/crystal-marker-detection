FROM tensorflow/tensorflow:2.10.1-gpu

WORKDIR /src
COPY ./pyproject.toml /src/pyproject.toml
COPY ./setup.cfg /src/setup.cfg
COPY ./ml_segmentation/__init__.py /src/ml_segmentation/__init__.py
COPY ./ml_segmentation/__main__.py /src/ml_segmentation/__main__.py
COPY ./ml_segmentation/training.py /src/ml_segmentation/training.py
COPY ./ml_segmentation/model.py /src/ml_segmentation/model.py
COPY ./ml_segmentation/batchgen.py /src/ml_segmentation/batchgen.py
COPY ./ml_segmentation/build_trainigdata.py /src/ml_segmentation/build_trainigdata.py
COPY ./ml_segmentation/loss.py /src/ml_segmentation/loss.py
COPY ./ml_segmentation/runconfigs.py /src/ml_segmentation/runconfigs.py

RUN apt -y update
RUN apt -y install blender
RUN pip install pillow keras matplotlib tikzplotlib

ENTRYPOINT python -m ml_segmentation