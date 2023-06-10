FROM tensorflow/tensorflow:2.10.1-gpu

WORKDIR /src
COPY ./pyproject.toml /src/pyproject.toml
COPY ./setup.cfg /src/setup.cfg
COPY ./cmd_tf/__init__.py /src/cmd_tf/__init__.py
COPY ./cmd_tf/__main__.py /src/cmd_tf/__main__.py
COPY ./cmd_tf/training.py /src/cmd_tf/training.py
COPY ./cmd_tf/model.py /src/cmd_tf/model.py
COPY ./cmd_tf/batchgen.py /src/cmd_tf/batchgen.py
COPY ./cmd_tf/build_trainigdata.py /src/cmd_tf/build_trainigdata.py
COPY ./cmd_tf/loss.py /src/cmd_tf/loss.py
COPY ./cmd_tf/runconfigs.py /src/cmd_tf/runconfigs.py

RUN apt -y update
RUN apt -y install blender
RUN pip install pillow keras matplotlib tikzplotlib

ENTRYPOINT python -m cmd_tf