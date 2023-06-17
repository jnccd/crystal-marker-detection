FROM tensorflow/tensorflow:2.10.1-gpu

WORKDIR /src
COPY . /src

RUN apt -y update
RUN pip install -r requirements.txt

ENTRYPOINT python -m cmd_tf