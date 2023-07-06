FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /src
COPY . /src

RUN apt -y update
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Start with -im