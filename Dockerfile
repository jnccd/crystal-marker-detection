FROM tensorflow/tensorflow:2.12.0-gpu

WORKDIR /src
COPY . /src

RUN apt-get update && apt-get install git ffmpeg libsm6 libxext6 -y
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
RUN python -m pip install /data/super-gradients/

# Start with -im