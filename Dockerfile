FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git && \
    pip3 install --upgrade pip

# Install dependencies
RUN pip3 install torch==2.4.0 unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git \
    xformers trl peft accelerate bitsandbytes datasets transformers

# Copy your training script to the container
COPY train.py /train.py

# Set the entrypoint
ENTRYPOINT ["python3", "/train.py"]