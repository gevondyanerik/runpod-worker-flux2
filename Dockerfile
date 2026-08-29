# syntax=docker/dockerfile:1

# FLUX.2 klein worker for Runpod Serverless.
#
# Two build arguments matter:
#
#   BAKE_VARIANT   which profile's weights go into the image (default klein-4b)
#   BAKE_WEIGHTS   set to 0 to build a weightless image; it then downloads on
#                  first boot. CI uses this so a pull request does not pull
#                  12 GB of checkpoints just to prove the image builds.
#
# Everything else — ComfyUI's revision, the CUDA base, the torch build — is
# pinned, because "whatever was latest when this layer was rebuilt" is not a
# reproducible deployment.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG COMFYUI_REF=v0.34.0
ARG BAKE_VARIANT=klein-4b
ARG BAKE_TEXT_ENCODER=bf16
ARG BAKE_WEIGHTS=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# torch first, from the CUDA 12.8 index. Installing it before ComfyUI's
# requirements stops the unpinned `torch` line there from resolving to a
# different build.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 \
        torchvision==0.26.0 \
        torchaudio==2.11.0

# ComfyUI, pinned. The tag is resolved to a commit so the layer is stable even
# if the tag is ever moved.
RUN git clone --depth 1 --branch ${COMFYUI_REF} \
        https://github.com/comfyanonymous/ComfyUI.git /comfyui \
    && cd /comfyui \
    && git rev-parse HEAD > /comfyui/.commit \
    && rm -rf /comfyui/.git
RUN pip install -r /comfyui/requirements.txt

WORKDIR /worker
COPY requirements.txt /worker/requirements.txt
RUN pip install -r /worker/requirements.txt

COPY app /worker/app
COPY bootstrap /worker/bootstrap
COPY scripts /worker/scripts
COPY handler.py /worker/handler.py

# Weights. A layer of its own so changing the worker's Python does not force a
# re-download of several gigabytes.
RUN if [ "${BAKE_WEIGHTS}" = "1" ]; then \
        python3 /worker/scripts/fetch_weights.py "${BAKE_VARIANT}" \
            --text-encoder "${BAKE_TEXT_ENCODER}" ; \
    else \
        echo "skipping weight bake; the worker will download on first boot" ; \
    fi \
    && rm -rf /root/.cache/huggingface /tmp/hf-cache

ENV PYTHONPATH=/worker \
    FLUX2_BAKED_VARIANT=${BAKE_VARIANT}

CMD ["python3", "-u", "/worker/handler.py"]
