FROM gcr.io/kaggle-gpu-images/python@sha256:940180dcea42e703acbeae669a52d68e6fa3130ef60805b36005534a5152a5f3

# PyTorchのバージョンをcuda12.4に対応したものに変える
RUN pip uninstall torch torchvision torchaudio --yes
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

USER root
RUN apt install git --yes