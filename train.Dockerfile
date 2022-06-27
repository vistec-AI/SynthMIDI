FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN apt update
WORKDIR /workspace
COPY . .
RUN pip install -r train.requirements.txt

ENTRYPOINT [ "python", "train.py" ]
