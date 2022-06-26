FROM python:3.8

ARG soundfont_url="https://musical-artifacts.com/artifacts/538/Roland_SC-88.sf2"

RUN apt update

# install fluidsynth
RUN apt install -y fluidsynth

# install sound fonts
WORKDIR /root
RUN mkdir -p /root/.fluidsynth
RUN wget $soundfont_url -O /root/.fluidsynth/default_soundfont.sf2

RUN pip install -U pip

WORKDIR /workspace
COPY . .
RUN pip install -r dataset.requirements.txt
ENTRYPOINT ["python", "generate_dataset.py"]
