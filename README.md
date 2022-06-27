# SynthMIDI Dataset
SynthMIDI, a synthesized dataset for a single-note classification training. The dataset consists of a customizable data generation with all possible MIDI programs and notes. The goal of this dataset is to be used for the academic purpose.

## Table of Content
1. [Commit History](#commit-history)
2. [Requirements](#requirements)
3. [Usage: Generating Dataset](#usage-generating-dataset)
    - [Running on Docker](#running-on-docker)
    - [Running on local](#running-on-local)
    - [Dataset config](#dataset-configuration)
4. [Usage: Training Baseline](#usage-training-baseline)
    - [Training on Docker](#training-on-docker)
    - [Training on local](#training-on-local)
    - [Training config](#training-configuration)
5. [Baseline](#baseline)
    - [Stratify Splitting]()
    - [Leave Instrument Out]()
6. [Contributing](#contributing)
7. [Author](#author)
8. [Acknowledgement](#acknowledgement)

## Commit History
(26/06/2022) **Refactor code structures 🪡**
- Refactor code structure to isolate training and data generation pipeline
- Fix bugs where note generation mapped the keys wrongly
- Add customizable baseline model training pipeline

(25/06/2022) **initial commit 🍢** 
- Push the initial code.
- The only preprocessing function includes a simple noise injection and minor scaling

## Dataset
(26/06/2022) `midisynth_dataset-v1` ([download here](https://drive.google.com/file/d/1LkLNaKdqsDbIJeWm4jHJpAD2u6dH9CLW/view?usp=sharing))
- The very first version of MIDI generated from `mido` library
- The only augmentation applied was gaussian noise injection

## Requirements
We assume the reader use docker to run this script, or else some modification will be needed. If you didn't have one, you can download it [here]().

## Usage: Generating Dataset
To begin with, download or clone this repository. This can be done by running the following command:
```bash
git clone https://github.com/vistec-AI/SynthMIDI.git
cd SynthMIDI
```

### Running on Docker
We recommend you to run this script using docker, as we test only on a docker environment.

#### 1. Building Docker
To build the docker, run:
```bash
docker build -t synthmidi/data-generator -f dataset.Dockerfile .
```
Once finished, you should have a built image on your docker. Run `docker images` to see if the image was built properly.

#### 2. Generating Dataset with Docker
Once you've built the docker, run the following command on your terminal:
```bash
docker run -it -v $PWD/conf:/workspace/conf -v /path/to/saved/dataset:/root/dataset --name your-docker-container-name synthmidi/data-generator
```

### Running on local
If you're still insist of running on local, that's fine. But note that we use python3.8, so make sure you use the same python version to prevent any bug.
#### 1. Activate Virtual Environment (Optional)
It is wise to work on a separated environment for each project. If you're not skipping this section even though you saw that is is optional, you're a man of culture.
```bash
python3 -m venv dataset.venv  # create virtual environment
source dataset.venv/bin/activate  # activate virtual environment
```
#### 2. Install the dependencies
There're several dependencies you needed to run this script. To install those libraries, run:
```bash
pip install -r dataset.requirements.txt
```
We also recommended you to install soundfont by the following command:
```bash
wget https://musical-artifacts.com/artifacts/538/Roland_SC-88.sf2 -O ~/.fluidsynth/default_soundfont.sf2
```
This step is optional as the script will handle this anyway.

#### 3. Edit the configuration file
This step is also optional. If you skip this step, the dataset will be saved on the default directory at `~/dataset`. However, if you feel like this is not a good place to store the dataset, please feels free to override save path on [`config.yaml`](conf/config.yaml).

#### 4. Run the generation script
To create the dataset, run:
```bash
python generate_datset.py
```
### Dataset Configuration
The [`config.yaml`](conf/config.yaml) receives the following keys:
|Key|Type|Description|
|:-:|:--:|:----------|
|`num_workers`|int|Number of workers used to generate the dataset. Use -1 for all CPU.|
|`sampling_rate`|int|Sampling frequency of generated file.|
|`available_notes`|List[str]|List of all notes that will be generated.|
|`available_program`|List[int]|List of program audio that will be included in the generation. See more info at [MidiProgram class](midi_enum.py)|
|`sample_per_program_note`|int|Number of samples for each specified note/program.|

Note that the total number of samples generated is equal to `len(available_notes) * len(available_program) * samples_per_program_note`.

## Usage: Training Baseline
To begin with, download or clone this repository. This can be done by running the following command:
```bash
git clone https://github.com/vistec-AI/SynthMIDI.git
cd SynthMIDI
```

### Training on Docker

#### 1. Building Docker
```bash
docker build -t synthmidi/trainer -f train.Dockerfile .
```
Once finished, you should have a built image on your docker. Run `docker images` to see if the image was built properly.

#### 2. Start Training
Once you have built your docker, run the following commands:
```bash
docker run -it -v $PWD/conf:/workspace/conf -v /path/to/saved/weights:/workspace/weights --shm-size=16G --name synthmidi-trainer synthmidi/trainer
```

### Training on local
#### 1. Activate Virtual Environment (Optional)
Make sure you use python 3.8.x!
```bash
python3 -m venv dataset.venv  # create virtual environment
source dataset.venv/bin/activate  # activate virtual environment
```

#### 2. Install requirements
```bash
pip install -r train.requirements.txt
```

#### 3. Start Training
Make sure your [config file](conf/baseline.yaml) was setup according to your environment. Once you finished setting up config file, run:
```bash
python train.py
```
Then, you're done!

### Training Configuration
See an example at [`baseline.yaml`](conf/baseline.yaml)

## Baseline
We construct two baseline for this experiment:
1. Stratify splitting - Simply apply stratify splitting the dataset.
2. Leave instrument out - Leaving specific instruments out as a testing data. This is way more challenging.

The split file and the whole dataset was available on [dataset section](#dataset)

\*For **Leave instrument out** setting, it's better to conduct it in a k-fold cross validation setting, but I'll leave this as an open contribution.

### Results
|Test Set|Accuracy|Precision|Recall|F1-Score|
|:------:|:------:|:-------:|:----:|:------:|
|Stratify spliting|97.00%|97.05%|97.02%|97.02%|
|Leave instrument out|64.30%|69.86%|64.33%|64.81%|

*Precision, Recall, and F1 are computed using macro average

## Contributing
There're several improvements that can be done and I'll be add several feature needed in the `issue` pannel. If you wish to contribute, fork this repository, and open the pull request.

## Author
Chompakorn Chaksangchaichot

## Acknowledgement
This works is made possible with several helps and guidance from [Noppawat Tantisiriwat](https://github.com/Noppawat-Tantisiriwat). Also, Kudos to the AI Builder Program and Aj. Ekapol Chuangsuwanich for making this dataset possible.
