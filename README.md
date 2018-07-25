# Wooppy Project Cooptatione

This project categorizes a CV and outputs a quantified suitability of the CV for multiple typical corporate departments. The engine views the content of the CV in two forms: text and image. The follow Github repositories models are used to build the engine:

* Text classifier : [Dennybritz text classifier](https://github.com/dennybritz/cnn-text-classification-tf)
* Image classifier : [Inception-v3](https://github.com/tensorflow/models/tree/master/research/inception)

## Getting Started

Let's get it start working on your machine!

### Prerequisites

* Python 3

Please refer to `pyinstall.txt` for detailed list of requirements. If you are a gambling man and a man of trust, you can immediately install all requirements in the list via:
```
pip install -r pyinstall.txt
```

Besides that, do install the requirements for OCR which is listed in a seperate text file `pdf_manipulator/librariesToInstall.txt`. Note that `pip install -r` **WILL NOT** work on this file so please read it through and install one-by-one. Cheers.

p/s: There are further `requirements.txt` located in each subfolder of this project. Do go through them as well...

### Installing
You need to reconfigure the directory of your model checkpoints. There are two checkpoint configuration files, each for the text and image classifier, respectively. They are located at `cnnTextClassifier/runs/1507001278/checkpoints/checkpoint` and `inception/cv_train/multilabel/checkpoint`.

### Retraining of Models
Please refer to documentations of the individual models on how to retrain the the model with your own data.

Do note that to update your lists of labels, remember to update the text file at `training_data/labels.txt`.

## Deployment

The main file to run in this project is `server/rest_server.py`. To run this project using command line just do:

1. Activate your environment (if you have one), the set the project root.

For Windows use:
```
set PYTHONPATH=.
```
For Linux use:
```
export PYTHONPATH=.
```

2. Run the project.
```
python3 server/rest_server.py
```

