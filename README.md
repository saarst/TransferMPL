<h2 align="center">TransferMPL-Roleplaying</h2> 
<h2 align="center">Final project for the Technion's EE Deep Learning course (046211)</h2> 
<h4 align="center">Implementation of Meta Pseudo Labels with "Role-Playing" and Transfer Learning.</h4> 


  <p align="center">
    Shira Lifshitz: <a href="https://www.linkedin.com/in/shira-lifshitz-313328248/">LinkedIn</a> , <a href="https://github.com/ShiraLifshitz">GitHub</a>
  <br>
    Saar Stern: <a href="https://www.linkedin.com/in/saar-stern-a43413246/">LinkedIn</a> , <a href="https://github.com/saarst">GitHub</a>
  </p>

Based on paper:
Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, Quoc V. Le [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)

- [TransferMPL with Roleplaying](#TransferMPL-Roleplaying)
  * [Background](#background)
  * [Results](#results)
  * [Installation](#installation)
  * [Files in the repository](#files-in-the-repository)
  * [API (`MPL.py --help`)](#api-mplpy---help)
  * [Usage](#usage)
  * [References](#references)

## Background
In semi-supervised setting we are given a dataset with low percent of labels (e.g. 3%) . MPL is an algorithm for such case:
The idea of this algorithm is to train a teacher model using "supervised loss" (with labels) , "self-supervised loss" (uda, which uses weak and strong augmentations) , and "semi-supervised loss" derived from a student model, that learns from Psuedo Labels (teacher's predictions)

![alt text](https://github.com/saarst/TransferMPL/blob/main/assets/MPL.png)

Figure taken from original paper

This design is asymetric, so we introduce Role-Playing, basically switch positions between the student and the teacher in the training phase. Eventually using ensemble learning to use both models.
We also wanted to examine:
1. The 16-classes flowers dataset, with 3% percent labels.
2. Negative cosine similarity criterion instead of CE.
3. Optuna hyper-parameter tuning.
4. Different augmentations from the original paper.

## Results
On [flowers dataset](https://www.kaggle.com/datasets/846e29ea90553aba96640836491fe6099a5ec3b31bbfd7c72dce4ca070dcffa9) with 3% labels, using RolePlaying and Ensemble:

<img src="https://github.com/saarst/TransferMPL/blob/main/assets/flowers.png" data-canonical-src="https://github.com/saarst/TransferMPL/blob/main/assets/flowers.png" width="750" height="150" />

<img src="https://github.com/saarst/TransferMPL/blob/main/results/switch_2023-01-20%2009-36-21/Both%20models_CM.png" data-canonical-src="https://github.com/saarst/TransferMPL/blob/main/results/switch_2023-01-20%2009-36-21/Both%20models_CM.png" width="750" height="550" />

Test set accuracy : 68.22% , improvement of 13.35% from baseline (only labels)


## Installation

Clone the repository and run:
```
$ conda env create --name TransferMPL --file env.yml
$ conda activate TransferMPL
$ python MPL.py
```
## Files in the repository

| File name                                                     | Purpsoe                                                                                                                                       |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `MPL.py`                                                      | general purpose main application                                                                                                              |
| `models.py`                                               | create the models.                                                                                                  |
| `optuna.py`                                               | optuna optimizatoin script                                                                                               |
| `data.py`                                                 | loaders, splits and augmentations.                                                                             |
| `args.py`                                                 | arguments parser                                            |
| `utils.py`                                                | utils                                                 |
| `train.py`                                                    | training functions                                                                                                    |
| `visualizatoin.py`                                        | plot graphs, confusion matrix and etc.                                       



## API (`MPL.py --help`)

You should use the `MPL.py` file with the following arguments:

|Argument                 | Description                                 |
|-------------------------|---------------------------------------------|
| -h, --help            show this help message and exit
|  --name |           experiment name
|  --data_dir |   data path (must start with /datasets , for e.g. /datasets/flowers)
|  --load_path |  folder in /checkpoints/ folder to load "best_student" from 
|  --num_labels_percent  | percent of labeled data
|  --num_epochs | number of epochs to run
|  --warmup_epoch_num | warmup steps
|  --model_name | model name for feature extracting (e.g vgg16)
|  --unsupervised | loss for unsupervised, can be "CE" or "cos"
|  --seed |           seed for initializing training
|  --threshold | pseudo label threshold
|  --temperature | pseudo label temperature
|  --lambda_u |   coefficient of unlabeled loss
|  --uda_steps | warmup steps of lambda-u
|  --show_images    |     show samples from dataset and sample from augmented dataset
|  --load_best      |     load best model to that dataset
|  --print_model    |     print the model we are training
|  --optuna_mode    |     for running optuna
|  --test_mode      |     only test
|  --switch_mode     |    switch models every epoch
|  --finetune_mode   |    only finetune model on labeled dataset
|  --n_trials  |  n_trials for optuna
|  --timeout    |  timeout [sec] for optuna


## Usage

1. Download [flowers dataset](https://www.kaggle.com/datasets/846e29ea90553aba96640836491fe6099a5ec3b31bbfd7c72dce4ca070dcffa9) and put in /datasets/flowers
2. Run with `MPL.py --name NameOfExperiment --data_dir /datasets/flowers`
3. Finetune with `MPL.py --name finetune --finetune_mode --load_best --data_dir /datasets/flowers --load_path /checkpoints/flowers/NameOfExperiment`
4. Results are in `/results/NameOfExperiment`

## References

* [flowers dataset](https://www.kaggle.com/datasets/846e29ea90553aba96640836491fe6099a5ec3b31bbfd7c72dce4ca070dcffa9) 
* Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong, Quoc V. Le [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580)
* [https://github.com/sally20921/Meta_Pseudo_Labels](https://github.com/sally20921/Meta_Pseudo_Labels) - pytorch implementation of the original paper
* [https://github.com/ifsheldon/MPL_Lightning](https://github.com/ifsheldon/MPL_Lightning) - another pytorch and Lightning implementation of the original paper
* Leslie N. Smith, Nicholay Topin [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
