## Requirement
install the conda environment using the requirements.txt file

```
conda create -n jt python=3.9
pip install -r requirements.txt
```

## Pre-trained Models

Please download pre-trained ViT-Base models from [MoCo v3](https://drive.google.com/file/d/1bshDu4jEKztZZvwpTVXSAuCsDoXwCkfy/view?usp=share_link) and 
then put or link the pre-trained models to ```./pretrained/```

## Dataset Preparation

Please download CIFAR-100 and then put or link the dataset to ```./data/```

## Training
to launch the training of CoFiMA on CIFAR-100, run the following command:

```python main.py```
