# LanGuideMedSeg-MICCAI2023
Pytorch code of MICCAI 2023 Paper-Ariadne’s Thread : Using Text Prompts to Improve Segmentation of Infected Areas from Chest X-ray images

ArXiv paper: [https://arxiv.org/abs/2307.03942](https://arxiv.org/abs/2307.03942)

## Preparation
1. Environment
Python==3.8
Pytorch==1.13.1
pytorch_lightning=1.9.1
monai

## Dataset
1. QaTa-COV19 Dataset(images & segmentation mask)
QaTa-COV19 Dataset See Kaggle: [https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

*We use QaTa-COV19-v2 in our experiments.*

2. QaTa-COV19 Text Annotations(from thrid party)
