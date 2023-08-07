## Transfer Learning for the Visual Arts: The Multi-Modal Retrieval of Iconclass Codes


This repository contains all the code used in the following [journal paper]():

> Nikolay Banar, Walter Daelemans, Mike Kestemont, "Transfer Learning for the Visual Arts: The Multi-Modal Retrieval of Iconclass Codes". *Journal on Computing and Cultural Heritage* (2022).

The following commands can be used to train or test the models:

CUDA_VISIBLE_DEVICES=0,1 python train.py --conf config_experiments/both_finetuning/ide2c_L19_L10.json
CUDA_VISIBLE_DEVICES=0,1 python test.py --conf config_experiments/both_finetuning/ide2c_L19_L10.json

The short tutorial is [here](https://drive.google.com/drive/folders/16kKuUYlx1_5oiJVHitqH9M2iWM2EMySx?usp=sharing).

```
@inproceedings{banar2022multi, title={Transfer Learning for the Visual Arts: The Multi-Modal Retrieval of Iconclass Codes.}, author={Banar, Nikolay and Daelemans, Walter and Kestemont, Mike}, booktitle={Journal on Computing and Cultural Heritage}, pages={to be published}, year={2022}}
```
