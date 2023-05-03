# Lightweight Multi-Branch Network for Animal Re-identification

Pytorch implementation for an Animal Re-Identification system based on the paper "Lightweight Multi-Branch Network for Person Re-Identification"

[![Paper](https://img.shields.io/badge/arXiv-2101.10774-important)](https://arxiv.org/abs/2101.10774)

![LightMB](/utils/LightMB.png)

## Most important contributions

Essential modifications from the original code include:

- Added options for mask usage, frame dropping, query-gallery-sepation, test on pretrained model (0. epoch)
- Added testing on 0. epoch and added this to log and performance graph generation
- Fixed visualization of activation maps
- Fixed cython test for faster testing (gpu instead of cpu)
- Added model architectures lmbn_only_global, lmbn_only_channel, lmbn_only_part. Modified network architecture to only use specific branch and forward the right features for inference.
- Added activation map output of standard lmbn_n model and fixed drop block. Drop block applied before activation is returned, not afterwards.
- ImageDataset classes baywald.py and wildpark.py including video dataset processing.

## Results

![1](/doc/images/experimentelle-durchfuehrung.png)
![2](/doc/images/trainingsparameter.png)
![3](/doc/images/mAP.png)
![4](/doc/images/activation_maps.png)
![5](/doc/images/similarity_rankings.png)
![6](/doc/images/architecture.png)
![7](/doc/images/frame_dropping.png)
![8](/doc/images/frame_dropping_mAP.png)
![9](/doc/images/separation.png)
![10](/doc/images/architecture.png)
![11](/doc/images/masks.png)

## Details

This repo supports easy dataset preparation, including Market-1501, DukeMTMC-ReID, CUHK03, MOT17, sota deep neural networks and various options(tricks) for reid, easy combination of different kinds of loss function, end-to-end training and evaluation and less package requirements.

List of functions

- Warm up cosine annealing learning rate
- Random erasing augmentation
- Cutout augmentation
- Drop Block and Batch Erasing
- Label smoothing(Cross Entropy loss)
- Triplet loss
- Multi-Simulatity loss
- Focal loss
- Center loss
- Ranked list loss
- Group Loss
- Different optimizers
- Attention modules
- BNNeck

Inplemented networks:

- Lightweight Multi-Branch Network(LightMBN), which we proposed
- PCB [[link]](https://arxiv.org/pdf/1711.09349.pdf)
- MGN [[link]](https://arxiv.org/abs/1804.01438)
- Bag of tricks [[link]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)
- OSNet [[link]](https://arxiv.org/abs/1905.00953)
- Batch Drop Block(BDB) for Person ReID [[link]](https://arxiv.org/abs/1811.07130)

## Getting Started

The python notebook can be uploaded to Google Colab and used instantly. Explanation on how to run the code and reproduce results can be found in the notebook. In the notebook, everything will be downloaded automatically from my GDrive without any registration. For independence from my GDrive, prepare required files for the notebook manually. Required are the folders BayWaldDataset, WildparkDataset and LightMBN. Note that the code it not adjusted for expired Google Drive connections and links for downloading this repo in a Colab machine as well as the animal datasets.

For more information, please contact me.


### BayWaldDataset and WildparkDataset

The folders BayWaldDataset and WildparkDataset contain annotation files and a csv file that defines the train-test split.

### Pre-trained models

and correpondent config files can be found [here](https://1drv.ms/u/s!Ap1wlV4d0agrao4DxXe8loc_k30?e=I9PJXP) .

If you have pretrained model and config file, run

```bash
python [path to repo]/main.py --test_only --config [path to repo]/lmbn_config.yaml --pre_train [path to pretrained model]
```
to see the performance of the model.

If you would like to re-inplement Bag of Tricks, run

```bash
python [path to repo]/main.py --datadir [path to datasets] --data_train market1501 --data_test market1501 --model ResNet50 --batchid 16 --batchimage 4 --batchtest 32 --test_every 10 --epochs 120 --save '' --decay_type step_40_70 --loss 0.5*CrossEntropy+0.5*Triplet --margin 0.3 --nGPU 1 --lr 3.5e-4 --optimizer ADAM --random_erasing --warmup 'linear' --if_labelsmooth
```

or

```bash
python [path to repo]/main.py --config [path to repo]/bag_of_tricks_config.yaml --save ''
```

If you would like to re-inplement PCB with powerful training tricks, run

```bash
python [path to repo]/main.py --datadir [path to datasets] --data_train Market1501 --data_test Market1501 --model PCB --batchid 8 --batchimage 8 --batchtest 32 --test_every 10 --epochs 120 --save '' --decay_type step_50_80_110 --loss 0.5*CrossEntropy+0.5*MSLoss --margin 0.7 --nGPU 1 --lr 5e-3 --optimizer SGD --random_erasing --warmup 'linear' --if_labelsmooth --bnneck --parts 3
```

Note that, the option '--parts' is used to set the number of stripes to be devided, original paper set 6.

And also, for MGN model run

```bash
python [path to repo]/main.py --datadir [path to datasets] --data_train Market1501 --data_test Market1501 --model MGN --batchid 16 --batchimage 4 --batchtest 32 --test_every 10 --epochs 120 --save '' --decay_type step_50_80_110 --loss 0.5*CrossEntropy+0.5*Triplet --margin 1.2 --nGPU 1 --lr 2e-4 --optimizer ADAM --random_erasing --warmup 'linear' --if_labelsmooth
```

### Resume Training

If you want to resume training process, we assume you have the checkpoint file 'model-latest.pth', run
```
python [path to repo]/main.py --config [path to repo]/lmbn_config.yaml --load [path to checkpoint]
```
Of course, you can also set options individually using argparse command-line without config file.


## Option Description

'--nThread': type=int, default=4, number of threads for data loading.

'--cpu', action='store_true', if raise, use cpu only.

'--nGPU', type=int, default=1, number of GPUs.

--config', type=str, default="", config path,if you have config file,use to set options, you don't need to input any option again.

'--datadir', type=str, is the dataset root, which contains folder Market-1501, DukeMTMC-ReID etw..

'--data_train' and '--data_test', type=str, specify the name of train/test dataset, which we can train on single or multiple datasets but test on another datasets, supported options: market1501, dukemtmc, MOT17, cuhk03_spilited(767/700 protocol) or e.g. market1501+dukemtmc.

'--batchid 6' and '--batchimage 8': type=int, indicate that each batch contrains 6 persons, each person has 8 different images, totally 48 images.

'--sampler', type=str,default='True', if 'True', sample batchid persons and batchimage in a batch, else, ramdom selected totally batchid\*batchimage in a batch.

''--batchtest', type=int, default=32, total batch size for evaluation.

'--test_only', action='store_true', if raise, only run the evaluation.

'--save', type=str, default='test', name of the folder to save output, if '', then it will create the name using current time.

'--load', type=str, default='', name of the output folder, if there is a checkpoint file in the folder, it will resume trainning.

'--pre_train', type=str, default='', path of pre-trained model file.

'--epochs', type=int, is the epochs we'd like to train, while '--test_every 10' means evaluation will be excuted in every 10 epochs, the parameters of network and optimizer are updated after every every evaluation. 

'--model', default='LMBN_n', name of model, options: LMBN_n, LMBN_r,  ResNet50, PCB, MGN, etw..

'--loss', type=str, default='0.5\*CrossEntropy+0.5\*Triplet', you can combine different loss functions and corresponding weights, you can use only one loss function or 2 and more functions, e.g. '1\*CrossEntropy', '0.5\*CrossEntropy+0.5\*MSLoss+0.0005\*CenterLoss', options: CrossEntropy, Triplet, MSLoss, CenterLoss, Focal, GroupLoss.

'--margin', type=float, margin for Triplet and MSLoss.

'--if_labelsmooth', action='store_true', if raise, label smoothing on.

'--bnneck', action='store_true', if raise, use BNNeck, only for ResNet and PCB.

'--drop_block', action='store_true', if raise, use Batch Drop Block, and '--h_ratio 0.3 and --w_ratio 1.0' indicate the erased region on the feature maps. 

'--pool', type=str, default='avg', choose pooling method, options: avg, max.

'--feats', type=int, default=256, dimension of feature maps for evaluation.

'--height', type=int, default=384, height of the input image.

''--width', type=int, default=128, width of the input image.

'--num_classes', type=int, default=751, number of classes of train dataset, but normally you don't need to set it, it'll be automatically setted depend on the dataset.

'--lr', type=float, default=6e-4, initial learning rate.

'--gamma', type=float, default=0.1,learning rate decay factor for step decay.

'--warmup', type=str, default='constant', learning rate warmup method, options: linear, constant.

'--w_cosine_annealing', action='store_true', if raise, use warm up cosine annealing learning rate scheduler.

'--pcb_different_lr', type=str, default='True', if 'True', use different lr only for PCB, if lr is 5e-3, then lr for classifier is 5e-3, lr for other part is 5e-4.

'--optimizer, default='ADAM', options: 'SGD','ADAM','NADAM','RMSprop'.

'--momentum', type=float, default=0.9, SGD momentum.

'--nesterov', action='store_true', if raise, SGD nesterov.

'--parts', type=int, default=6, is used to set the number of stripes to be devided, original paper set 6.

'--re_rank', action='store_true', 'if raise, use re-ranking.

'--cutout', action='store_true', if raise, use cutout augmentation.

'--random_erasing', action='store_true', if raise, use random erasing augmentation.

'--probability', type=float, default=0.5, probability of random erasing.

'--T', type=int, default=3, number of iterations of computing group loss.

'--num_anchors', type=int, default=1, number of iterations of computing group loss.
