# The PPT model for Basket Recommendation
The implementation of the paper:

METHOD: Mixed Models with Preferences and Transitions for Next-Basket Recommendation

Arxiv: https://arxiv.org/abs/2004.01646

Author: Bo Peng (peng.707@buckeyemail.osu.edu)

**Feel free to send me an email if you have any questions.**

```

## Environments

- python 3.7.3
- PyTorch (version: 1.4.0)
- numpy (version: 1.16.2)
- scipy (version: 1.2.1)
- sklearn (version: 0.20.3)


## Dataset and Data preprocessing:

Please refer to the "Datasets" Section in the paper for the details of the datasets and preprocessing procedure.
We uploaded the processed TaFeng datasets for the seek of reproducibility. 
Please feel free to contact me if you need more preprocessed data

## Example

Please refer to the following example on how to train and evaluate the mode (you are strongly recommended to run the program on a machine with GPU):

```
python main.py --dataset=TaFeng --decay=0.6 --l2=1e-2 --dim=32 --numIter=150 --model=SNBR --isTrain=1 --k=0 --testOrder=1 --isPreTrain=0 --batchSize=100
```
