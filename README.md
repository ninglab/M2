# The PPT model for Basket Recommendation
The implementation of the paper:

$M^2$: Mixed Models with Preferences, Popularities and Transitions for Next-Basket Recommendation

Arxiv: https://arxiv.org/abs/2004.01646

Author: Bo Peng (peng.707@buckeyemail.osu.edu)

**Feel free to send me an email if you have any questions.**

## Environments

- python 3.7.3
- PyTorch (version: 1.4.0)
- numpy (version: 1.16.2)
- scipy (version: 1.2.1)
- sklearn (version: 0.20.3)


## Dataset and Data preprocessing:

Please download the data from the url in the paper and refer to the scripts/preprocessing\_TaFeng.py script for the data preprocessing.
We also uploaded all the processed datasets in the processed\_data.zip for the seek of reproducibility. 

## Training
Please refer to the scripts/create\_jobs\_github.sh script for hyper parameter tuning. This script could generate different jobs 
for different hyper parameter configurations. After generating jobs, you could run multiple jobs parallely using perl scripts (e.g. Drone.pl).

Please refer to the following example on how to train and evaluate the model (you are strongly recommended to run the program on a machine with GPU):

```
python main.py --dataset=TaFeng --decay=0.6 --l2=1e-2 --dim=32 --numIter=150 --model=SNBR --isTrain=1 --k=0 --testOrder=1 --isPreTrain=0 --batchSize=100 --mode='time_split'
```
