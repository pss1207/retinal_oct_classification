# Retinal Optical Coherence Tomography Classification
Pytorch implementation for retinal OCT images of Kaggle

## Model
![alt text](https://github.com/pss1207/retinal_oct_classification/blob/master/model.png)
## Results
![alt text](https://github.com/pss1207/retinal_oct_classification/blob/master/test_results.png)
![alt text](https://github.com/pss1207/retinal_oct_classification/blob/master/confusion_matrix.png)

## Prerequisites
- Pytorch
- [Retinal OCT Image Dataset](https://www.kaggle.com/paultimothymooney/kermany2018/data)

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```

### Retinal OCT Image Dataset
- To train the model, modify the dataset path in train.py 
```bash
data_dir = 'dataset path'
```

### Train
- Train a model:
```bash
python train.py 
```

### Test
- Test the model:
```bash
python test.py
```
