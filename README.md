# PFG: Preconditioned Functional Gradient Flow

> This repository contains the codes for [Particle-based Variational Inference with Preconditioned Functional Gradient Flow](https://arxiv.org/abs/2211.13954)  
by Hanze Dong, Xi Wang, Yong Lin, Tong Zhang.


Docs can be found in `build/index.html`.

## Setup

Our code works with the following environment.

`notebook`

`torch`

### Installation

```
pip install -r requirements.txt
python setup.py install
```



## Data preparation

In our experiments, the data are placed at `./data`. 

UCI datasets are downloaded from https://archive.ics.uci.edu/ml/datasets.php.

For your own data, you can refer to the format of UCI data and establish corresponding dataloader.


## Sampling tasks

In this repo, we have several examples to demonstrate the effectiveness of our algorithm.

### Ill-conditioned Gaussian distribution

For ill-conditioned Gaussian distribution, we show that the preconditioning matters in the sampling algorithm, which accelerate the convergence significantly.
```
cd examples
ipython notebook ill_Gaussian.ipynb
```


### Gaussian Mixture Model

For Gaussian Mixture Model, the function class of our model is more powerful than kernel function class, due to the non-linearity included.
```
cd examples
ipython notebook Gaussian_mixture_10.ipynb
```


### Bayesian Logistic Regression

For Bayesian Logistic regression, we provide demo for `sonar` dataset, which is already included in `./data`.
```
cd examples
python bayesian_logistic_regression.py --hdim 32 --inner_iter 5 --num_particles 200
```


### Bayesian Neural Networks

For Bayesian Neural Networks, we provide a demo for `boston_housing` dataset.
```
cd examples
python bayesian_nn.py --hdim 32 --inner_iter 1 --num_particles 200
```



## Contact

If you meet any problem in this repo, please describe them and contact:

Hanze Dong: A (AT) B, where A=hdongaj, B=ust.hk.

