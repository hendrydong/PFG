.. PFG documentation master file, created by
   sphinx-quickstart on Wed Oct 12 13:39:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PFG's documentation!
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Introduction
------------

Preconditioned functional gradient flow (PFG) is a particle-based sampling framework, which aims to minimize the KL divergence between particle samples and the target distribution with the gradient flow estimates within a parametric function class.


Installation
------------

Install the requirements with ::

   pip install -r requirements.txt


This package can be be installed from sources with the following command::

    python setup.py install


Description
------------

In this work, we mainly have two parts to construct our package: ``pfg.sampler`` and ``pfg.tasks``.

``pfg.tasks`` defines the unnormalized log posterior density conditioned on a dataset, where the log density can be queried by subsampling (stochastic estimation).


``pfg.sampler`` contains the particle samplers, including PFG, SVGD, SGLD. Given an unnormalized density, the sampler will produce several sample particles from the corresponding distribution.


Usage
------------

To use our code, we provide a standard procedure to use PFG framework to obtain particle samples

**Step 1:** 
Import data ``X_train`` and ``y_train``;


**Step 2:** 
Construct a model by feeding data to a task, e.g. Bayesian Neural Networks.

.. code-block:: python

   model = pfg.tasks.BayesianNN(X_train, y_train, batch_size,
                                    num_particles, hidden_dim)

**Step 3:**   
Initialize a sampler trainer.

.. code-block:: python

   # Initialize particles
   theta = torch.randn(...)

   # For PFG, we have to define the function class first
   activation = nn.Sigmoid()
   net = nn.Sequential(nn.Linear(n_features, h), 
                        activation, nn.Linear(h, h),
                        activation, nn.Linear(h, n_features)) 

   # Define trainer
   trainer = sampler.PFG(theta, lr, net, optim = opt)


**Step 4:**   
Train a sampler.

.. code-block:: python

   for epoch in range(ITERATION+1):    
      trainer.compute_grad(model)
      trainer.step()


**Step 5:**   
Return particles ``theta``.


Examples
------------
We provide several examples to show how to use our package in your problem.

You may refer to ``examples/`` for more details.


API
------------

.. toctree::
   :maxdepth: 1
   
   api






Citing
------

If this software is useful for you, please consider citing
`our paper <https://arxiv.org/abs/2211.13954>`_ that describes
the PFG framework:

.. code-block:: bibtex

      @inproceedings{
         dong2023particlebased,
         title={Particle-based Variational Inference with Preconditioned Functional Gradient Flow},
         author={Hanze Dong and Xi Wang and Yong Lin and Tong Zhang},
         booktitle={International Conference on Learning Representations},
         year={2023},
         url={https://openreview.net/forum?id=6OphWWAE3cS}
      }


Support
-------

If you are having issues, please let us know and send email to hdongaj AT ust.hk.




Indices and tables
------------

* :ref:`genindex`
* :ref:`search`

