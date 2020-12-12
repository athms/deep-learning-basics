# An introduction to the basics of deep learning

This repository contains a reproducible workshop on the basics of deep learning.

Each topic is covered in a separate [Jupyter notebook](https://jupyter.org).


## 1. Running the notebooks

For a general introduction to the Jupyter environment, we recommend [this tutorial](https://www.dataquest.io/blog/jupyter-notebook-tutorial/).

You can either install all required dependencies locally (see below) or run the Jupyter notebooks on  [Jupyter Binder](https://mybinder.org) using the following link: 

https://mybinder.org/v2/gh/athms/deep-learning-basics/HEAD

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/athms/deep-learning-basics/HEAD)


## 2. Local installation

### 2.1. Clone and switch to this repository.

### 2.2. Install all dependencies** listed in [`requirements.txt`](requirements.txt). 

For this we recommend setting up a new Python environment (e.g., with the [miniconda installer](https://docs.conda.io/en/latest/miniconda.html)). 

You can create a new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using the following `bash` command:

```bash
conda create -n deep-learning-basics python==3.8
```

This will create a new Python 3.8 environment with the name `deep-learning-basics` .

Switch to the new environment with:

```bash
conda activate deep-learning-basics
```

Install all required dependencies in this environment with: 

```bash
pip3 install -r requirements.txt
```

### 2.3. Start the Jupyter notebook server:**

```bash
jupyter notebook
```

## 3. Course content

So far, this workshop covers the following topics:

0. [`Perceptrons and gradient descent`](0-Perceptron-Gradient-Descent.ipynb)
1. [`Neural networks and backpropagation`](1-Neural-Networks-Backpropagation.ipynb)
2. [`Computer vision with convolutional neural networks`](2-Convolutional-Neural-Networks.ipynb)

We are hoping to expand on this over time, adding introductions to: 

- Recurrent neural networks
- Explainable artificial intelligence
- Deep reinforcement learning
- Supervised, unsupervised, and self-supervised deep learning


## 4. Workshop requests

If you're interested in this workshop and would like to host it at your university (or other institution), email us at: athms.research@gmail.com 