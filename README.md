# deep-learning-basics

Run this code with Jupyter Binder:
https://mybinder.org/v2/gh/athms/deep-learning-basics/HEAD

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/athms/deep-learning-basics/HEAD)

## Usage

### Run the Jupyter notebooks locally

1. Clone this repo
2. Install the required packages listed in [`requirements.txt`](requirements.txt), ideally in a virtual environment, e.g.:

```bash
$ mkvirtualenv deep-learning-basics -p python3 -r requirements.txt
```

This command used the [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/index.html) to create a virtual environment called `deep-learning-basics` with the locally installed version of Python3 (here, Python 3.8.6, see below) and install all required packages listed in [`requirements.txt`](requirements.txt).

```bash
$ python3 --version
Python 3.8.6
```

3. To run the Jupyter kernel inside the virtual environment, you need to run the kernel self-install inside the virtual environment:

```bash
$ python -m ipykernel install --user --name=deep-learning-basics
```

4. To start the Jupyter interface, run

```
jupyter notebook
```

4. Finally, switch the kernel (named `deep-learning-basics` here) in the Jupyter user interface.
