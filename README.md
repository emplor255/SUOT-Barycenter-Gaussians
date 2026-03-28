# On Barycenter Computation: Analyzing Semi-Unbalanced Optimal Transport-based Method on Bures-Wasserstein manifold
This is the official implementation of [On Barycenter Computation: Analyzing Semi-Unbalanced Optimal Transport-based Method on Bures-Wasserstein manifold] - Accepted Paper at AISTATS 2026


## Installation
Using conda environment is recommended:
```
conda create -n myenv python=3.10
conda activate myenv
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running the codes
1. Drawing barycenter 

<p align="center">
  <img src="SUOT_Barycenter\Image\demo_1.png" alt="Demo" width="850">
</p>

To draw the barycenter in Section 5, run this jupyter notebook:

```
/SUOT-Based Barycenter/visualize.ipynb
```

2. Estimate convergence

<p align="center">
  <img src="SUOT_Barycenter\Image\demo_2.png" alt="Demo" width="800">
</p>

To estimate convergence of optimization methods, run this Python file

``` 
/SUOT-Based Barycenter/estimate_convergence.py
```


