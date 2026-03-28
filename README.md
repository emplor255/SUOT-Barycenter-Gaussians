# On Barycenter Computation: Analyzing Semi-Unbalanced Optimal Transport-based Method on Bures-Wasserstein manifold
This is the official implementation of [On Barycenter Computation: Analyzing Semi-Unbalanced Optimal Transport-based Method on Bures-Wasserstein manifold]


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

<img src="SUOT_Barycenter\Image\demo_1.png" alt="Demo" width="600">

To draw the barycenter in Section 5, run this jupyter notebook:

```
/SUOT-Based Barycenter/visualize.ipynb
```

2. Estimate convergence

<img src="SUOT_Barycenter\Image\demo_2.png" alt="Demo" width="600">

To estimate convergence of optimization methods, run this Python file

``` 
/SUOT-Based Barycenter/estimate_convergence.py
```


