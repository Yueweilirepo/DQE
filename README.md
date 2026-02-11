<h1 align="center">DQE: A Dual-Quality Evaluation Metric for Time Series Anomaly Detection</h1>


This repository contains the code for DQE (Dual-Quality Evaluation Metric), a novel evaluation metric for assessing anomaly detection in time series data. 
DQE is designed around the principles of locality and
integrality, which holistically synthesizes true positive quality (TQ) and true positive quality (FQ), enabling a fine-grained evaluation of timely, early and late detections.
The methodology is detailed in our paper, demonstrating that DQE provides more reliable and
discriminative evaluations through experiments with both synthetic and real-world datasets.

## Environment

### Environment Setup
To use DQE, start by creating and activating a new Conda environment using the following commands:

```bash
conda create --name dqe_env python=3.9
conda activate dqe_env
```


### Install Dependencies

Install the required Python packages via:

```bash
pip install -r requirements.txt
```



## How to use DQE? 

Begin by importing the `DQE` module in your Python script:


```bash
from dqe.dqe_metric import DQE
```

Prepare your input as arrays of anomaly scores (continues or binary) and binary labels. 

Example usage of DQE:

```bash
dqe = DQE(labels, scores)
```

### Basic Example

```python 
import numpy as np
from dqe.dqe_metric import DQE

# Example data setup
labels = np.array([0, 1, 0, 1, 0])
scores = np.array([0.1, 0.8, 0.1, 0.9, 0.2])

# Compute DQE

dqe = DQE(labels, scores)

print(dqe)
```
For the single-threshold DQE-F1, 
begin by importing the `DQE_F1` module in your Python script:

```bash
from dqe.dqe_metric import DQE_F1
```

Example usage of DQE-F1:

```bash
dqe_f1 = DQE_F1(labels, detections)
```

### Basic Example

```python 
import numpy as np
from dqe.dqe_metric import DQE_F1

# Example data setup

labels = np.array([0, 1, 0, 1, 0])
detections = np.array([0, 1, 0, 1, 1])

# Compute DQE-F1

dqe_f1 = DQE_F1(labels, detections)

print(dqe_f1)
```

DQE and DQE-F1 allow for comprehensive customization of parameters by parameter `parameter_dict`.
Please refer to the main code documentation for a full list of configurable options.

---

## Experiments
For researchers interested in reproducing the experiments or exploring the evaluation metric further with various data sets:

### with Synthetic Data

To run experiments on synthetic data, navigate to the `experiments` directory and execute the Python script `synthetic_data_exp.py`.
This script allows for the modification of various scenarios, comparing DQE against other established metrics.


```bash
python synthetic_data_exp.py --exp_name "over-counting tp"
```

The parameter `exp_name` can be set to one of the following values: 
["over-counting tp", "tp timeliness", "fp proximity", "fp insensitivity or overevaluation of duration", "fp duration overevaluation (af)"].


### with Real-World Data

#### Download the Dataset
The real-world datasets for experiments can be downloaded from the following link:

Dataset Link: https://www.thedatum.org/datasets/TSB-AD-U.zip 

Ref: This dataset is made available through the GitHub page of the project "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (TSB-AD)": https://github.com/TheDatumOrg/TSB-AD


#### Running the Experiments

After downloading, place the unzipped dataset in the directory `dataset`. If you store the data in a different location, ensure you update the directory paths in the code to match.

 Navigate to the `experiments` directory and execute the Python script `get_algorithms_outputs.py` for producing algorithms' outputs by entering the following command:

```bash
python get_algorithms_outputs.py
```

Execute the Python script `real_data_exp_case.py` for producing case results in paper by entering the following command:


```bash
python real_data_exp_case.py --exp_name "YAHOO case"
```
The parameter `exp_name` can be set to one of the following values: ["YAHOO case", "WSD case", "partition strategy", "detection rate", "weighting strategy"]. 

In experiments of design strategies, parameter `exp_name` are used in conjunction with the parameter `parameter_dict` in by DQE entering the following command:.
```bash
python real_data_exp_case.py --exp_name "partition strategy" --design_strategy "whole"
```

The parameter `design_strategy` can be set in: ["whole", "split"] for "partition strategy", ["use detection rate", "not use detection rate"] for "detection rate", ["triangle", "equal"] for "weighting strategy".

Execute the Python script `real_data_exp_all_dataset.py` for producing average results in paper by entering the following command:


```bash
python real_data_exp_all_dataset.py
```