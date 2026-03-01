<h1 align="center">DQE: A Semantic-Aware Evaluation Metric for Time Series
Anomaly Detection</h1>


[//]: # (This repository contains the code for DQE &#40;Detection Quality Evaluation metric&#41;, a novel evaluation metric for time series anomaly detection.)

[//]: # ([//]: # &#40;DQE is designed from a detection semantics perspective, which holistically synthesizes anomaly capture quality, near-miss detection quality, and false alarm detection quality, enabling a fine-grained evaluation.&#41;)

This repository contains the implementation of Detection Quality Evaluation (DQE), a novel metric for time series anomaly detection.
DQE is designed from a detection semantics perspective. It holistically synthesizes anomaly capture quality, near-miss detection quality, and false alarm detection quality, enabling fine-grained evaluation.
[//]: # (The methodology is detailed in our paper, demonstrating that DQE provides more reliable, discriminative, robust, and interpretable evaluations through experiments with both synthetic and real-world data.)
The full methodology is described in our paper, where extensive experiments on both synthetic and real-world data demonstrate that DQE provides reliable, discriminative, robust, interpretable evaluations compared to existing metrics.

## Environment

### Environment Setup
To use DQE, start by creating and activating a new Conda environment using the following commands:

```bash
conda create --name dqe_env python=3.9
conda activate dqe_env
```


### Install Dependencies

Install required Python packages via:

```bash
pip install -r requirements.txt
```



## How to use DQE? 

Begin by importing the `DQE` module in your Python script:


```bash
from dqe.dqe_metric import DQE
```

Prepare your input as arrays of anomaly scores (continuous or binary) and binary labels. 

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

For single-threshold DQE, use the `SDQE` module:

```bash
from dqe.dqe_metric import SDQE
```

Example usage of SDQE:

```bash
sdqe = SDQE(labels, detections)
```

### Basic Example

```python 

import numpy as np
from dqe.dqe_metric import SDQE

# Example data setup

labels = np.array([0, 1, 0, 1, 0])
detections = np.array([0, 1, 0, 1, 1])

# Compute SDQE

sdqe = SDQE(labels, detections)

print(sdqe)
```

[//]: # (DQE and SDQE allow for comprehensive customization of parameters by parameter `parameter_dict`.)

[//]: # (Please refer to the main code documentation for a full list of configurable options.)

---

## Experiments

For researchers interested in reproducing the experiments or exploring the evaluation metric further with various data sets:

[//]: # (For researchers interested in reproducing the experiments in our paper or exploring DQE further, we provide scripts for both synthetic and real-world data.)

### with Synthetic Data

[//]: # (To run experiments on synthetic data, navigate to the `experiments/synthetic_data_exp` directory. and execute the Python script `case_study_exp/case_analysis_exp.py`.)
[//]: # (To run experiments of exiting metric's issues on synthetic data,)
To reproduce the synthetic experiments on existing metric issues,
navigate to the `experiments/synthetic_data_exp/case_study_exp` directory and execute the Python script `case_analysis_exp.py`.
This script allows for the modification of various scenarios, comparing DQE against other established metrics.


```bash
python case_analysis_exp.py --exp_name "anomaly_event_coverage"
```

The parameter `exp_name` can be set to one of the following values: 
["anomaly_event_coverage", "near_miss_proximity", "proximity_inconsistency", "proximity_inconsistency_af", "false_alarm_frequency", "random_case"].

[//]: # (To reproduce the synthetic experiments on effect of point-level coverage bias,)

[//]: # (navigate to the `experiments/synthetic_data_exp/case_study_exp/stability_and_discriminability_exp` directory and execute the Python script `anomaly_event_anomaly_len.py`, `anomaly_event_anomaly_number.py`, and `anomaly_event_anomaly_ratio.py` for the effect of anomaly length, number, and ratio, respectively.)

To reproduce the synthetic experiments analyzing the effect of point-level coverage bias, navigate to `experiments/synthetic_data_exp/case_study_exp/stability_and_discriminability_exp` and execute the following scripts:

- `python anomaly_event_anomaly_len.py` (effect of anomaly length)

- `python anomaly_event_anomaly_number.py` (effect of anomaly number)

- `python anomaly_event_anomaly_ratio.py` (effect of anomaly ratio)

### with Real-World Data

#### Dataset Download
[//]: # (The real-world datasets for experiments can be downloaded from the following link:)
The real-world datasets used in our experiments can be downloaded from:

Dataset Link: https://www.thedatum.org/datasets/TSB-AD-U.zip 

Ref: This dataset is made available through the GitHub page of the project "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark (TSB-AD)": https://github.com/TheDatumOrg/TSB-AD


#### Running the Experiments

After downloading, place the unzipped dataset in the directory `dataset`. If you store the data in a different location, ensure you update the directory paths in the code to match.

 Navigate to the `experiments/real_data_exp/benchmark_evaluation_exp` directory and execute the Python script `get_algorithms_outputs.py` to produce outputs of algorithms by entering the following command:

```bash
python get_algorithms_outputs.py
```

Execute the Python script `get_evaluation_results.py` for producing evaluation scores of different metrics by entering the following command:


```bash
python get_evaluation_results.py
```

Execute the Python script `get_mean_result.py` for producing average results at the time-series level by entering the following command:

```bash
python get_mean_result.py
```

Execute the Python script `case_analysis.py` for producing results of case studies in paper by entering the following command:

```bash
python case_analysis.py --exp_name "UCR case"
```

The parameter `exp_name` can be set to one of the following values: ["UCR case", "WSD case", "AUC-ROC/AUC-PR issue case"]. 


To get the anomaly-level evaluation results of DQE, execute the Python script `anomaly_event_level_evaluation_dqe.py` by entering the following command:

```bash
python anomaly_event_level_evaluation_dqe.py --exp_name "UCR"
```


The parameter `exp_name` can be set to one of the following values: ["UCR", "WSD"]. Another parameter `file_index` which is the file index number in the dataset can be set to get DQE scores for each anomaly by entering the following command:

```bash
python anomaly_event_level_evaluation_dqe.py --exp_name "UCR"  --file_index "452"
```


To reproduce the robustness experiments of existing metrics,
navigate to the `experiments/synthetic_data_exp/robustness_exp` directory and execute the Python script `robustness_exp.py` to produce the outputs of algorithms and scores of metrics across variations.

[//]: # (This script allows for the modification of various scenarios, comparing DQE against other established metrics.)

```bash
python robustness_exp.py
```


To get averaged performance of the evaluation robustness, execute the Python script `robustness_analysis.py`.

[//]: # (This script allows for the modification of various scenarios, comparing DQE against other established metrics.)

```bash
python robustness_analysis.py
```