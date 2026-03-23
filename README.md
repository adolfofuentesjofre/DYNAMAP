DYNAMAP Project

This repository contains the DYNAMAP model developed in Python 3.9.16. The following diagram includes only Python files (.py), Jupyter Notebook files (.ipynb), and folders.

    └── DYNAMAP
        ├── Brazil and France
        ├── Chile First Cycle
        ├── replication_sample
        ├── data
        │   ├── labels
        ├── Library
        │   ├── ranking
        │   ├── botprediction.py
        │   ├── preferences.py
        │   └── space.py
        └── requirements.txt 
Folders:
<em><b>Brazil and France</b></em>: This folder contains the predictions of the DYNAMAP model and the analysis of results for voting processes conducted in France and Brazil.
<em><b>Chile First Cycle</b></em>: This folder contains the predictions of the DYNAMAP model and the analysis of results for the first cycle of voting conducted in Chile.
<em><b>replication_sample</b></em>: This folder provides a reduced and self-contained subset of the original data designed to facilitate replication and reproducibility of the main results. Given the large scale of the full datasets, which may pose computational and storage constraints, this sample offers a practical starting point for researchers and reviewers to validate the methodology, run the model, and reproduce key findings in a controlled setting.
<em><b>data</b></em>: Folder that stores voting information.
<em><b>Library</b></em>: Folder that contains the libraries for Preferences, Bot Prediction, and Space.

## Reproducibility

To facilitate reproducibility, we provide a reduced dataset in the `replication_sample` folder.

## Data Availability

The full datasets used in this study are large-scale and may be subject storage constraints. 

To ensure transparency and reproducibility, we provide a `replication_sample`, which allows validation of the methodology and reproduction of key results.

Instructions for accessing or reconstructing the full dataset are described in the accompanying paper.
