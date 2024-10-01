# HM Retrieval Two-Tower

This repository contains a Python/Tensorflow implementation of a two-tower neural network trained on [H&M transaction data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data). Practically, this model could be used for the candidate generation step in a recommendation system.

The provided repository can easily be refactored for use on other datasets, and for use in production environments in industry.

## Setup

This repository has been tested on an M1 Mac, but should work on Linux systems too. Windows support is limited.

### Prerequisites

- Python 3.10
- [Poetry](https://python-poetry.org/) - For managing Python dependencies and virtual environment management

Additional dependencies may need to be installed using Homebrew (Mac) or Apt (Linux).

### Installation Steps

1. Install the Python dependencies into a virtual environment

```bash
poetry install
```

2. Download the data from [here](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and store it in a directory within the repo.
1. In the [entrypoint](./main.py), change the settings based on where the data is store, where you want to store transformed data, and other settings.
1. In the [entrypoint](./main.py), change the Schema based on what input features you want to use and what model configuration.
1. Run the entrypoint

```
poetry run python main.py
```

## Repository Structure

### `pkg`

This directory contains all of the core source code for running the entrypoint. The entrypoint runs a series of runner functions, which are defined in a `runner.py` in each directory. These runner functions can easily be refactored into pipeline steps using an orchestration framework of choice when running this code in production.

The core source code contains the following modules:

- `etl` - Contains code for transforming raw data. This uses Pandas, but in production could contain queries for data warehouse transformations, or code for Spark.
- `modelling` - Contains code for custom layers, models and evaluation metrics. In addition, contains code to create indices. An index returns an ordered set of candidates. For the two-tower model, the index contains the query embedding model, static embeddings for all the candidates, and operations for returning the best candidates for a query. Also contains code for using TFRecord files for modelling.
- `schema` - In order to make the code reproducible across use cases, all model specific information is stored in a custom Schema object. This contains the input features used, model configuration, as well as training options.
- `tfrecord_writer` - To make the codebase scalable to large volumes of data, training / testing data is stored on disk, and loaded lazily as it is used. This means all the data does not need to be loaded into memory at once. Code in this directory is responsible for writing raw data to [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files.
- `utils` - Contains simple utils for logging and a custom settings object. The custom settings object mimics the behaviour of production pipeline parameters.

### `tests`

Due to this being a project focused on learning, the test coverage is very light. There is currently only tests for:

- The ETL transformations
- The Custom Recall@K calculation
