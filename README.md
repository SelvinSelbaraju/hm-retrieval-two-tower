# H&M Retrieval: Two-Tower Model

- [H\&M Retrieval: Two-Tower Model](#hm-retrieval-two-tower-model)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
  - [Repository Structure](#repository-structure)
    - [`pkg`](#pkg)
    - [`tests`](#tests)
  - [Model Overview](#model-overview)
    - [Training](#training)
    - [Inference](#inference)

## Introduction

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
1. In the [entrypoint](./main.py), change the settings based on where the data is stored, where you want to store transformed data, and other settings.
1. In the [entrypoint](./main.py), change the Schema based on what input features you want to use and what model configuration.
1. We are using legacy Keras optimizers as they are faster. So an environment variable needs to be set.

```bash
export TF_USE_LEGACY_KERAS=True
```

6. Run the entrypoint

```
poetry run python main.py
```

## Repository Structure

### `pkg`

This directory contains all of the core source code for running the entrypoint. The entrypoint runs a series of runner functions, which are defined in a `runner.py` in each directory. These runner functions can easily be refactored into pipeline steps using an orchestration framework of choice when running this code in production.

The core source code contains the following modules:

- `etl` - Contains code for transforming raw data. This uses Pandas, but in production could contain SQL queries for data warehouse (eg. Snowflake) transformations, or code for Spark.
- `modelling` - Contains code for custom layers, models and evaluation metrics. In addition, contains code to create indices. An index returns an ordered set of candidates. For the two-tower model, the index contains the query embedding model, static embeddings for all the candidates, and operations for returning the best candidates for a query. Also contains code for using TFRecord files for modelling.
- `schema` - In order to make the code reproducible across use cases, all model specific information is stored in a custom Schema object. This contains the input features used, model configuration, as well as training options.
- `tfrecord_writer` - To make the codebase scalable to large volumes of data, training / testing data is stored on disk, and loaded lazily as it is used. This means all the data does not need to be loaded into memory at once. Code in this directory is responsible for writing raw data to [TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord) files.
- `utils` - Contains a custom settings object. The custom settings object mimics the behaviour of production pipeline parameters.

### `tests`

Due to this being a project focused on learning, the test coverage is very light. There are currently only tests for:

- The ETL transformations
- The Custom Recall@K calculation

## Model Overview

The Two-Tower model is often used for the candidate generation step in recommender systems. Candidate generation is the process of taking all items that could be recommended, and refining it down to a smaller subset of hunderds/low-thousands. Because it operates on the full corpus of items, it has to be relatively lightweight.

The smaller subset returned from candidate generation is then passed to a different model, in the ranking stage. The ranking model is usually more complex, as it adds static features and has more trainable weights.

![Two Tower Model](/docs/images/two_tower.png)
Source: Nvidia Merlin Docs

### Training

In the Two-Tower Model architecture, an embedding of queries (users) and candidates (items) are computed independently, and then they are dot product to calculate an affinity score. Computing these representations separately (and potentially at different points in time) is a key reason why the Two-Tower model can be used at large scale for candidate generation.

Each tower of the model does not need to be identical, the only condition is that the final output of each tower must have the same dimensions. This late-crossing constraint is a key reason why Two-Tower models may be used less often for ranking. Raw query and candidate features are not allowed to interact, only transformed versions at the end via a dot product. More neural network layers could replace the dot product, but this could come at latency costs.

Data containing only positives is often used to train Two-Tower models. This is because negatives may be hard to collect directly. In order to make the model still learn, in-batch negative sampling is used.

In-batch negative sampling uses other positives in the same batch as negatives for a query. Intuitively, this works because we want the model to learn that the positive is relevant, and almost everything else is irrelevant. It is important to ensure the data is shuffled, as otherwise we might mistakenly use positives as negatives.

In-batch negative sampling is an efficient way to generate negatives, as we don't have to randomly sample from the full data. In addition, a batch of query embeddings $(B \times E)$ matrix multiplied with a batch of candidate embeddings $(B \times E)$ matrix (transposed) returns a $(B \times B)$ matrix where the $ij^{th}$ element is the affinity score of query $i$ with candidate $j$.

To create a probability distribution amongst elements in the batch, scores are transformed using a row-wise Softmax. The Cross Entropy loss function is used, where we want to maximise the probability of the actual positive. This indirectly pushes down the probabilities of all the negatives due to the Softmax calculation.

One nuance with this approach is that popular items which will frequently appear as positives, will also be frequently used as negatives. The model would learn that these popular items are often irrelevant, which is not the case. To overcome this issue, a **logQ** correction is used, where Q is the probability of that item being included in a random batch. The log of the probability for the $j^{th}$ item is subtracted from the affinity score with each query. For a statistical derivation, see the Sampled Softmax section [here](https://www.tensorflow.org/extras/candidate_sampling.pdf). This logQ correction has been shown to drastically improve recall (eg. [here](https://medium.com/expedia-group-tech/candidate-generation-using-a-two-tower-approach-with-expedia-group-traveler-data-ca6a0dcab83e)).

### Inference

Once a Two-Tower model is trained, embeddings for candidates be generated in batch, and stored in a vector database. At inference time, a query is embedded in real-time, and approximate nearest neighbours are fetched from the vector database of candidates. Approximate nearest neighbours are used instead of a true, brute-force nearest neighbour search for efficiency reasons.

In this repository, the Brute Force Index calculates exact nearest neighbour matches, and is slower as a result. The Index is a Tensorflow model, that when saved in the Tensorflow SavedModel format, can be deployed for serving requests using Tensorflow Serving.
