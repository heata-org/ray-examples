#!/usr/bin/env python

# Copyright 2023 Ray Authors
# Copyright 2025 heata <tech@heata.co> - Tweaked dataset and made configurable

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tweaked from https://github.com/ray-project/ray/blob/3cff3e55e5aaa87cca0be5f7be720fc926deede7/doc/source/tune/examples/tune-xgboost.ipynb


import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ray import tune
import ray

def train_breast_cancer(config):
    # Load dataset
    data, labels = sklearn.datasets.make_classification(
        n_samples=500 * DATASET_MULTIPLIER,
        n_features=200,
        n_informative=50,
        random_state=0
    )
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier
    results = {}
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        evals_result=results,
        verbose_eval=False,
    )
    # Return prediction accuracy
    accuracy = 1.0 - results["eval"]["error"][-1]
    tune.report({"mean_accuracy": accuracy, "done": True})


config = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
    "max_depth": tune.randint(1, 9),
    "min_child_weight": tune.choice([1, 2, 3]),
    "subsample": tune.uniform(0.5, 1.0),
    "eta": tune.loguniform(1e-4, 1e-1),
}

ray.init()


# Alter this to tweak how many samples to measure, this will change the number of Ray actors created by this job
NUM_SAMPLES = 30
# Altering this tweaks how intensive the work each Ray actor performs is, this is for simulating a more/less intensive model.
DATASET_MULTIPLIER = 500

print("Optimizing hyperparameters across heata")
print(f"NUM_SAMPLES  = {NUM_SAMPLES}   ;    DATASET_MULTIPLIER = {DATASET_MULTIPLIER}")



tuner = tune.Tuner(
    tune.with_resources(train_breast_cancer, {"memory": 2.5e9 }), # From empirical testing with the default settings, a sample takes around 2GB.
    tune_config=tune.TuneConfig(num_samples=NUM_SAMPLES),
    param_space=config,
)
results = tuner.fit()
