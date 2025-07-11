"""
This file is obtained from evaluate.py from the original repo,
modified to handle consistency calibration.
"""

import random
import logging
from pathlib import Path
from typing import Iterable, Optional, Literal, Union, Tuple, List
import gc
from gluonts.dataset.arrow import ArrowFile
from multiprocessing import Pool
import multiprocessing

import datasets
import gluonts
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy
from torchmetrics.classification import MulticlassCalibrationError
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

from chronos import (
    BaseChronosPipeline,
    ChronosBoltPipeline,
    ChronosPipeline,
    ForecastType,
)

app = typer.Typer(pretty_exceptions_enable=False)

# Taken from pandas._libs.tslibs.dtypes.OFFSET_TO_PERIOD_FREQSTR
offset_alias_to_period_alias = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q",
    "BQE-JAN": "Q",
    "BQE-FEB": "Q",
    "BQE-MAR": "Q",
    "BQE-APR": "Q",
    "BQE-MAY": "Q",
    "BQE-JUN": "Q",
    "BQE-JUL": "Q",
    "BQE-AUG": "Q",
    "BQE-SEP": "Q",
    "BQE-OCT": "Q",
    "BQE-NOV": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "Y": "Y",
    "BYE": "Y",
    "BYE-DEC": "Y",
    "BYE-JAN": "Y",
    "BYE-FEB": "Y",
    "BYE-MAR": "Y",
    "BYE-APR": "Y",
    "BYE-MAY": "Y",
    "BYE-JUN": "Y",
    "BYE-JUL": "Y",
    "BYE-AUG": "Y",
    "BYE-SEP": "Y",
    "BYE-OCT": "Y",
    "BYE-NOV": "Y",
    "YS": "Y",
    "BYS": "Y",
    "QS-JAN": "Q",
    "QS-FEB": "Q",
    "QS-MAR": "Q",
    "QS-APR": "Q",
    "QS-MAY": "Q",
    "QS-JUN": "Q",
    "QS-JUL": "Q",
    "QS-AUG": "Q",
    "QS-SEP": "Q",
    "QS-OCT": "Q",
    "QS-NOV": "Q",
    "QS-DEC": "Q",
    "BQS-JAN": "Q",
    "BQS-FEB": "Q",
    "BQS-MAR": "Q",
    "BQS-APR": "Q",
    "BQS-MAY": "Q",
    "BQS-JUN": "Q",
    "BQS-JUL": "Q",
    "BQS-AUG": "Q",
    "BQS-SEP": "Q",
    "BQS-OCT": "Q",
    "BQS-NOV": "Q",
    "BQS-DEC": "Q",
    "YS-JAN": "Y",
    "YS-FEB": "Y",
    "YS-MAR": "Y",
    "YS-APR": "Y",
    "YS-MAY": "Y",
    "YS-JUN": "Y",
    "YS-JUL": "Y",
    "YS-AUG": "Y",
    "YS-SEP": "Y",
    "YS-OCT": "Y",
    "YS-NOV": "Y",
    "YS-DEC": "Y",
    "BYS-JAN": "Y",
    "BYS-FEB": "Y",
    "BYS-MAR": "Y",
    "BYS-APR": "Y",
    "BYS-MAY": "Y",
    "BYS-JUN": "Y",
    "BYS-JUL": "Y",
    "BYS-AUG": "Y",
    "BYS-SEP": "Y",
    "BYS-OCT": "Y",
    "BYS-NOV": "Y",
    "BYS-DEC": "Y",
    "Y-JAN": "Y-JAN",
    "Y-FEB": "Y-FEB",
    "Y-MAR": "Y-MAR",
    "Y-APR": "Y-APR",
    "Y-MAY": "Y-MAY",
    "Y-JUN": "Y-JUN",
    "Y-JUL": "Y-JUL",
    "Y-AUG": "Y-AUG",
    "Y-SEP": "Y-SEP",
    "Y-OCT": "Y-OCT",
    "Y-NOV": "Y-NOV",
    "Y-DEC": "Y-DEC",
    "Q-JAN": "Q-JAN",
    "Q-FEB": "Q-FEB",
    "Q-MAR": "Q-MAR",
    "Q-APR": "Q-APR",
    "Q-MAY": "Q-MAY",
    "Q-JUN": "Q-JUN",
    "Q-JUL": "Q-JUL",
    "Q-AUG": "Q-AUG",
    "Q-SEP": "Q-SEP",
    "Q-OCT": "Q-OCT",
    "Q-NOV": "Q-NOV",
    "Q-DEC": "Q-DEC",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "W-SUN": "W-SUN",
}


def compute_arguments(
    num_splits: int,
    logits_list: torch.Tensor,
    n_perturbations: int,
    std: float,
    compute_naive: bool,
):
    """
    Generate argument tuples for parallel calls to the `compute_probabilities` function.

    This function splits the input `logits_list` into smaller chunks to be processed in parallel,
    optimizing memory usage and ensuring balanced workload distribution. Each chunk is paired
    with the necessary parameters to perform perturbed sampling of probabilities.

    :param num_splits: Number of parallel workers/processes to use.
    :type num_splits: int

    :param logits_list: torch Tensor of shape (num_samples, prediction_length, vocab_size) containing the logits to be processed.
    :type logits_list: torch.Tensor

    :param n_perturbations: Number of perturbations to apply per sample.
    :type n_perturbations: int

    :param std: Standard deviation used for generating Gaussian perturbations.
    :type std: float

    :param compute_naive: Whether to use the naive version of the `compute_probabilities` function.
    :type compute_naive: bool

    :return: A list of argument tuples, where each tuple contains:
             (logits_chunk, n_perturbations, std, chunk_id, compute_naive, process_seed),
             suitable for parallel execution of `compute_probabilities`.
    :rtype: List[Tuple[torch.Tensor, int, float, int, bool, int]]
    """
    # Initialize variables to compute arguments
    size = logits_list.shape[0]
    arguments = []
    chunk_size = max(1, size // num_splits)  # *2?

    for i in range(0, size, chunk_size):

        # Create a seed for each process, based on the external random seed
        # This ensures replicability without forcing the same seed (and samples)
        # In all the datasets
        process_seed = random.randint(0, 1e9)

        end = min(i + chunk_size, size)  # final length of the chunk
        chunk_id = 1 + i  # // chunk_size # compute chunk id for logging

        # add the arguments of the process to the list of arguments
        arguments.append(
            (
                logits_list[i:end],
                n_perturbations,
                std,
                chunk_id,
                compute_naive,
                process_seed,
            )
        )

    return arguments


def compute_metrics_path(
    mode: Literal["naive", "consistency"], model: str, std: float, n_pert: int
):
    """
    Generate the file path for saving metrics based on the evaluation mode and parameters.

    Constructs a standardized filename using the model name, standard deviation, number
    of perturbations, and evaluation mode (either "naive" or "consistency"). For instance, the filename
    is formatted "Results/chronos-t5-small_std_2_00_npert_128_consistency.csv in case of
    parameters mode:"consistency", model: "amazon/cronos-t5-small", std: 2.00, n_pert: 128.

    :param mode: The evaluation mode. Must be either "naive" or "consistency".
    :type mode: Literal["naive", "consistency"]

    :param model: The model identifier or path. Only the final part (e.g., model name) is used.
    :type model: str

    :param std: Standard deviation used in the evaluation (for perturbations).
    :type std: float

    :param n_pert: Number of perturbations applied during evaluation.
    :type n_pert: int

    :return: The constructed path string where metrics will be saved (as a `.csv` file).
    :rtype: str
    """
    # Ensure the mode is recognized
    assert mode in [
        "naive",
        "consistency",
    ], "The metrics paths mode should be 'naive' or 'consistency'"

    # Obtain the final part of the model's name
    model_name = model.split("/")[-1]

    # Compute the name
    results_path = Path(__file__).parent/ "Results"
    if mode == "consistency":
        file_name = (
            f"{model_name}_std_{std:.2f}_npert_{n_pert}_{mode}".replace(".", "_")
            + ".csv"
        )
        return results_path / file_name
    else:
        return results_path / f"{model_name}_naive.csv"


def softmax(x: Union[np.ndarray, torch.Tensor]):
    """
    Compute the softmax of a 1D tensor or array.

    This function normalizes the input vector into a probability distribution,
    where the output values sum to 1. Supports both NumPy arrays and PyTorch tensors on CPU.

    :param x: Input vector of raw scores (logits), as a NumPy array or PyTorch tensor.
    :type x: Union[np.ndarray, torch.Tensor]

    :return: Softmax-normalized probability distribution.
    :rtype: Same as input (np.ndarray or torch.Tensor)
    """
    return np.exp(x) / sum(np.exp(x))


def compute_probabilities(
    logits_list,
    n_perturbations=10,
    std=0.1,
    instance=1,
    compute_naive=True,
    process_seed=None,
):
    """
    Compute naive softmax probabilities and consistency-based probabilities over perturbed logits.

    This function processes a list of sequences of logits, computing two types of probabilities:
    - Naive probabilities: using the softmax of each unperturbed logit vector.
    - Consistency probabilities: using repeated perturbations and argmax voting.

    To reduce memory pressure, the function processes the input in batches and explicitly frees
    memory when intermediate results are no longer needed.

    :param logits_list: A list of sequences, where each sequence is a list or tensor of logit vectors
                        (shape: (n_samples, prediction_length, vocab_size)).
    :type logits_list: torch.Tensor

    :param n_perturbations: Number of perturbations applied to each logit vector for consistency estimation.
    :type n_perturbations: int

    :param std: Standard deviation of the Gaussian noise added during perturbation.
    :type std: float

    :param instance: Process index (used for display/logging purposes in parallel settings).
    :type instance: int

    :param compute_naive: Whether to compute naive softmax probabilities.
    :type compute_naive: bool

    :param process_seed: Optional random seed for reproducibility.
    :type process_seed: Optional[int]

    :return: Tuple of two arrays:
             - naive_probs: Array of shape (N, T, V) if compute_naive is True; empty list otherwise.
             - consistency_probs: Array of shape (N, T, V) with normalized consistency-based probabilities.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Setup initial parameters
    naive_probs = []
    consistency_probs = []
    batch_size = 250  # Process in smaller batches with better memory management

    # Compute process seed
    if process_seed:
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)

    # Output the current process
    print(" ", end="", flush=True)
    desc = f"Parallel process {instance}"

    for i in range(0, len(logits_list), batch_size):
        batch = logits_list[i : i + batch_size]
        batch_naive = []
        batch_consistency = []

        for idx, logits in enumerate(batch):
            if idx % 25 == 0:  # Less frequent progress updates
                print(f"\r{desc}: {i+idx}/{len(logits_list)}", end="", flush=True)

            # Process one sample at a time
            naive_probs_sample = []
            consistency_probs_sample = []

            for logit in logits:
                # Calculate softmax once and reuse
                if compute_naive:
                    naive_prob = softmax(logit)
                    naive_probs_sample.append(naive_prob)

                # Use more memory-efficient approach for consistency
                consistency = np.zeros_like(
                    logit, dtype=np.float32
                )  # Specify dtype for memory efficiency

                for _ in range(int(n_perturbations)):
                    # Generate perturbation directly into pre-allocated array
                    perturb = np.random.normal(0, std, size=logit.shape)
                    logit_perturb = logit + perturb
                    max_index = np.argmax(logit_perturb)
                    consistency[max_index] += 1

                    # Clean up temporary arrays
                    del perturb
                    del logit_perturb

                # Normalize in-place
                consistency /= n_perturbations
                consistency_probs_sample.append(consistency)
            if compute_naive:
                batch_naive.append(naive_probs_sample)
            batch_consistency.append(consistency_probs_sample)

            # Clear sample variables explicitly
            del naive_probs_sample
            del consistency_probs_sample

        # Convert batch results to arrays and extend results
        if compute_naive:
            naive_probs.extend(batch_naive)
        consistency_probs.extend(batch_consistency)

        # Explicit cleanup
        del batch
        del batch_naive
        del batch_consistency

        # Force garbage collection
        gc.collect()

    print(f"\r{desc}: Completed {len(logits_list)} samples", flush=True)

    # Convert to arrays at the end to minimize intermediate memory usage
    return np.array(naive_probs, dtype=np.float32), np.array(
        consistency_probs, dtype=np.float32
    )


def get_sample_tokens(p_total: np.ndarray, num_samples: int):
    """
    Sample token indices from a sequence of probability distributions.

    For each time step in each input sequence, this function draws `num_samples` token indices
    according to the categorical probability distribution provided. Sampling is done in batches
    to manage memory more efficiently.

    :param p_total: A 3D NumPy array of shape (N, T, V), where:
                    - N is the number of sequences
                    - T is the number of time steps
                    - V is the vocabulary size
                    Each [i, t, :] vector represents a probability distribution over V tokens.
    :type p_total: np.ndarray

    :param num_samples: The number of samples to draw per time step for each sequence.
    :type num_samples: int

    :return: A 3D NumPy array of shape (N, T, num_samples), where each entry contains sampled token indices.
    :rtype: np.ndarray
    """
    x = np.arange(p_total.shape[-1], dtype=np.int32)
    prediction_tokens = []

    # Process in batches
    batch_size = 500
    for i in range(0, len(p_total), batch_size):
        # compute batch
        batch = p_total[i : i + batch_size]
        batch_tokens = []

        for p_series in batch:
            series_tokens = []
            # sample each timestep accroding to the logits
            for p_step in p_series:
                series_tokens.append(np.random.choice(x, num_samples, p=p_step))
            batch_tokens.append(series_tokens)

        prediction_tokens.extend(batch_tokens)

        # Clean up
        del batch
        del batch_tokens
        gc.collect()

    return np.array(prediction_tokens, dtype=np.int32)


def plot_time_series(
    data_naive,
    data_consistency,
    dataset_name,
    quantile=95,
    ground_truth=None,
    max_preceding=256,
):
    """
    Plot forecasted time series with confidence intervals.

    For each time series, this function plots the mean prediction and quantile range
    from both naive and consistency-based probabilities. Optionally includes ground truth
    and preceding context.

    :param data_naive: Naive model forecast samples. Shape: (N, T, S)
    :type data_naive: np.ndarray

    :param data_consistency: Consistency model forecast samples. Shape: (N, T, S)
    :type data_consistency: np.ndarray

    :param dataset_name: Name of the dataset (used in titles).
    :type dataset_name: str

    :param quantile: Confidence interval level (default: 95).
    :type quantile: int

    :param ground_truth: Optional object with ground truth and context data.
    :type ground_truth: object or None

    :param max_preceding: Max number of preceding steps to show (default: 256).
    :type max_preceding: int

    :return: None. Displays matplotlib plots.
    :rtype: None
    """
    num_series = data_naive.shape[0]
    prediction_length = data_naive.shape[1]
    timesteps = np.arange(prediction_length)

    for i in range(num_series):
        # for i in range(10):
        plt.figure(figsize=(10, 6))  # Create a new figure for each plot

        # Calculate statistics for naive and consistency data for the current series
        mean_series_naive = data_naive[i].mean(axis=1)
        mean_series_cons = data_consistency[i].mean(axis=1)

        # Compute upper and lower bounds for shaded areas both for naive and consistency probabilities
        lower_bound_naive = np.percentile(data_naive[i], (100 - quantile) / 2, axis=1)
        upper_bound_naive = np.percentile(
            data_naive[i], quantile + (100 - quantile) / 2, axis=1
        )

        lower_bound_cons = np.percentile(
            data_consistency[i], (100 - quantile) / 2, axis=1
        )
        upper_bound_cons = np.percentile(
            data_consistency[i], quantile + (100 - quantile) / 2, axis=1
        )

        # handle the plotting with ground truth and preceding timesteps
        if ground_truth is not None:
            # preceding timesteps
            if max_preceding > 0:
                preceding_series = ground_truth.input.test_data.dataset[i][
                    "target"
                ]  # Extract time series
                n = min(
                    len(preceding_series) - prediction_length, max_preceding
                )  # Determine the length dynamically
                preceding_timesteps = np.arange(
                    -n, 0
                )  # Negative indices for proper alignment

                plt.plot(
                    preceding_timesteps,
                    preceding_series[-n - prediction_length : -prediction_length],
                    label="Preceding Data",
                    color="green",
                    linestyle="dashed",
                    linewidth=2.5,
                )
            # ground truth
            real_ts = ground_truth.dataset[i]["target"][-prediction_length:]
            plt.plot(
                timesteps, real_ts, label="Ground Truth", color="black", linewidth=2.5
            )

        # Plot the mean time series for naive and consistency
        plt.plot(
            timesteps, mean_series_naive, label="Original", color="blue", linewidth=2.5
        )
        plt.plot(
            timesteps, mean_series_cons, label="Consistency", color="red", linewidth=2.5
        )

        # Plot the quantile ranges for naive and consistency
        plt.fill_between(
            timesteps,
            lower_bound_naive,
            upper_bound_naive,
            color="blue",
            alpha=0.3,
            label="Original range",
        )
        plt.fill_between(
            timesteps,
            lower_bound_cons,
            upper_bound_cons,
            color="red",
            alpha=0.3,
            label="Consistency range",
        )

        # Add title, axes labels and legend
        plt.title(f"Time Series {i+1} - Dataset {dataset_name}", fontsize=25)
        plt.xlabel("Timesteps", fontsize=23)
        plt.ylabel("Value", fontsize=23)
        plt.tick_params(axis='both', labelsize=20)
        plt.legend(loc="lower left", fontsize=20)
        plt.tight_layout()
        plt.show()


def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_freq = pd.infer_freq(hf_dataset[0]["timestamp"])
    dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamp"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    assert len(gts_dataset) == dataset_length

    return gts_dataset


def load_and_split_dataset(backtest_config: dict, max_series: int):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]

    # This is needed because the datasets in autogluon/chronos_datasets_extra cannot
    # be distribued due to license restrictions and must be generated on the fly
    trust_remote_code = hf_repo == "autogluon/chronos_datasets_extra"

    ds = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )
    ds.set_format("numpy")
    n_samples = ds.shape[0]

    # limit the number of time series that we sample to avoid memory issues
    if max_series is not None and n_samples > max_series:
        logger.info(
            f"Sampling only {max_series} at random from the dataset (original size: {n_samples})"
        )
        sample_indices = np.random.choice(
            n_samples, max_series, replace=False
        )  # get indices
        ds = ds.select(sample_indices)  # subsample the dataset
        ds.info.splits["train"].num_examples = (
            max_series  # update the information in the dataset
        )

    gts_dataset = to_gluonts_univariate(ds)

    # Split dataset for evaluation
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls)

    return test_data


def compute_data_for_cc(
    test_data_input: Iterable,
    pipeline: BaseChronosPipeline,
    prediction_length: int,
    batch_size: int,
    test_targets: gluonts.dataset.split.TestData,
    **predict_kwargs,
):
    """
    Generate logits and token labels for consistency calibration from a forecasting pipeline.

    Processes test time series in batches and runs the model in inference mode to collect:
    - Logits from the model's decoder
    - Ground truth token labels aligned with the prediction window
    - Scaling factors (if used for normalization)

    :param test_data_input: Iterable of input time series dicts with "target" fields.
    :type test_data_input: Iterable

    :param pipeline: A Chronos pipeline used for forecasting and tokenization.
    :type pipeline: BaseChronosPipeline

    :param prediction_length: Number of future steps to predict.
    :type prediction_length: int

    :param batch_size: Number of time series to process per batch.
    :type batch_size: int

    :param test_targets: gluonts.dataset.split.TestData containing ground truth targets aligned with `test_data_input`.
    :type test_targets: gluonts.dataset.split.TestData

    :param predict_kwargs: Additional keyword arguments passed to `pipeline.predict()`.
    :type predict_kwargs: dict

    :return: Tuple of:
             - correct_tokens (torch.Tensor): True label tokens for each prediction (shape: [N, T])
             - logits (torch.Tensor): Model output logits (shape: [N, T, V])
             - scales (torch.Tensor): Optional scale values used for normalization
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    pipeline.tokenizer.config.prediction_length = prediction_length
    correct_tokens = []
    logits = []
    test_targets = test_targets.dataset
    scales = []
    for batch in tqdm(
        batcher(zip(test_data_input, test_targets), batch_size=batch_size)
    ):
        context = [torch.tensor(entry[0]["target"]) for entry in batch]
        tgt = [torch.tensor(entry[1]["target"][-prediction_length:]) for entry in batch]
        _, original_logits, scale, predicted_tokens = pipeline.predict(
            context=context,
            prediction_length=prediction_length,
            return_logits=True,
            **predict_kwargs,
        )
        # ignore if len is one because it has problems with the shape and doesn't change much if only one ts is removed
        if len(context) == 1:
            continue
        tgt = torch.stack(tgt, dim=0)
        correct_batch_tokens, _ = pipeline.tokenizer.label_input_transform(tgt, scale)
        correct_tokens.append(correct_batch_tokens)
        logits.append(original_logits)
        scales.append(scale)

    # compute correct tokens, logits and scales (needed to convert labels to tokens)
    correct_tokens = [tok[:, :-1] for tok in correct_tokens]
    correct_tokens = torch.cat(correct_tokens)

    scales = torch.cat(scales)

    logits = torch.cat(logits, axis=1)
    logits = logits.swapaxes(0, 1)

    return correct_tokens.cpu(), logits.cpu(), scales.cpu()


def get_forecasts_cc(
    test_data_input: gluonts.dataset.split.InputDataset,
    predictions: Union[
        np.ndarray, torch.Tensor
    ],  # predictions should be of shape (num_series, pred_length, num_samples)
    pipeline: BaseChronosPipeline,
    scales: torch.Tensor,
) -> Tuple[torch.Tensor, List[gluonts.model.forecast.Forecast]]:
    """
    Convert model prediction samples into denormalized time series and GluonTS forecast objects.

    :param test_data_input: List of input series dicts with "target" and "start" fields.
    :type test_data_input: gluonts.dataset.split.InputDataset

    :param predictions: Tensor or ndarray of shape (N, T, S) with N series, T prediction steps, and S samples.
    :type predictions: Union[numpy.ndarray, torch.Tensor]

    :param pipeline: Chronos pipeline with tokenizer and forecast_type (SAMPLES or QUANTILES).
    :type pipeline: BaseChronosPipeline

    :param scales: Tensor of scaling factors for denormalization.
    :type scales: torch.Tensor

    :return:
        - predicted_ts (Tensor): Denormalized forecast samples of shape (N, S, T).
        - forecasts (List): List of GluonTS forecast objects (SampleForecast or QuantileForecast).
    :rtype: Tuple[torch.Tensor, List[gluonts.model.forecast.Forecast]]
    """
    forecasts = []
    predicted_ts = []

    # Convert numpy to torch
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)

    assert isinstance(
        predictions, torch.Tensor
    ), "The predictions array is neither a tensor nor a numpy ndarray"

    # Convert labels back to the time series numerical values
    for i, samples in enumerate(predictions):
        scale = scales[i]
        predicted_ts.append(
            pipeline.tokenizer.output_transform(samples.to(scale.device), scale)
        )

    predicted_ts = torch.stack(predicted_ts, dim=0)
    forecast_outputs = torch.swapaxes(predicted_ts, 1, 2).numpy()

    # Create SampleForecast and QuantileForecast objects used in evaluation
    for item, ts in zip(forecast_outputs, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        if pipeline.forecast_type == ForecastType.SAMPLES:
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )
        elif pipeline.forecast_type == ForecastType.QUANTILES:
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, pipeline.quantiles)),
                    start_date=forecast_start_date,
                )
            )
    return predicted_ts, forecasts


def compute_metrics(forecasts_naive, forecasts_cons, test_data, batch_size: int = 5000):
    """Compute evaluation for both naive and consistency predictions.

    Computes MASE and Mean Weighted Sum Quantile Loss (MWSQL) for both sets of forecasts,
    returning the results as lists of dictionaries.

    :param forecasts_naive: List of naive forecast objects (can be None or empty).
    :param forecasts_cons: List of consistency-calibrated forecast objects.
    :param test_data: Ground truth time series aligned with the forecasts.
    :param batch_size: Number of series to evaluate in parallel.

    :return:
        - metrics_naive: Evaluation results for the naive forecasts, or None if not provided.
        - metrics_cons: Evaluation results for the consistency-calibrated forecasts.
    :rtype: Tuple[Optional[List[Dict[str, float]]], List[Dict[str, float]]]
    """

    # If naive forecasts are passed as argument compute the metrics (otherwise the corresponding value will contain None)

    metrics_naive = None
    if forecasts_naive:
        metrics_naive = (
            evaluate_forecasts(
                forecasts_naive,
                test_data=test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1))],
                batch_size=batch_size,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

    metrics_cons = (
        evaluate_forecasts(
            forecasts_cons,
            test_data=test_data,
            metrics=[
                MASE(),
                MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
            ],
            batch_size=5000,
        )
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    return metrics_naive, metrics_cons


@app.command()
def main(
    config_path: Path,
    metrics_path_naive: Optional[Path] = None,
    metrics_path_consistency: Optional[Path] = None,
    chronos_model_id: str = "amazon/chronos-t5-small",
    std: float = 4,
    n_perturbations: int = 16,
    n_bins: int = 10,
    n_samples: int = 100,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    batch_size: int = 32,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    max_series: Optional[int] = 24_000,
    compute_naive: Optional[bool] = False,
    plot: Optional[bool] = False,
    seed: Optional[int] = 42,
):
    """Evaluate Chronos models.

    Parameters
    ----------
    config_path : Path
        Path to the evaluation config. See ./configs/.
    metrics_path : Path
        Path to the CSV file where metrics will be saved.
    chronos_model_id : str, optional, default = "amazon/chronos-t5-small"
        HuggingFace ID of the Chronos model or local path
        Available models on HuggingFace:
        Chronos:
            - amazon/chronos-t5-tiny
            - amazon/chronos-t5-mini
            - amazon/chronos-t5-small
            - amazon/chronos-t5-base
            - amazon/chronos-t5-large
        Chronos-Bolt:
            - amazon/chronos-bolt-tiny
            - amazon/chronos-bolt-mini
            - amazon/chronos-bolt-small
            - amazon/chronos-bolt-base
    device : str, optional, default = "cuda"
        Device on which inference will be performed
    torch_dtype : str, optional
        Model's dtype, by default "bfloat16"
    batch_size : int, optional, default = 32
        Batch size for inference. For Chronos-Bolt models, significantly larger
        batch sizes can be used
    num_samples : int, optional, default = 20
        Number of samples to draw when using the original Chronos models
    temperature : Optional[float], optional, default = 1.0
        Softmax temperature to used for the original Chronos models
    top_k : Optional[int], optional, default = 50
        Top-K sampling, by default None
    top_p : Optional[float], optional, default = 1.0
        Top-p sampling, by default None
    max_series : Optional[int], optional, default = 24_000
        Maximum number of series that get sampled within a dataset
    compute_naive : Optional[bool], optional, default = False
        Whether to compute the stats about the naive method, i.e.
        applying softmax to the logits
    plot : Optional[bool], optional, default = False
        Whether to perform the plotting. Should be done only
        when processing few time series
    seed : Optional[int], optional, default = 42
        Seed to use for reproducibility.
    """
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_cores = multiprocessing.cpu_count()
    # Load Chronos
    pipeline = BaseChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
    )

    # Compute metrics paths
    if metrics_path_naive is None:
        metrics_path_naive = compute_metrics_path(
            mode="naive", model=chronos_model_id, std=std, n_pert=n_perturbations
        )
    if metrics_path_consistency is None:
        metrics_path_consistency = compute_metrics_path(
            mode="consistency", model=chronos_model_id, std=std, n_pert=n_perturbations
        )
    assert isinstance(
        pipeline, ChronosPipeline
    ), "The model needs to be of type ChronosPipeline"

    predict_kwargs = dict(
        num_samples=1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows_naive = []
    result_rows_cons = []

    for config in backtest_configs:

        # --------------------------------------------------------
        # ------------------------ STEP 1 ------------------------
        # ------ Load the dataset and compute the forecasts ------
        # --------------------------------------------------------

        starting_time = datetime.now()  # start the timer

        # Gather data from the config (used for logging)
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        # Load the dataset
        logger.info(f"Loading {dataset_name}")
        test_data = load_and_split_dataset(
            backtest_config=config, max_series=max_series
        )

        # Print info and compute kwargs for the .generate() method
        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )

        # Compute ground_truth, logits and scales
        # Scales are used to convert tokens back to numerical values
        correct_tokens, logits, scales = compute_data_for_cc(
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            test_targets=test_data,
            **predict_kwargs,
        )

        # --------------------------------------------------------
        # ------------------------ STEP 2 ------------------------
        # ----- Compute ECE based on logits and ground truth -----
        # --------------------------------------------------------

        # Establish maximum number of parallel processes
        num_processes = min(len(test_data.input), num_cores)

        # Compute arguments for parallel processes
        arguments = compute_arguments(
            num_processes, logits, n_perturbations, std, compute_naive
        )

        # Process the consistency (and naive if needed) probabilities in parallel
        with Pool(num_processes) as pool:
            results = pool.starmap(compute_probabilities, arguments)
            pool.close()
            pool.join()

        # Obtain naive and consistency probabilities
        naive_probs, consistency_probs = zip(*results)

        # Delete results and handle garbage collection to save memory
        del results
        gc.collect()

        # Compute sampling and forecasting from naive probabilities if required
        if compute_naive:
            naive_probs = np.concatenate(naive_probs)
            naive_sample_tokens = get_sample_tokens(naive_probs, n_samples)
            predicted_ts_naive, forecasts_naive = get_forecasts_cc(
                test_data.input, naive_sample_tokens, pipeline, scales
            )

        # Compute sampling and forecasts from consistency probabilities
        consistency_probs = np.concatenate(consistency_probs)
        cons_sample_tokens = get_sample_tokens(consistency_probs, n_samples)
        predicted_ts_cons, forecasts_cons = get_forecasts_cc(
            test_data.input, cons_sample_tokens, pipeline, scales
        )

        # Perform plotting
        if plot and compute_naive:
            plot_time_series(naive_sample_tokens, cons_sample_tokens, dataset_name)
            plot_time_series(
                predicted_ts_naive,
                predicted_ts_cons,
                dataset_name,
                ground_truth=test_data,
                max_preceding=50,
            )  # Set length of initial timesteps of the ground truth

        # Reshape arrays for compatibility reasons with ECE library implementation
        consistency_probs = torch.from_numpy(consistency_probs).flatten(
            start_dim=0, end_dim=1
        )
        correct_tokens = correct_tokens.flatten(start_dim=0, end_dim=1)

        # Instantiate ECE object with the correct number of classes (4096) and bins
        ECE = MulticlassCalibrationError(
            num_classes=consistency_probs.shape[-1], n_bins=n_bins
        )

        # Compute naive ECE
        if compute_naive:
            # Reshape naive probabilities to ensure they are properly handled by ECE
            naive_probs = torch.from_numpy(naive_probs).flatten(start_dim=0, end_dim=1)
            ece_naive = ECE(naive_probs, correct_tokens)
            logger.info("Naive probs ECE: ", ece_naive)

        # Compute consistency ECE
        ece_consistency = ECE(consistency_probs, correct_tokens)
        logger.info("Consistency probs ECE: ", ece_consistency)

        # --------------------------------------------------------
        # ------------------------ STEP 3 ------------------------
        # --------- Compute metrics and save the results ---------
        # --------------------------------------------------------

        logger.info(f"Evaluating forecasts for {dataset_name}")

        if not compute_naive:
            forecasts_naive = None

        # Compute MASE and WQL for consistency and, if present, naive forecasts
        metrics_naive, metrics_cons = compute_metrics(
            forecasts_naive, forecasts_cons, test_data
        )

        # Output data
        if compute_naive:
            logger.info("Naive metrics: ", metrics_naive)
        logger.info("Cons metrics: ", metrics_cons)

        # Store the data about naive metrics
        if compute_naive:
            metrics_naive_dict = metrics_naive[0]
            metrics_naive_dict["ECE"] = ece_naive.item()

        # Store the data about consistency metrics
        metrics_cons_dict = metrics_cons[0]
        metrics_cons_dict["ECE"] = ece_consistency.item()

        # Compute elapsed time for processing the whole dataset and print it
        elapsed_time = datetime.now() - starting_time
        logger.info("Elapsed time: ", elapsed_time.total_seconds())
        logger.info(
            "Time/series: ", elapsed_time.total_seconds() / len(test_data.input)
        )

        # -------- Save partial results to CSV files --------

        # Compute the rows with the partial result to store it in the dataset later
        if compute_naive:
            result_rows_naive.append(
                {
                    "dataset": dataset_name,
                    "model": chronos_model_id,
                    **metrics_naive_dict,
                }
            )
        result_rows_cons.append(
            {"dataset": dataset_name, "model": chronos_model_id, **metrics_cons_dict}
        )

        # Create the folder "Results" if it's not there
        results_path = Path(__file__).parent/ "Results"

        os.makedirs(results_path, exist_ok=True)

        # Save naive probabilities evaluation (if computed)
        if compute_naive:
            results_df_naive = (
                pd.DataFrame(result_rows_naive)
                .rename(
                    {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
                    axis="columns",
                )
                .sort_values(by="dataset")
            )

            # Save the data about naive probabilities
            results_df_naive.to_csv(metrics_path_naive, index=False)

        # Save consistency probailities evaluation
        results_df_cons = (
            pd.DataFrame(result_rows_cons)
            .rename(
                {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
                axis="columns",
            )
            .sort_values(by="dataset")
        )

        # Save the data about consistency probabilities
        results_df_cons.to_csv(metrics_path_consistency, index=False)


if __name__ == "__main__":
    # Handle logger
    logger = logging.getLogger("Chronos Evaluation")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

    # Run the script
    app()
