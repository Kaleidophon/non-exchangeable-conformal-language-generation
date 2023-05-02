"""
Define the core functions for conformal risk control in NLG.
"""

# STD
from collections import namedtuple
from typing import Tuple

# EXT
import numpy as np
import torch

# PROJECT
from src.datastore import DataStore
from src.custom_types import Device

# TYPES
ConformalResult = namedtuple("ConformalResult", ["q_hat", "n_eff"])


def simple_conformity_scores(predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
    """
    Compute conformity scores based on the predictions and targets.

    Parameters
    ----------
    predictions: torch.FloatTensor
        Predictions of the model.
    targets: torch.LongTensor
        Targets of the model.

    Returns
    -------
    torch.FloatTensor
        Conformity scores for each prediction-target pair.
    """
    # Compute conformity scores
    target_probs = torch.gather(predictions, 1, targets.unsqueeze(1))
    conformity_scores = 1 - target_probs

    return conformity_scores


def adaptive_conformity_score(predictions: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
    """
    Compute adaptive conformity score based on the predictions and targets.
    In comparison to the simple conformity score, the adaptive conformity score uses the cumulative sum or probabilities
    until the target token is reached.
    """
    sorted_classes, index = torch.sort(-predictions)
    sorted_probs = predictions[sorted_classes]
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    unsorted_cum_probs = torch.gather(cum_probs, -1, index.argsort(-1))

    conformity_scores = torch.gather(unsorted_cum_probs, -1, targets.unsqueeze(1))

    return conformity_scores


class ConformalCalibrator:
    """
    Class to create calibrated prediction sets using conformal prediction. This base version is based on
    non-exchangeable conformal prediction, were some weights are taken into account when computing the quantile.
    """

    def __init__(self, data_store: DataStore, alpha: float, temperature: float = 1.0, device: Device = "cpu", **kwargs):
        """
        Initialize a conformal calibrator.

        Parameters
        ----------
        data_store: DataStore
            DataStore containing the data to be used for the calibration.
        alpha: float
            Pre-specified confidence level.
        temperature: float
            Temperature to be used when computing the weights. Large temperatures will lead to more uniform weights.
            Default is 1.0.
        device: Device
            Device the model and datastore live on. Default is "cpu".
        """
        self.data_store = data_store
        self.alpha = alpha
        self.temperature = temperature
        self.device = device

        self.prediction_set_methods = {
            "classic": self.compute_classic_prediction_set,
            "adaptive": self.compute_adaptive_prediction_set
        }

    def compute_weights(self, distances: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute weights for each retrieved neighbor based on the distance.
        This done inspired by the way kNN-LMs work, using the exponential of the negative distance, times a temperature.

        Parameters
        ----------
        distances: torch.FloatTensor
            Distances for each retrieved neighbor.

        Returns
        -------
        torch.FloatTensor
            (Unnormalized) weights for each retrieved neighbor based on the neighbor's distance.
        """
        weights = np.exp(-self.temperature * distances)

        return weights

    def compute_q_hat(
        self,
        weights: torch.FloatTensor,
        conformity_scores: torch.FloatTensor
    ) -> Tuple[float, float]:
        """
        Compute quantile based on the computed weights and conformity scores.

        Parameters
        ----------
        weights: torch.FloatTensor
            Weights for each retrieved neighbor based on the distance.
        conformity_scores: torch.FloatTensor
            Conformity scores for each retrieved neighbor.

        Returns
        -------
        Tuple[float, float]
            NamedTuple of (q_hat, n_eff) where q_hat is the computed quantile and n_eff is the effective sample size.
        """
        weights = weights.cpu().numpy()

        # Normalize weights
        normed_weights = weights / (np.sum(weights) + 1)

        # Sort by conformal score ascending since we're trying to find the smallest q
        sorted_weights_and_scores = sorted(
            zip(normed_weights, conformity_scores), key=lambda tpl: tpl[1]
        )

        # Find the smallest q (compared to conformal scores)
        # for which the sum of corresponding weights is bigger equal than 1 - alpha
        q_hat = np.inf
        cumsum = 0

        for weight, score in sorted_weights_and_scores:
            if cumsum >= 1 - self.alpha:
                q_hat = score
                break

            cumsum += weight

        n_eff = np.sum(weights) / np.sum(weights ** 2)

        return ConformalResult(q_hat=q_hat, n_eff=n_eff)

    @staticmethod
    def compute_classic_prediction_set(predictions: torch.FloatTensor, q_hat: float) -> Tuple[torch.FloatTensor, int]:
        """
        Compute the classic prediction set based on the predictions and the quantile.

        Parameters
        ----------
        predictions: torch.FloatTensor
            Predictions for each retrieved neighbor.
        q_hat: float
            Quantile to be used for the prediction set.

        Returns
        -------
        Tuple[torch.FloatTensor, int]
            Prediction set (in the form of a zeroed-out and re-normalized output distribution) and its size.
        """
        set_size = torch.sum((predictions > q_hat).long()).numpy()
        predictions[predictions > q_hat] = 0
        predictions /= predictions.sum()

        return predictions, set_size

    @staticmethod
    def compute_adaptive_prediction_set(predictions: torch.FloatTensor, q_hat: float) -> Tuple[torch.FloatTensor, int]:
        """
        Compute the adaptive prediction set based on the predictions and the quantile.

        Parameters
        ----------
        predictions: torch.FloatTensor
            Predictions for each retrieved neighbor.
        q_hat: float
            Quantile to be used for the prediction set.

        Returns
        -------
        Tuple[torch.FloatTensor, int]
            Prediction set (in the form of a zeroed-out and re-normalized output distribution) and its size.
        """
        # TODO: Is there a way to make this more efficient?
        # This has log K complexity for sorting, followed by three passes through K classes (one for the cumsum, one
        # to create the prediction set and one to zero out the classes that are not in the prediction set). I would
        # think that it would be more efficient to re-order the cumulative probs back into the original order of the
        # classes instead of creating an intermediate set object.
        sorted_classes, index = torch.sort(-predictions)
        sorted_probs = predictions[sorted_classes]
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # TODO: Below is the old function logic. Delete when verified that more efficient version works
        #pred_set = set(sorted_classes[cum_probs < q_hat])

        #if len(pred_set) < len(predictions):
        #    pred_set.add(sorted_classes[len(pred_set)])

        #predictions[torch.range(0, len(predictions) - 1).long() not in pred_set] = 0
        #predictions /= predictions.sum()

        # Adapted from https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order
        # -after-torch-sort
        set_size = torch.sum((cum_probs < q_hat).long()).numpy()
        unsorted_cum_probs = torch.gather(cum_probs, -1, index.argsort(-1))
        predictions[unsorted_cum_probs < q_hat] = 0
        predictions /= predictions.sum()

        return predictions, set_size
