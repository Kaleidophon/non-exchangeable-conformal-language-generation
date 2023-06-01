"""
Define the core functions for conformal risk control in NLG.
"""

# STD
from collections import namedtuple
from functools import wraps
import types
from typing import Tuple

# EXT
from transformers import PreTrainedModel
from transformers.generation import LogitsProcessor
import torch
import torch.nn.functional as F

# PROJECT
from src.custom_types import Device

# TYPES
ConformalResult = namedtuple("ConformalResult", ["q_hat", "n_eff", "normed_weights"])


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
    sorted_classes, index = torch.sort(-predictions, dim=-1)
    sorted_probs = torch.gather(predictions, -1, index)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    unsorted_cum_probs = torch.gather(cum_probs, -1, index.argsort(-1))
    conformity_scores = torch.gather(unsorted_cum_probs, -1, targets.unsqueeze(1))

    return conformity_scores


class ConformalCalibrator:
    """
    Class to create calibrated prediction sets using conformal prediction. This base version is based on
    non-exchangeable conformal prediction, were some weights are taken into account when computing the quantile.
    """

    def __init__(
        self, alpha: float, distance_type: str = "inner_product",
            temperature: float = 1.0, device: Device = "cpu", **kwargs
    ):
        """
        Initialize a conformal calibrator.

        Parameters
        ----------
        data_store: DataStore
            DataStore containing the data to be used for the calibration.
        distance_type: str
            Type of distance measure being used. Either has to be "inner_product" or "l2".
        temperature: float
            Temperature to be used when computing the weights. Large temperatures will lead to more uniform weights.
            Default is 1.0.
        device: Device
            Device the model and datastore live on. Default is "cpu".
        """
        assert distance_type in ("inner_product", "l2", "cosine"), \
            "Distance type has to be either 'inner_product', 'cosine' or 'l2'."

        self.alpha = alpha
        self.temperature = temperature
        self.device = device
        self.distance_type = distance_type

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
        # For the inner product, larger values mean more similar vectors
        if self.distance_type in ("inner_product", "cosine"):
            weights = torch.exp(distances / self.temperature)

        # Use the same idea, but here we penalize larger distances
        elif self.distance_type == "l2":
            weights = torch.exp(-distances / self.temperature)

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
        # Normalize weights
        normed_weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1)

        # Sort by conformal score ascending since we're trying to find the smallest q
        sorted_conformity_scores, index = torch.sort(conformity_scores)
        sorted_weights = torch.gather(normed_weights, -1, index)

        # Find the smallest q (compared to conformal scores)
        # for which the sum of corresponding weights is bigger equal than 1 - alpha
        cumsum = torch.cumsum(sorted_weights, dim=-1)
        mask = (cumsum >= (1 - self.alpha)).long().to(self.device)
        threshold_index = torch.argmax(mask, dim=-1)
        q_hat = torch.gather(sorted_conformity_scores, -1, threshold_index.unsqueeze(-1)).squeeze(-1)

        # Make q_hat infinity in cases where the threshold was never reached
        infinity_mask = (mask.sum(dim=-1) == 0).to(self.device)
        q_hat[infinity_mask] = torch.inf

        n_eff = torch.sum(weights, dim=-1) / torch.sum(weights ** 2, dim=-1)

        return {
            "q_hat": q_hat,
            "n_eff": n_eff,
            "normed_weights": normed_weights
        }

    def get_prediction_sets(
        self,
        method: str,
        predictions: torch.FloatTensor,
        q_hat: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute the prediction set based on the predictions and the quantile.
        """
        return self.prediction_set_methods[method](predictions.to(self.device), q_hat.to(self.device))

    @staticmethod
    def compute_classic_prediction_set(predictions: torch.FloatTensor, q_hat: torch.FloatTensor) -> Tuple[torch.FloatTensor, int]:
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
        q_hat = q_hat.unsqueeze(-1).repeat(1, predictions.shape[-1])
        set_sizes = torch.sum((predictions > q_hat).cpu().long(), -1).numpy()

        predictions[predictions > q_hat] = 0
        predictions /= predictions.sum(-1, keepdim=True)

        return predictions, set_sizes

    @staticmethod
    def compute_adaptive_prediction_set(predictions: torch.FloatTensor, q_hat: torch.FloatTensor) -> Tuple[torch.FloatTensor, int]:
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
        sorted_classes, index = torch.sort(-predictions)  # Sort descendingly
        sorted_probs = torch.gather(predictions, -1, index)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        q_hat = q_hat.unsqueeze(-1).repeat(1, predictions.shape[-1])

        # Because adaptive prediction sets are supposed to include one more class than the ones that would come out of
        # the comparisons between the cumulative probabilities and q_hat (to avoid empty predictions sets), we adjust
        # q_hat by adding the difference to the next cumulative probability that would otherwise not be included.
        diffs = cum_probs - q_hat  # The value we are looking for here will be the smallest positive value per row
        diffs[diffs <= 0] = 1  # Second: Set all 0s to 1 to make sure that we don't pick them
        offset_values = torch.min(diffs, dim=-1, keepdim=True)[0]  # Third: Find the smallest positive value per row
        offset_values += 5e-3  # Add small value so that we include the minimum value as well
        # (The value has a strange value to accommodate errors caused floating point precision)
        q_hat[torch.isinf(q_hat)] = 0  # Remove infinity values, these will be replaced by > 1, so functionally the same
        q_hat += offset_values

        # Compute set sizes
        set_sizes = torch.sum((cum_probs < q_hat).long(), -1).cpu().numpy()

        # Compute actual prediction sets
        # Adapted from https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order
        # -after-torch-sort
        unsorted_cum_probs = torch.gather(cum_probs, -1, index.argsort(-1))
        predictions[unsorted_cum_probs >= q_hat] = 0
        predictions /= predictions.sum(-1, keepdim=True)

        return predictions, set_sizes


class ConformalLogitProcessor(LogitsProcessor):
    """
    Warper that uses the conformal prediction framework to compute prediction sets.
    """
    def __init__(self, q_hat: torch.FloatTensor, conformity_score: str, calibrator: ConformalCalibrator):
        super().__init__()
        self.q_hat = q_hat
        self.conformity_score = conformity_score
        self.calibrator = calibrator

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]

        # Avoid any modifications to the scores by the ForcedBOSTokenLogitsProcessor
        if cur_len > 1:
            scores = F.softmax(scores, dim=-1)
            q_hat = self.q_hat.repeat(scores.shape[0])  # Adapt to batch size
            scores = self.calibrator.get_prediction_sets(self.conformity_score, scores, q_hat)[0]

            # Put back into log space
            scores = torch.log(scores + 1e-12)

        return scores


class NonExchangeableConformalLogitProcessor(LogitsProcessor):
    """
    Warper that uses the non-exchangeable conformal prediction framework to compute prediction sets.
    """
    def __init__(
        self,
        conformity_score: str,
        distance_type: str,
        num_neighbors: int,
        data_store,
        calibrator: ConformalCalibrator
    ):
        super().__init__()
        self.distance_type = distance_type
        self.num_neighbors = num_neighbors
        self.conformity_score = conformity_score
        self.data_store = data_store
        self.calibrator = calibrator

        self.last_decoder_encodings = None

    def patch_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Patch the model forward pass to save the decoder hidden encodings.
        """
        def hook_fn(model, inputs, outputs):
            self.last_decoder_encodings = outputs.decoder_hidden_states[-1].squeeze(1)

            if self.distance_type == "inner_product":
                self.last_decoder_encodings /= model.config.d_model ** 0.25

            elif self.distance_type == "cosine":
                self.last_decoder_encodings = F.normalize(self.last_decoder_encodings, p=2, dim=-1)

        model.register_forward_hook(hook_fn)

        return model

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]

        # Avoid any modifications to the scores by the ForcedBOSTokenLogitsProcessor
        if cur_len > 1:
            distances = []
            conformity_scores = []

            for i in range(0, self.last_decoder_encodings.shape[0]):

                batch_distances, batch_conformity_scores = self.data_store.search_k(
                    self.last_decoder_encodings[i, :].unsqueeze(0), k=self.num_neighbors
                )
                distances.append(batch_distances)
                conformity_scores.append(batch_conformity_scores)

            distances = torch.cat(distances, dim=0)
            conformity_scores = torch.cat(conformity_scores, dim=0).squeeze(-1)

            weights = self.calibrator.compute_weights(distances)
            conformal_results = self.calibrator.compute_q_hat(
                weights, conformity_scores
            )
            q_hat = conformal_results["q_hat"]
            scores = F.softmax(scores, dim=-1)
            scores, set_size = self.calibrator.get_prediction_sets(self.conformity_score, scores, q_hat)

            # Put back into log space
            scores = torch.log(scores + 1e-12)

        return scores
