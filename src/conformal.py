"""
Define the core functions for conformal risk control in NLG.
"""

# STD
from collections import namedtuple
from typing import Tuple, Dict, List

# EXT
import numpy as np
from transformers import PreTrainedModel, M2M100PreTrainedModel
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
    sorted_classes, index = torch.sort(-predictions, dim=-1)
    sorted_probs = torch.gather(predictions, -1, index)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    unsorted_cum_probs = torch.gather(cum_probs, -1, index.argsort(-1))
    conformity_scores = torch.gather(unsorted_cum_probs, -1, targets.unsqueeze(1))

    return conformity_scores


class ConformalCalibrator:
    """
    Class to create calibrated prediction sets using conformal prediction. This base version is based on
    non-exchangeable conformal prediction, where some weights are taken into account when computing the quantile.
    """

    def __init__(
        self,
        alpha: float,
        distance_type: str = "inner_product",
        temperature: float = 1.0,
        device: Device = "cpu",
        **kwargs
    ):
        """
        Initialize a conformal calibrator.

        Parameters
        ----------
        alpha: float
            Define desired coverage through 1 - alpha.
        distance_type: str
            Type of distance measure being used. Either has to be "inner_product", "l2" or "cosine".
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
    ) -> Dict[str, float]:
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
        Dict[str, float]
            Data corresponding the q_hat calculation in a dictionary, including the computed quantile (q_hat), effective
            sample size (n_eff) and the normalized weights.
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
    ) -> Tuple[torch.FloatTensor, List[int]]:
        """
        Compute the prediction set based on the predictions and the quantile.

        Parameters
        ----------
        method: str
            Name of the prediction set method. Must be either "classic" or "adaptive".
        predictions: torch.FloatTensor
            Original predictions by the model.
        q_hat: torch.FloatTensor
            Compute quantiles to create the prediction sets.

        Returns
        -------
        Tuple[torch.FloatTensor, torch.LongTensor]
            Tuple of predictive distributions (with excluded token probabilities set to 0) as well as a list of
            corresponding set sizes.
        """
        return self.prediction_set_methods[method](predictions.to(self.device), q_hat.to(self.device))

    @staticmethod
    def compute_classic_prediction_set(
        predictions: torch.FloatTensor,
        q_hat: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, List[int]]:
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
        Tuple[torch.FloatTensor, List[int]]
            Prediction set (in the form of a zeroed-out and re-normalized output distribution) and its size.
        """
        q_hat = q_hat.unsqueeze(-1).repeat(1, predictions.shape[-1])
        set_sizes = torch.sum((predictions > q_hat).cpu().long(), -1).numpy()

        predictions[predictions > q_hat] = 0
        predictions /= predictions.sum(-1, keepdim=True)

        return predictions, set_sizes

    @staticmethod
    def compute_adaptive_prediction_set(
        predictions: torch.FloatTensor,
        q_hat: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, List[int]]:
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
        Tuple[torch.FloatTensor, List[int]]
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
    LogitProcessor that uses the conformal prediction framework to compute prediction sets.
    This corresponds to the method used by Ravfogel et al. (2023).
    """
    def __init__(
        self,
        alpha: float,
        conformity_score: str,
        data_store,
        calibrator: ConformalCalibrator,
        num_bins: int = 10
    ):
        """
        Initialize the ConformalLogitProcessor.

        Parameters
        ----------
        alpha: float
            Define desired coverage through 1 - alpha.
        conformity_score: str
            Name of conformity score method being used. Should be "classic" or "adapted".
        data_store: DataStore
            Pre-populated datastore.
        calibrator: ConformalCalibrator
            Instance of conformal calibrator class.
        num_bins: int
            Number of entropy bins to compute bin-wise quantiles for. Set to 10 by default.
        """
        super().__init__()
        self.conformity_score = conformity_score
        self.calibrator = calibrator
        self.num_bins = num_bins

        # Bin conformity stores by entropy and get the corresponding q_hat
        conformity_scores = data_store.value_tensor.to(self.calibrator.device)
        entropy_values = data_store.value_tensor.to(self.calibrator.device)
        bin_size = int(entropy_values.shape[0] / self.num_bins)

        # Sort by entropy values
        entropy_indices = torch.argsort(entropy_values.squeeze(-1), dim=0)
        conformity_scores = conformity_scores[entropy_indices].squeeze(-1)

        bins = [
            conformity_scores[i * bin_size:(i+1) * bin_size].squeeze(-1) for i in range(self.num_bins)
        ]
        self.bin_boundaries = torch.FloatTensor(
            [torch.min(bin) for bin in bins]
        ).to(self.calibrator.device)

        # Compute q_hat for each bin
        self.q_hats = []

        for bin in bins:
            N = len(bin)
            q_level = np.clip(np.ceil((N + 1) * (1 - alpha)) / N, 0, 1)
            q_hat = torch.FloatTensor(
                [np.quantile(bin.cpu().numpy(), q_level, method='higher')]
            ).squeeze(-1).to(self.calibrator.device)
            self.q_hats.append(q_hat)

    def get_q_hats(self, predictions: torch.FloatTensor) -> torch.FloatTensor:
        """
        Get the q_hat corresponding to an entropy bin.

        Parameters
        ----------
        predictions: torch.FloatTensor
            Model predictions.

        Returns
        -------
        torch.FloatTensor
            q_hat values corresponding to predictions.
        """
        entropy = torch.sum(-torch.log(predictions) * predictions, dim=-1)
        bin_indices = torch.searchsorted(self.bin_boundaries, entropy) - 1
        q_hats = torch.stack([self.q_hats[i] for i in bin_indices])

        return q_hats

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Adapt the predictions of the model but zeroing out all tokens that are not included in the prediction set.

        Parameters
        ----------
        input_ids: torch.LongTensor
            Input ids for the current batch.
        scores: torch.FloatTensor
            Predictions of the model.

        Returns
        -------
        torch.FloatTensor
            Adapted model predictions.
        """
        cur_len = input_ids.shape[-1]

        # Avoid any modifications to the scores by the ForcedBOSTokenLogitsProcessor
        if cur_len > 1:
            scores = F.softmax(scores, dim=-1)

            q_hats = self.get_q_hats(scores).to(scores.device)
            scores = self.calibrator.get_prediction_sets(self.conformity_score, scores, q_hats)[0]

            # Put back into log space
            scores = torch.log(scores + 1e-12)

        return scores


class NonExchangeableConformalLogitProcessor(LogitsProcessor):
    """
    LogitsProcessor that uses the non-exchangeable conformal prediction framework to compute prediction sets. This
    corresponds to our proposed method, non-exchangeable conformal language generation through nearest neighbors.
    """
    def __init__(
        self,
        conformity_score: str,
        distance_type: str,
        num_neighbors: int,
        data_store,
        calibrator: ConformalCalibrator,
        store_set_sizes: bool = False,
    ):
        """
        Initialize a NonExchangeableConformalLogitProcessor.

        Parameters
        ----------
        conformity_score: str
            Name of conformity score method being used. Should be "classic" or "adapted".
        distance_type: str
            Type of distance measure being used. Either has to be "inner_product", "l2" or "cosine".
        num_neighbors: int
            Number of neighbors to consider when computing quantiles.
        data_store: DataStore
            Pre-populated datastore.
        calibrator: ConformalCalibrator
            Conformal calibrator.
        store_set_sizes: bool
            Flag to indicate whether set sizes should be stored or not.
        """
        super().__init__()
        self.distance_type = distance_type
        self.num_neighbors = num_neighbors
        self.conformity_score = conformity_score
        self.data_store = data_store
        self.calibrator = calibrator

        self.store_set_sizes = store_set_sizes

        self.last_decoder_encodings = None
        self.last_set_sizes = []

    def patch_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Patch the model forward pass to save the decoder hidden encodings for generation with conformal prediction sets.

        Parameters
        ----------
        model: PreTrainedModel
            Model whose forward pass should be patched.

        Returns
        -------
        PreTrainedModel
            Patched model.
        """
        def hook_fn(model, inputs, outputs):
            if isinstance(model, M2M100PreTrainedModel):
                self.last_decoder_encodings = outputs.decoder_hidden_states[-1].squeeze(1)

            else:
                self.last_decoder_encodings = outputs.hidden_states[-1][:, -1, :]

            if self.distance_type == "inner_product":
                self.last_decoder_encodings /= model.config.d_model ** 0.25

            elif self.distance_type == "cosine":
                self.last_decoder_encodings = F.normalize(self.last_decoder_encodings, p=2, dim=-1)

        model.register_forward_hook(hook_fn)

        return model

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Adapt the predictions of the model but zeroing out all tokens that are not included in the prediction set.

        Parameters
        ----------
        input_ids: torch.LongTensor
            Input ids for the current batch.
        scores: torch.FloatTensor
            Predictions of the model.

        Returns
        -------
        torch.FloatTensor
            Adapted model predictions.
        """
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
            scores, set_sizes = self.calibrator.get_prediction_sets(self.conformity_score, scores, q_hat)

            if self.store_set_sizes:
                self.last_set_sizes.append(set_sizes)

            # Put back into log space
            scores = torch.log(scores + 1e-12)

        return scores
