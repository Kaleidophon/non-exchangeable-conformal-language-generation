"""
Define the core functions for conformal risk control in NLG.
"""

# STD
import dill
from functools import reduce
from operator import add

# EXT
from einops import rearrange
import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class CalibrationData:
    """
    Class for storing calibration data.
    """

    def __init__(self, num_hypotheses: int):
        """
        Initialize a calibration data store.

        Parameters
        ----------
        num_hypotheses: int
            Number of hypotheses to be stored per claibration point.
        """
        self.num_hypotheses = num_hypotheses
        self.losses = torch.empty([0, num_hypotheses])
        self.log_probs = torch.empty([0, num_hypotheses])

    def add_calibration_points(self, losses: torch.Tensor, log_probs: torch.Tensor):
        """
        Add a calibration point to the calibration data store.

        Parameters
        ----------
        losses: torch.Tensor
            Losses of the hypotheses.
        log_probs: torch.Tensor
            Log probabilities of the hypotheses.
        """
        # Sort by log probs and add to data store
        log_probs, indices = log_probs.sort(descending=True)
        losses = torch.gather(losses, dim=1, index=indices)

        self.losses = torch.cat([self.losses, losses])
        self.log_probs = torch.cat([self.log_probs, log_probs])

    def get_losses(self, k: int):
        """
        Get the losses of the k best hypotheses.

        Parameters
        ----------
        k: int
            Number of hypotheses to be considered.

        Returns
        -------
        torch.Tensor
            Losses of the k best hypotheses.
        """
        return self.losses[:, :k]

    @property
    def size(self):
        return self.losses.shape[0]

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return dill.load(f)

    def save(self, path: str):
        with open(path, "wb") as f:
            dill.dump(self, f)


def build_calibration_data(
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    data_loader: DataLoader,
    num_beams: int,
    source_path: str,
    references_path: str,
    max_length: int = 64,  # TODO: Debug 256,
) -> CalibrationData:
    """
    Build calibration data from dev set using the specified model.

    Parameters
    ----------
    model: MBartForConditionalGeneration
        Model to be used for the experiments.
    tokenizer: MBart50TokenizerFast
        Tokenizer to be used for the experiments.
    data_loader: DataLoader
        Data loader to be used for the experiments.
    num_beams: int
        Number of beams to be used for the experiments.
    source_path: str
        Path to the source file. Used for evaluating translations.
    references_path: str
        Path to the references file. Used for evaluating translations.

    Returns
    -------
    CalibrationData
        Calibration data, including the losses (i.e. the quality) and log probabilities of the hypotheses.
    """
    calibration_data = CalibrationData(num_beams)
    comet_metric = evaluate.load('bleu')  # TODO: Change this back to comet

    model.eval()

    # Load source sentences and references
    with open(source_path, "r") as f:
        sources = [line.strip() for line in f.readlines()]

    # Load reference translations
    with open(references_path, "r") as f:
        references = [line.strip() for line in f.readlines()]

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_size = len(batch["input_ids"])

        # Get input and target
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Generate hypotheses
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
        batch_translations = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        batch_scores = outputs.sequences_scores

        # Duplicate references and sources for this batch since we have every source represented multiple times (for
        # every beam) on the hypotheses
        batch_references = reduce(add, [
           [references[j]] * num_beams
           for j in range(i * batch_size, (i + 1) * batch_size)
        ])
        batch_sources = reduce(add, [
            [sources[j]] * num_beams
            for j in range(i * batch_size, (i + 1) * batch_size)
        ])
        comet_scores = torch.FloatTensor([
            comet_metric.compute(
                predictions=[trans], references=[ref],  # sources=batch_sources
            )["bleu"] / 100   # TODO: Change this back to comet
            for trans, ref, source in zip(batch_translations, batch_references, batch_sources)
        ])
        batch_losses = 1 - comet_scores

        # Do some reshaping
        batch_losses = rearrange(batch_losses, "(batch_size num_beams) -> batch_size num_beams", batch_size=batch_size)
        batch_scores = rearrange(batch_scores, "(batch_size num_beams) -> batch_size num_beams", batch_size=batch_size)

        # Add calibration points
        calibration_data.add_calibration_points(batch_losses, batch_scores)

    return calibration_data


def get_optimal_k(calibration_data: CalibrationData, alpha: float, b: float = 1.0):
    """
    Find the optimal k based on the calibration data.
    The optimal k is the one for which N / (N + 1) * r_hat + b / (N + 1) <= alpha,
    which we can find by binary search in O(log num_hypotheses).

    Parameters
    ----------
    calibration_data: CalibrationData
        Calibration data to be used for the evaluation.
    alpha: float
        Pre-specified confidence level.
    b: float
        Upper bound of the loss function.

    Returns
    -------
    int k
        Optimal k given the calibration data.
    """
    # Binary search for k
    k_min = 1
    k_max = calibration_data.num_hypotheses

    while k_max - k_min > 1:
        current_k = (k_min + k_max) // 2

        if evaluate_k(current_k, calibration_data, b) <= alpha:
            k_min = current_k

        else:
            k_max = current_k

    return k_min


def evaluate_k(k: int, calibration_data: CalibrationData, b: float):
    """
    Evaluate a specific choice of k based on the calibration data.

    Parameters
    ----------
    k: int
        Current value of k to be evaluated.
    calibration_data: CalibrationData
        Calibration data to be used for the evaluation.
    b: float
        Upper bound of the loss function.

    Returns
    -------
    float
        Value of for the current value of k that is going to be compared with alpha.
    """
    losses = calibration_data.get_losses(k)
    losses = torch.max(losses, dim=1)[0]

    N = calibration_data.size
    r_hat = torch.mean(losses)

    res = N / (N + 1) * r_hat + b / (N + 1)

    return res
