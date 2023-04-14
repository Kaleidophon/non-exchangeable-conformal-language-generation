"""
Define the core functions for conformal risk control in NLG.
"""

# EXT
import torch

def conformity_score(predictions: torch.FloatTensor, lambda_: float, metric):
    ...