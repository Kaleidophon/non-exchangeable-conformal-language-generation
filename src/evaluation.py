"""
Module implementing necessary function for evaluating NMT models.
"""

# STD
import codecs
import os
import re
import subprocess
from typing import Dict, List, Tuple

# EXT
import evaluate


def evaluate_model(
    translations: List[str],
    source_file: str,
    reference_file: str,
    metrics: Tuple[str] = ("bleu", "chrf", "comet"),
) -> Dict[str, float]:
    """
    Evaluate a model on the test set.

    Parameters
    ----------
    source_file: str
        Path to the source file.
    reference_file:
        Path to the reference file.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the evaluation results.
    """

    # Load source sentences
    with codecs.open(source_file, "r", "utf-8") as f:
        source_sentences = [line.strip() for line in f.readlines()]

    # Load reference translations
    with codecs.open(reference_file, "r", "utf-8") as f:
        reference_translations = [line.strip() for line in f.readlines()]

    # Evaluate translations
    result_dict = {}

    if "bleu" in metrics:
        bleu = evaluate.load("sacrebleu")
        bleu_results = bleu.compute(predictions=translations, references=reference_translations)
        result_dict["bleu"] = bleu_results["score"]

    if "comet" in metrics:
        comet_result = evaluate_comet(translations=translations, sources=source_sentences, references=reference_translations)
        result_dict["comet"] = comet_result

    if "chrf" in metrics:
        chrf = evaluate.load("chrf")
        chrf_results = chrf.compute(predictions=translations, references=reference_translations)
        result_dict["chrf"] = chrf_results["score"]

    return result_dict


def evaluate_comet(
    translations: List[str],
    sources: List[str],
    references: List[str],
    use_gpu: bool = False
) -> float:
    """
    Evaluate translations with COMET.

    Parameters
    ----------
    translations_path: str
        Path to generated translations.
    source_path: str
        Path to source sentences.
    references_path: str
        Path to reference translations.
    use_gpu: bool
        Indicate whether GPU is available for faster eval.
    """
    # Create temp files
    with codecs.open("translations.tmp", "w", "utf-8") as f:
        f.write("\n".join(translations))

    with codecs.open("sources.tmp", "w", "utf-8") as f:
        f.write("\n".join(sources))

    with codecs.open("references.tmp", "w", "utf-8") as f:
        f.write("\n".join(references))

    # TODO: Keep for now in case we need to use it for CometKiwi
    comet_command = f"comet-score -s sources.tmp -t translations.tmp -r references.tmp"

    if not use_gpu:
        comet_command += " --gpus 0"

    process = subprocess.Popen(comet_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Propagate error (if one occured)
    if error is not None:
        raise RuntimeError(error)

    # Catch final score in output
    score = re.search(r"score: (\d+.\d+)", str(output)).group(1)
    score = float(score)

    # Remove temp files
    os.remove("translations.tmp")
    os.remove("sources.tmp")
    os.remove("references.tmp")

    return score
