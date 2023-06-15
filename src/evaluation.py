"""
Module implementing necessary function for evaluating NMT models.
"""

# STD
import codecs
import os
import re
import subprocess
from typing import Dict, List, Tuple, Optional, Union
import random

# EXT
import numpy as np
import evaluate

# PROJECT
from src.custom_types import Device


def evaluate_translation_model(
    translations: Union[List[str], List[List[str]]],
    source_file: str,
    reference_file: str,
    use_mbr: bool,
    metrics: Tuple[str] = ("bleu", "chrf", "comet"),
    device: Optional[Device] = None
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

    # Use minimum Bayes risk decoding - we write all samples for a source into a file, use COMET to score them and
    # the select the best scoring one
    if use_mbr:
        rnd = random.randint(0, 100000)  # Add random number to avoid collisions
        num_samples = len(translations)
        num_translations = len(translations[0])

        with codecs.open(f"mbr_{rnd}.txt", "w", "utf-8") as f:
            for translation in range(num_translations):
                for sample in range(num_samples):
                    f.write(f"{translations[sample][translation]}\n")

        # Run COMET MBR script
        num_samples = len(translations)
        comet_command = f"comet-mbr -s {source_file} -t mbr_{rnd}.txt --num_sample {num_samples} -o mbr_out_{rnd}.txt"

        if device is not None:
            comet_command += f" --gpus {device.split(':')[-1]}"

        process = subprocess.Popen(comet_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        # Propagate error (if one occured)
        if error is not None:
            raise RuntimeError(error)

        # Load best translations
        with codecs.open(f"mbr_out_{rnd}.txt", "r", "utf-8") as f:
            translations = [line.strip() for line in f.readlines()]

        # Remove temporary files
        os.remove(f"mbr_{rnd}.txt")

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


def evaluate_generation_model(
    generations: List[str],
    reference_file: str,
    metrics: Tuple[str] = ("mauve", "bleurt", "bert_score"),
    device: Optional[Device] = None
):
    # Load reference generations
    with codecs.open(reference_file, "r", "utf-8") as f:
        reference_generations = [line.strip() for line in f.readlines()]

    # Evaluate translations
    result_dict = {}

    if "mauve" in metrics:
        mauve = evaluate.load("mauve")
        mauve_results = mauve.compute(
            predictions=generations, references=reference_generations, featurize_model_name="gpt2",
            device_id=int(device.split(":")[-1]) if device is not None else None,
        )
        result_dict["mauve"] = mauve_results.mauve
        del mauve

    if "bleurt" in metrics:
        bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="bleurt-tiny-128")
        bleurt_results = bleurt.compute(predictions=generations, references=reference_generations)
        result_dict["bleurt"] = np.mean(bleurt_results["scores"])
        del bleurt

    if "bert_score" in metrics:
        bertscore = evaluate.load("bertscore", lang="en")
        bertscore_results = bertscore.compute(
            predictions=generations, references=reference_generations, lang="en", model_type="distilbert-base-uncased",
            device=device
        )
        result_dict["bert_score"] = np.mean(bertscore_results["f1"])
        del bertscore

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
    rnd = random.randint(0, 100000)  # Add random number to avoid collisions

    # Create temp files
    with codecs.open(f"translations_{rnd}.tmp", "w", "utf-8") as f:
        f.write("\n".join(translations))

    with codecs.open(f"sources_{rnd}.tmp", "w", "utf-8") as f:
        f.write("\n".join(sources))

    with codecs.open(f"references_{rnd}.tmp", "w", "utf-8") as f:
        f.write("\n".join(references))

    # TODO: Keep for now in case we need to use it for CometKiwi
    comet_command = f"comet-score -s sources_{rnd}.tmp -t translations_{rnd}.tmp -r references_{rnd}.tmp"

    if not use_gpu:
        comet_command += " --gpus 0"

    process = subprocess.Popen(comet_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Propagate error (if one occured)
    if error is not None:
        raise RuntimeError(error)

    # Catch final score in output
    print(output)  # TODO: Debug
    score = re.search(r"score: (\d+.\d+)", str(output)).group(1)
    score = float(score)

    # Remove temp files
    os.remove(f"translations_{rnd}.tmp")
    os.remove(f"sources_{rnd}.tmp")
    os.remove(f"references_{rnd}.tmp")

    return score
