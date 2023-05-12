"""
Module implementing necessary function for evaluating NMT models.
"""

# STD
import re
import subprocess
from typing import Dict, List, Tuple

# EXT
import evaluate
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


def evaluate_model(
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    data_loader: DataLoader,
    source_file: str,
    reference_file: str,
    metrics: Tuple[str] = ("bleu", "chrf", "comet"),
) -> Dict[str, float]:
    """
    Evaluate a model on the test set.

    Parameters
    ----------
    model: MBartForConditionalGeneration
        Model to be evaluated.
    tokenizer: MBart50TokenizerFast
        Tokenizer used for the model.
    data_loader: DataLoader
        DataLoader for the test set.
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
    with open(source_file, "r") as f:
        source_sentences = [line.strip() for line in f.readlines()]

    # Load reference translations
    with open(reference_file, "r") as f:
        reference_translations = [line.strip() for line in f.readlines()]

    # Generate translations
    print("Generate translations...")
    translations = generate_test_translations(model, tokenizer, data_loader)
    print(translations)

    # Evaluate translations
    result_dict = {}

    if "bleu" in metrics:
        print("Evaluate BLEU...")
        bleu = evaluate.load("sacrebleu")
        bleu_results = bleu.compute(predictions=translations, references=reference_translations)
        print(bleu_results)
        result_dict["bleu"] = bleu_results["score"]

    if "comet" in metrics:
        # TODO: Comet doesn't actually give scores here for a weird reason
        comet_metric = evaluate.load('comet', 'Unbabel/wmt20-comet-da')
        comet_score = comet_metric.compute(
            predictions=translations, references=reference_translations, sources=source_sentences
        )
        result_dict["comet1"] = comet_score["scores"][0]
        result_dict["comet2"] = comet_score["scores"][1]

    if "chrf" in metrics:
        chrf = evaluate.load("chrf")
        chrf_results = chrf.compute(predictions=translations, references=reference_translations)
        result_dict["chrf"] = chrf_results["score"]

    return result_dict


def generate_test_translations(
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    data_loader: DataLoader,
) -> List[str]:
    """
    Generate translations for test set and write them to a file.

    Parameters
    ----------
    model: MBartForConditionalGeneration
        Model to use for translation.
    tokenizer: MBart50TokenizerFast
        Tokenizer to use for translation.
    data_loader: DataLoader
        Data loader for test set.
    """
    model.eval()

    translations = []

    for batch in data_loader:
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            num_beams=4,
            max_length=128,
            early_stopping=True,
            decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"]
        )

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        translations += outputs

    return translations


def evaluate_comet(
    translations_path: str,
    source_path: str,
    references_path: str,
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
    # TODO: Keep for now in case we need to use it for CometKiwi
    comet_command = f"comet-score -s {source_path} -t {translations_path} -r {references_path}"

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

    return score
