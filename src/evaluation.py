"""
Module implementing necessary function for evaluating NMT models.
"""

# STD
from collections import namedtuple
import re
import subprocess
from typing import Tuple

# EXT
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# TYPES
SacreBleuResult = namedtuple("SacreBleuResult", ["bleu", "chrF"])


def generate_test_translations(
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    data_loader: DataLoader,
    target_file: str
) -> None:
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
    target_file: str
        Path to file where translations should be written.
    """
    model.eval()

    with open(target_file, "w") as f:

        for batch in data_loader:
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                num_beams=4,
                max_length=256,
                early_stopping=True
            )
            for output in outputs:
                f.write(tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) + "\n")


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


def evaluate_sacrebleu(
    translations_path: str,
    references_path: str,
    src_lang: str,
    tgt_lang: str
) -> Tuple[float, float]:
    """
    Evaluate translations with BLEU and chrF using sacrebleu.

    Parameters
    ----------
    translations_path: str
        Path to generated translations.
    references_path: str
        Path to reference translations.
    src_lang: str
        Source language abbreviation.
    tgt_lang: str
        Target language abbreviation.
    """
    sacrebleu_command = f"sacrebleu {references_path} -i {translations_path} -m bleu chrf -b -w 4 -l {src_lang}-{tgt_lang}"

    process = subprocess.Popen(sacrebleu_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Propagate error (if one occured)
    if error is not None:
        raise RuntimeError(error)

    bleu, chrf = eval(output)  # Usually bad style but I know which process this output comes from

    return SacreBleuResult(bleu=bleu, chrF=chrf)
