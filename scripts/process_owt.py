"""
Quick and dirty script to create evaluation data based on openwebtext.
"""

# STD
import codecs
from datasets import load_dataset

DATA_PATH = "../data/openwebtext/openwebtext2.txt"
OUTPUT_PATHS = ("../data/openwebtext/test.txt", "../data/openwebtext/references.txt")
NUM_PROMPTS = 1000
MIN_LENGTH = 50
MAX_LENGTH = 200

if __name__ == "__main__":
    dataset_name = "stas/openwebtext-10k"
    name = dataset_name.split('/')[-1]
    ds = load_dataset(dataset_name, split='train')
    sents = [d["text"] for d in ds][:1000]
    prefixes = [" ".join(s.split(" ")[:35]) for s in sents]

    with codecs.open(OUTPUT_PATHS[0], "w", "utf-8") as test_file:
        for prefix in prefixes:
            test_file.write(prefix + "</s>")

    with codecs.open(OUTPUT_PATHS[1], "w", "utf-8") as ref_file:
        for sent in sents:
            ref_file.write(sent + "</s>")
