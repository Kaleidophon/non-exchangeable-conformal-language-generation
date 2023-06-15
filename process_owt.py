"""
Quick and dirty script to create evaluation data based on openwebtext.
"""

# STD
import codecs

DATA_PATH = "./data/openwebtext/openwebtext.txt"
OUTPUT_PATHS = ("./data/openwebtext/test.txt", "./data/openwebtext/references.txt")
NUM_PROMPTS = 1000
MIN_LENGTH = 50

if __name__ == "__main__":
    with codecs.open(DATA_PATH, "r", "utf-8") as f:
        with codecs.open(OUTPUT_PATHS[0], "w", "utf-8") as out_file1, codecs.open(OUTPUT_PATHS[1], "w", "utf-8") as out_file2:

            for line in f.readlines():
                if len(line.split(" ")) < MIN_LENGTH:
                    continue

                out_file1.write(line)
                out_file2.write(line)
