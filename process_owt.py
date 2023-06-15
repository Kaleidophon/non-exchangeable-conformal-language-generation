"""
Quick and dirty script to create evaluation data based on openwebtext.
"""

# STD
import codecs

DATA_PATH = "./data/openwebtext/openwebtext.txt"
OUTPUT_PATHS = ("./data/openwebtext/test.txt", "./data/openwebtext/references.txt")
NUM_PROMPTS = 1000
MIN_LENGTH = 50
MAX_LENGTH = 200

if __name__ == "__main__":
    with codecs.open(DATA_PATH, "r", "utf-8") as f:
        with codecs.open(OUTPUT_PATHS[0], "w", "utf-8") as out_file1, codecs.open(OUTPUT_PATHS[1], "w", "utf-8") as out_file2:
            num_lines = 0

            for line in f.readlines():
                if len(line.split(" ")) < MIN_LENGTH or len(line.split(" ")) < 200:
                    continue

                out_file1.write(line)
                out_file2.write(line)

                num_lines += 1

                if num_lines >= NUM_PROMPTS:
                    break
