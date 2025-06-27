"""
open_library.py

This script processes a gzipped Open Library works dump, extracts descriptions,
and filters them based on specified criteria such as
word count, punctuation, and exclusion of certain words.
"""

import gzip
import json
from typing import Optional


def load_exclude_words(file_path: str) -> set[str]:
    """
    Load a list of words to exclude from the descriptions.
    The file should contain one word per line.

    Args:
        file_path (str): Path to the file containing words to exclude.
    Returns:
        set[str]: A set of words to exclude.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No words will be excluded.")
        return []


def should_keep(
    text: str,
    min_words: int = 15,
    max_words: int = 50,
    max_punctuation: int = 5,
    punctuation: str = ";-_",
    exclude_words: Optional[set] = None,
) -> bool:
    """
    Determine if a text description should be kept based on various criteria.
    Args:
        text (str): The text description to evaluate.
        min_words (int): Minimum words required.
        max_words (int): Maximum words allowed.
        max_punctuation (int): Maximum punctuation characters allowed.
        punctuation (str): String of punctuation characters to check against.
        exclude_words (Optional[set]): Set of words that should return False.
    Returns:
        bool: True if the description should be kept, False otherwise.
    """
    # todo: language check

    # exclude if more caps than lower
    if sum(1 for c in text if c.isupper()) > sum(
        1 for c in text if c.islower()
    ):
        return False

    # exclude if more than 5 semicolons, hyphens, or underscores
    if any(text.count(char) > max_punctuation for char in punctuation):
        return False

    text = text.lower()
    words = text.split(" ")

    if len(words) < min_words or len(words) > max_words:
        return False

    if exclude_words and any(word in text for word in set(exclude_words)):
        return False
    return True


def main() -> None:
    """Main function to process the Open Library dump and filter descriptions.
    It reads a gzipped file containing Open Library works data,
    extracts descriptions, and filters them based on specified criteria.

    Args:
        None
    Returns:
        None
    """
    exclude_words = load_exclude_words("data/ignore_words.txt")

    file = gzip.open("ol_dump_works_2025-05-31.txt.gz", "rt")
    print("Processing Open Library works dump...")

    for line in file:
        edition = json.loads(line.split("\t")[-1])

        if "description" in edition:
            desc = edition["description"]
            if not isinstance(desc, str):
                desc = desc["value"]

            if not should_keep(
                desc,
                min_words=15,
                max_words=50,
                max_punctuation=5,
                punctuation=";-_",
                exclude_words=exclude_words,
            ):
                continue

            print(desc)

    print("Processing complete.")
    file.close()


if __name__ == "__main__":
    main()
