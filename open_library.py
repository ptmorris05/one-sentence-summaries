"""
open_library.py

This script processes a gzipped Open Library works dump, extracts descriptions,
and filters them based on specified criteria such as
word count, punctuation, and exclusion of certain words.
"""

import gzip
import json
from pathlib import Path
from typing import Optional

import typer


def load_exclude_words(file_path: Path) -> set[str]:
    """
    Load a list of words to exclude from the descriptions.
    The file should contain one word per line.

    Args:
        file_path (Path): Path to the file containing words to exclude.
    Returns:
        set[str]: A set of words to exclude.
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        print(
            f"Warning: {file_path} not found. Not using word-based exclusion."
        )
        return set()


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


def main(
    open_library_path: Path = typer.Argument(
        ...,
        help="Path to the gzipped Open Library works dump file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    exclude_words_path: Path = typer.Option(
        "data/exclude_words.txt",
        help="Path to the file containing words to exclude",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    min_words: int = typer.Option(
        15, help="Minimum number of words required", min=1
    ),
    max_words: int = typer.Option(
        50, help="Maximum number of words allowed", min=1
    ),
    max_punctuation: int = typer.Option(
        5,
        help="Maximum punctuation characters allowed",
        min=0,
    ),
    punctuation: str = typer.Option(
        ";-_", help="Punctuation characters to check against"
    ),
) -> None:
    """Process the Open Library dump and filter descriptions.

    This command reads a gzipped file containing Open Library works data,
    extracts descriptions, and filters them based on specified criteria.

    Args:
        open_library_path (Path): The gzipped Open Library works dump file.
        exclude_words_path (Path): File containing words to exclude.
        min_words (int): Minimum number of words required.
        max_words (int): Maximum number of words allowed.
        max_punctuation (int): Maximum punctuation characters allowed.
        punctuation (str): Punctuation characters to check against.
    Returns:
        None
    """
    exclude_words = load_exclude_words(exclude_words_path)

    file = gzip.open(open_library_path, "rt")
    print("Processing Open Library works dump...")

    for line in file:
        edition = json.loads(line.split("\t")[-1])

        if "description" not in edition:
            continue

        desc = edition["description"]
        if not isinstance(desc, str):
            desc = desc["value"]

        if not should_keep(
            desc,
            min_words=min_words,
            max_words=max_words,
            max_punctuation=max_punctuation,
            punctuation=punctuation,
            exclude_words=exclude_words,
        ):
            continue

        print(desc)

    print("Processing complete.")
    file.close()


if __name__ == "__main__":
    typer.run(main)
