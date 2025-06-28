"""
open_library.py

This script processes a gzipped Open Library works dump, extracts descriptions,
and filters them based on specified criteria such as
word count, punctuation, and exclusion of certain words.
"""

import gzip
import json
from functools import partial
from pathlib import Path
from typing import Optional

import gcld3
import typer
from tqdm import tqdm
from typing_extensions import Annotated


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
    detector: Optional[gcld3.NNetLanguageIdentifier] = None,
    language: Optional[str] = None,
    min_language_proportion: float = 0.99,
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
        detector (Optional[gcld3.NNetLanguageIdentifier]): Language detector.
        language (Optional[str]): Expected language of the text.
        min_language_proportion (float): Mininum same-language text proportion.
    Returns:
        bool: True if the description should be kept, False otherwise.
    """
    # language check
    if detector:
        result = detector.FindLanguage(text)
        if (
            result.language != language
            or result.proportion < min_language_proportion
        ):
            return False

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

    if exclude_words and any(word in text for word in exclude_words):
        return False

    return True


def main(
    open_library_path: Annotated[
        Path,
        typer.Argument(
            ...,
            help="Path to the gzipped Open Library works dump file",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    exclude_words_path: Annotated[
        Path,
        typer.Option(
            "--exclude-words",
            help="Path to the file containing words to exclude",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("data/exclude_words.txt"),
    dataset_path: Annotated[
        Path,
        typer.Option(
            help="Path to save the filtered descriptions",
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("data/one-sentence-summaries.json"),
    min_words: Annotated[
        int, typer.Option(help="Minimum number of words required", min=1)
    ] = 15,
    max_words: Annotated[
        int, typer.Option(help="Maximum number of words allowed", min=1)
    ] = 50,
    max_punctuation: Annotated[
        int,
        typer.Option(
            help="Maximum punctuation characters allowed",
            min=0,
        ),
    ] = 5,
    punctuation: Annotated[
        str, typer.Option(help="Punctuation characters to check against")
    ] = ";-_",
    separator: Annotated[
        str,
        typer.Option(
            help="Separator used in the Open Library dump file",
        ),
    ] = "\t",
    description_key: Annotated[
        str,
        typer.Option(
            help="Key in the JSON object that contains the description",
        ),
    ] = "description",
    value_key: Annotated[
        str,
        typer.Option(
            help="Key in the JSON object for text values",
        ),
    ] = "value",
    language: Annotated[
        str,
        typer.Option(
            help="Cld3 language code for the language to filter by",
        ),
    ] = "en",
    cld3_max_num_bytes: Annotated[
        int,
        typer.Option(
            help="Maximum number of bytes for language detection",
            min=1,
        ),
    ] = 1000,
    cld3_min_language_proportion: Annotated[
        float,
        typer.Option(
            help="Minimum proportion of same-language text required",
            min=0.0,
            max=1.0,
        ),
    ] = 0.99,
) -> None:
    """Process the Open Library dump and filter descriptions.

    This command reads a gzipped file containing Open Library works data,
    extracts descriptions, and filters them based on specified criteria.

    Args:
        open_library_path (Path): The gzipped Open Library works dump file.
        exclude_words_path (Path): File containing words to exclude.
        dataset_path (Path): Path to save the filtered descriptions.
        min_words (int): Minimum number of words required.
        max_words (int): Maximum number of words allowed.
        max_punctuation (int): Maximum punctuation characters allowed.
        punctuation (str): Punctuation characters to check against.
        separator (str): Separator used in the Open Library dump file.
        description_key (str): Key in the JSON object with the description.
        value_key (str): Key in the JSON object for text values.
        language (str): Cld3 language code for the language to filter by.
        cld3_max_num_bytes (int): Max number of bytes for language detection.
        cld3_min_language_proportion (float): Minimum proportion of
        same-language text required.
    Returns:
        None
    """
    exclude_words = load_exclude_words(exclude_words_path)

    should_keep_fn = partial(
        should_keep,
        min_words=min_words,
        max_words=max_words,
        max_punctuation=max_punctuation,
        punctuation=punctuation,
        exclude_words=exclude_words,
        detector=gcld3.NNetLanguageIdentifier(
            min_num_bytes=0,
            max_num_bytes=cld3_max_num_bytes,
        ),
        language=language,
        min_language_proportion=cld3_min_language_proportion,
    )

    file = gzip.open(open_library_path, "rt")
    typer.echo("Processing Open Library works dump...")

    descs = []
    for line in (pbar := tqdm(file)):
        edition = json.loads(line.split(separator)[-1])

        if description_key not in edition:
            continue

        desc = edition[description_key]
        if not isinstance(desc, str):
            desc = desc[value_key]

        if not should_keep_fn(desc):
            continue

        descs.append(desc)
        pbar.set_description(f"{len(descs)} included")

        # typer.echo(desc)
        # typer.echo("-" * 80)

    json.dump(
        descs,
        dataset_path.open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    typer.echo("Processing complete.")
    file.close()


if __name__ == "__main__":
    typer.run(main)
