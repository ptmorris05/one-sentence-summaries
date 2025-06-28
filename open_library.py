"""
open_library.py

This script processes a gzipped Open Library works dump, extracts descriptions,
and filters them based on specified criteria such as
word count, punctuation, and exclusion of certain words.
"""

import gzip
import json
import os
from functools import partial
from pathlib import Path
from typing import Optional

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
    detector: Optional[object] = None,
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
        detector (Optional[object]): Language detector.
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

    # exclude if more than max_punctuation instaces of any punctuation
    if any(text.count(char) > max_punctuation for char in punctuation):
        return False

    text = text.lower()
    words = text.split(" ")

    if len(words) < min_words or len(words) > max_words:
        return False

    if exclude_words and any(word in text for word in exclude_words):
        return False

    return True


def embedding_filter(
    client: object,
    descriptions: list[str],
    batch_size: int = 100,
    dataset_size: int = 20000,
    n_clusters: int = 100,
) -> list[str]:
    """
    Filter descriptions using embeddings.

    Args:
        client (object): The embedding client.
        descriptions (list[str]): List of descriptions to filter.
        batch_size (int): Batch size for processing embeddings.
        dataset_size (int): Size of the final dataset after filtering.
        n_clusters (int): Number of clusters for KMeans clustering.
    Returns:
        list[str]: Filtered list of descriptions.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    # embed the descriptions
    typer.echo("Generating embeddings for descriptions...")
    embeddings = []
    for i in tqdm(range(0, len(descriptions), batch_size)):
        batch = descriptions[i : i + batch_size]
        result = client.models.embed_content(contents=batch)
        for embedding in result.embeddings:
            embeddings.append(embedding.values)

    # normalize for cosine similarity
    embeddings = np.array(embeddings, dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # perform clustering
    typer.echo("Clustering embeddings...")
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings)

    # get the index of the cluster center of the largest cluster
    largest_cluster = np.argmax(np.bincount(kmeans.labels_))
    center = embeddings[kmeans.labels_ == largest_cluster].mean(axis=0)

    # filter descriptions based on distance to the cluster center
    typer.echo("Filtering descriptions based on cluster center...")
    scores = embeddings @ center
    top_score_idxs = np.argpartition(scores, -dataset_size)[-dataset_size:]

    return [descriptions[idx] for idx in top_score_idxs]


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
    filter_language: Annotated[
        bool,
        typer.Option(
            help="Filter descriptions by language using Cld3",
            is_flag=True,
        ),
    ] = False,
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
    filter_embedding: Annotated[
        bool,
        typer.Option(
            help="Filter descriptions using embeddings",
            is_flag=True,
        ),
    ] = False,
    embedding_model: Annotated[
        str,
        typer.Option(
            help="Gemini Embedding model to use for filtering",
        ),
    ] = "text-embedding-004",
    embedding_batch_size: Annotated[
        int,
        typer.Option(
            help="Batch size for processing embeddings",
            min=1,
            max=100,
        ),
    ] = 100,
    embedding_dimensions: Annotated[
        int,
        typer.Option(
            help="Dimensions of the embedding model",
            min=1,
            max=768,
        ),
    ] = 256,
    embedding_task_type: Annotated[
        str,
        typer.Option(
            help="Gemini task type for embeddings, e.g., 'CLUSTERING'",
        ),
    ] = "CLUSTERING",
    embedding_dataset_size: Annotated[
        int,
        typer.Option(
            help="Size of the final dataset after embedding filtering",
            min=1,
        ),
    ] = 20000,
    embedding_clusters: Annotated[
        int,
        typer.Option(
            help="Number of clusters for KMeans clustering",
            min=1,
        ),
    ] = 100,
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
        filter_language (bool): Whether to filter descriptions by language.
        language (str): Cld3 language code for the language to filter by.
        cld3_max_num_bytes (int): Max number of bytes for language detection.
        cld3_min_language_proportion (float): Minimum proportion of
        same-language text required.
        filter_embedding (bool): Filter descriptions with embeddings.
        embedding_model (str): Gemini Embedding model to use for filtering.
        embedding_batch_size (int): Batch size for processing embeddings.
        embedding_dimensions (int): Dimensions of the embedding model.
        embedding_task_type (str): Gemini task type for embeddings.
        embedding_dataset_size (int): Size of the final dataset.
        embedding_clusters (int): Number of clusters for KMeans clustering.
    Returns:
        None
    """
    exclude_words = load_exclude_words(exclude_words_path)

    # language filtering setup
    language_kwargs = {}
    if filter_language:
        try:
            import gcld3
        except ImportError:
            typer.echo("gcld3 not installed. Cannot filter by language.")
            raise typer.Exit(code=1)

        typer.echo(f"Using Cld3 language filtering for language: {language}")

        language_kwargs = {
            "detector": gcld3.NNetLanguageIdentifier(
                min_num_bytes=0,
                max_num_bytes=cld3_max_num_bytes,
            ),
            "language": language,
            "min_language_proportion": cld3_min_language_proportion,
        }

    # setup
    should_keep_fn = partial(
        should_keep,
        min_words=min_words,
        max_words=max_words,
        max_punctuation=max_punctuation,
        punctuation=punctuation,
        exclude_words=exclude_words,
        **language_kwargs,
    )

    file = gzip.open(open_library_path, "rt")
    typer.echo("Processing Open Library works dump...")

    # read the file line by line and filter descriptions
    descriptions = []
    for line in (pbar := tqdm(file)):
        edition = json.loads(line.split(separator)[-1])

        if description_key not in edition:
            continue

        description = edition[description_key]
        if not isinstance(description, str):
            description = description[value_key]

        if not should_keep_fn(description):
            continue

        descriptions.append(description)
        pbar.set_description(f"{len(descriptions)} included")

    typer.echo(f"Included {len(descriptions)} descriptions.")

    # remove duplicates
    descriptions = list(set(descriptions))
    typer.echo(f"Removed duplicates, {len(descriptions)} remaining.")

    # filter by embeddings
    if filter_embedding:
        # load GEMINI_API_KEY from .env file
        try:
            from dotenv import load_dotenv
        except ImportError:
            typer.echo("python-dotenv not installed. Cannot load .env file.")
            raise typer.Exit(code=1)

        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            typer.echo("GEMINI_API_KEY not set in .env file.")
            raise typer.Exit(code=1)

        # setup embedding client
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            typer.echo("google-genai not installed. Cannot use embeddings.")
            raise typer.Exit(code=1)

        client = genai.Client(api_key=api_key)
        client.models.embed_content = partial(
            client.models.embed_content,
            model=embedding_model,
            config=types.EmbedContentConfig(
                task_type=embedding_task_type.upper(),
                output_dimensionality=embedding_dimensions,
            ),
        )

        typer.echo("Filtering descriptions using embeddings...")

        descriptions = embedding_filter(
            client=client,
            descriptions=descriptions,
            batch_size=embedding_batch_size,
            dataset_size=embedding_dataset_size,
            n_clusters=embedding_clusters,
        )

        typer.echo(f"Final dataset size: {embedding_dataset_size}")

    # setup dataset path
    stem = f"-{len(descriptions)}"
    if filter_language:
        stem = f"-{language}{stem}"
    dataset_path = dataset_path.with_stem(dataset_path.stem + stem)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Saving filtered descriptions to {dataset_path}...")

    # save the filtered descriptions to a JSON file
    json.dump(
        descriptions,
        dataset_path.open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    typer.echo("Processing complete.")
    file.close()


if __name__ == "__main__":
    typer.run(main)
