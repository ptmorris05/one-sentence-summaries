import gzip
import json
from typing import Optional


def should_keep(
    text: str,
    min_words: int = 15,
    max_words: int = 50,
    max_punctuation: int = 5,
    punctuation: str = ";-_",
    exclude_words: Optional[list] = None,
) -> bool:
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


def main():
    with gzip.open("ol_dump_works_2025-05-31.txt.gz", "rt") as f:
        for line in f:
            edition = json.loads(line.split("\t")[-1])

            if "description" in edition:
                desc = edition["description"]
                if not isinstance(desc, str):
                    desc = desc["value"]

                if not should_keep(desc):
                    continue

                print(desc)


if __name__ == "__main__":
    main()
