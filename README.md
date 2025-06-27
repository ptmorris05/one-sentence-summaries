# one-sentence-summaries

A subset of book descriptions from https://openlibrary.org used to prompt a language model to generate short stories.

### Data

get the latest works data dump from https://openlibrary.org/developers/dumps

```shell
wget https://openlibrary.org/data/ol_dump_works_latest.txt.gz
```

# Usage

```
python3 open_library.py --open-library-path PATH_TO_DATA_DUMP.txt.gz
```
