# one-sentence-summaries

A subset of book descriptions from https://openlibrary.org used to prompt a language model to generate short stories.

Note: summaries are short, not necessarily one sentence.
https://www.unco.edu/center-enhancement-teaching-learning/pdf/assessment/CAT-KIT.pdf

## Dataset

```
data/one-sentence-summaries.json
```

## Setup

install the protobuf compiler (for cld3 language detection)

```
sudo apt install -y protobuf-compiler
```

Install the project dependencies

```
git clone https://github.com/ptmorris05/one-sentence-summaries.git
cd one-sentence-summaries
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
```


## Generate the dataset

get the latest works data dump from https://openlibrary.org/developers/dumps

```shell
wget https://openlibrary.org/data/ol_dump_works_latest.txt.gz
```

Run the script

```
python3 open_library.py PATH_TO_DATA_DUMP.txt.gz
```
