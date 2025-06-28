# one-sentence-summaries

A subset of book descriptions from https://openlibrary.org used to prompt a language model to generate short stories.

The dataset primarily contains english-language children's books and young adult (YA) fiction, based on the `open_library.py` script's default settings. 

Note: the summaries are short, but they may contain more than one sentence.
https://en.wikipedia.org/wiki/Log_line

## Download the dataset

https://github.com/ptmorris05/one-sentence-summaries/releases/download/v0.1.0/one-sentence-summaries-en-20000.json

```
wget https://github.com/ptmorris05/one-sentence-summaries/releases/download/v0.1.0/one-sentence-summaries-en-20000.json
```

## Generate the dataset (simple)

Install the project dependencies

```
git clone https://github.com/ptmorris05/one-sentence-summaries.git
cd one-sentence-summaries
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install .
```

get the latest works data dump from https://openlibrary.org/developers/dumps

```shell
wget https://openlibrary.org/data/ol_dump_works_latest.txt.gz
```

Run the script

```
python3 open_library.py PATH_TO_DATA_DUMP.txt.gz
```

## Generate the dataset (with language and embedding filtering)

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
python3 -m pip install .[embedding,language]
```

get the latest works data dump from https://openlibrary.org/developers/dumps

```shell
wget https://openlibrary.org/data/ol_dump_works_latest.txt.gz
```

Get a free embeddings API key by signing up at https://ai.google.dev/. The embeddings API is free (as of June 2025).

Set environment variables in a .env file.
```shell
echo "GEMINI_API_KEY=YOUR_API_KEY" > .env
```

Run the script with language and embedding flags

```
python3 open_library.py PATH_TO_DATA_DUMP.txt.gz --filter-language --filter-embedding
```

# Script arguments
```
# open_library.py

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    open_library_path      FILE  Path to the gzipped Open Library works dump file [default: None] [required]                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --exclude-words                                            FILE                       Path to the file containing words to exclude [default: data/exclude_words.txt]     │
│ --dataset-path                                             FILE                       Path to save the filtered descriptions [default: data/one-sentence-summaries.json] │
│ --min-words                                                INTEGER RANGE [x>=1]       Minimum number of words required [default: 15]                                     │
│ --max-words                                                INTEGER RANGE [x>=1]       Maximum number of words allowed [default: 50]                                      │
│ --max-punctuation                                          INTEGER RANGE [x>=0]       Maximum punctuation characters allowed [default: 5]                                │
│ --punctuation                                              TEXT                       Punctuation characters to check against [default: ;-_]                             │
│ --separator                                                TEXT                       Separator used in the Open Library dump file [default:                             │
│                                                                                                                                    ]                                     │
│ --description-key                                          TEXT                       Key in the JSON object that contains the description [default: description]        │
│ --value-key                                                TEXT                       Key in the JSON object for text values [default: value]                            │
│ --filter-language                 --no-filter-language                                Filter descriptions by language using Cld3 [default: no-filter-language]           │
│ --language                                                 TEXT                       Cld3 language code for the language to filter by [default: en]                     │
│ --cld3-max-num-bytes                                       INTEGER RANGE [x>=1]       Maximum number of bytes for language detection [default: 1000]                     │
│ --cld3-min-language-proportion                             FLOAT RANGE [0.0<=x<=1.0]  Minimum proportion of same-language text required [default: 0.99]                  │
│ --filter-embedding                --no-filter-embedding                               Filter descriptions using embeddings [default: no-filter-embedding]                │
│ --embedding-model                                          TEXT                       Gemini Embedding model to use for filtering [default: text-embedding-004]          │
│ --embedding-batch-size                                     INTEGER RANGE [1<=x<=100]  Batch size for processing embeddings [default: 100]                                │
│ --embedding-dimensions                                     INTEGER RANGE [1<=x<=768]  Dimensions of the embedding model [default: 256]                                   │
│ --embedding-task-type                                      TEXT                       Gemini task type for embeddings, e.g., 'CLUSTERING' [default: CLUSTERING]          │
│ --embedding-dataset-size                                   INTEGER RANGE [x>=1]       Size of the final dataset after embedding filtering [default: 20000]               │
│ --embedding-clusters                                       INTEGER RANGE [x>=1]       Number of clusters for KMeans clustering [default: 100]                            │
│ --help                                                                                Show this message and exit.                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```