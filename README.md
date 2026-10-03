# Text Embedder

Text Embedder is an educational Streamlit application that creates and visualizes text embeddings from scratch. It uses statistical word relationships and linear algebra instead of a pretrained embedding model or external API.

Users provide a training corpus and one or more sentences. The application builds a temporary embedding space from that corpus, generates word and sentence vectors, and exposes each stage of the process in the UI.

## Features

- Train corpus-specific embeddings directly in the browser UI.
- Inspect cleaned tokens and the generated vocabulary.
- Explore weighted word co-occurrence and PPMI matrices.
- View complete word and sentence embedding vectors.
- Visualize individual embedding dimensions.
- Compare sentences using cosine similarity.
- Download generated embeddings as CSV files.
- Run without pretrained model weights, API keys, or external AI services.

## Embedding pipeline

```text
Training corpus
      |
      v
Tokenization and vocabulary
      |
      v
Weighted co-occurrence matrix
      |
      v
Positive Pointwise Mutual Information (PPMI)
      |
      v
Singular Value Decomposition (SVD)
      |
      v
Dense word embeddings
      |
      v
Averaged and normalized sentence embeddings
      |
      v
Cosine similarity
```

### Fixed training configuration

- Context window size: `3`
- Embedding dimensions: `5`

The fixed context window considers up to three tokens before and after each center token. Nearby words receive more weight than distant words. SVD retains five dimensions so the output remains easy to inspect and visualize.

The corpus must contain at least five unique words because the application produces five-dimensional embeddings.

## Project structure

```text
embeddings/
|-- app.py
|-- README.md
|-- requirements.txt
|-- embedding_engine/
|   |-- __init__.py
|   |-- text_processing.py
|   |-- cooccurrence.py
|   |-- ppmi.py
|   |-- generator.py
|   `-- similarity.py
|-- ui/
|   |-- __init__.py
|   `-- components.py
`-- tests/
    `-- test_engine.py
```

## Getting started

### 1. Clone the repository

```powershell
git clone https://github.com/YOUR_USERNAME/text-embedder.git
cd text-embedder
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate the environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Run the application

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Running tests

```powershell
python -m unittest discover -s tests -v
```

## Training the Version 3 financial model

After the private annual-report corpus has been built and validated with
`python -m training.version3_pipeline`, train the frozen phrase corpus with:

```powershell
python -m training.version3_trainer
```

The command verifies the corpus checksums before training and writes a separate
model package to `artifacts/indian_financial/version_3/`. It includes the model,
query vectors, training provenance, and copied corpus audit reports. It refuses
to overwrite an existing Version 3 package.

## How to use the application

1. Enter or edit the training corpus.
2. Enter one or more sentences, with one sentence per line.
3. Select **Generate embeddings**.
4. Explore the learning-process, word-embedding, sentence-embedding, and similarity tabs.
5. Optionally download the generated vectors as CSV files.

## Limitations

- The embeddings only learn relationships present in the supplied corpus.
- A small or repetitive corpus produces weak semantic relationships.
- Words absent from the corpus cannot receive trained word vectors.
- Sentence embeddings are averages, so they do not preserve word order.
- Each new corpus creates a different embedding space.
- Vectors produced from different corpora or training runs should not be compared directly.
- These statistical embeddings are not equivalent to embeddings from a large pretrained neural model.

## Planned enhancement

The next phase will retain the current custom-corpus laboratory and add a second mode backed by a domain-specific embedding model trained by this project. That mode will use saved vocabulary and vector artifacts, produce a stable embedding space, and accept sentences without requiring users to submit a corpus.

## Technology

- Python
- Streamlit
- NumPy
- pandas
- `unittest`


