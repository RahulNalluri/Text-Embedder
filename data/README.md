# Training data

This directory holds metadata and locally generated data for the Indian corporate financial-reporting embedding model.

## Layout

```text
data/
|-- metadata/
|   |-- companies.example.csv
|   |-- companies.csv (local and ignored by Git)
|   `-- target_terms.csv
|-- raw/
|   `-- annual_reports/
`-- processed/
```

Copy `companies.example.csv` to `companies.csv`, then replace the fictional row
with the companies selected for your local training corpus. The real manifest is
ignored by Git so company selections and report URLs are not published.

## Collection policy

- Use annual reports from official company investor-relations pages.
- Record the exact source URL in your local `metadata/companies.csv`.
- Review each source's terms before downloading or processing its report.
- Do not automate collection from NSE; its terms prohibit systematic automated collection.
- Keep complete annual-report PDFs local unless redistribution permission is confirmed.
- Do not commit extracted full-text corpora until their redistribution status is reviewed.

## Pilot design

The first pilot targets one company from each of six sectors and two completed financial years per company. Proposed companies are not approved sources until their investor-relations URL and reuse status have been verified.

## Generated files

The processing and phrase-detection stages produce:

```text
data/processed/sentences.txt
data/processed/vocabulary_counts.csv
data/processed/corpus_report.json
data/processed/quality_report.json
data/processed/sentences_phrased.txt
data/processed/phrase_counts.csv
data/processed/phrase_report.json
```

These generated files should be reproducible from the source manifest and processing scripts.
