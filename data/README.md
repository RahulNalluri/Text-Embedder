# Training data

This directory holds metadata and locally generated data for the Indian corporate financial-reporting embedding model.

## Layout

```text
data/
|-- metadata/
|   |-- companies.example.csv
|   |-- companies.csv (local and ignored by Git)
|   |-- evaluation_pairs.csv
|   `-- target_terms.csv
|-- raw/
|   `-- annual_reports/ (local and ignored by Git)
|       `-- version_3/
|           |-- candidates/
|           |-- approved/
|           `-- rejected/
`-- processed/
```

Copy `companies.example.csv` to `companies.csv`, then replace the fictional row
with the companies selected for your local training corpus. The real manifest is
ignored by Git so company selections and report URLs are not published.

## Collection policy

- Use annual reports from official company investor-relations pages.
- The training corpus may include multiple financial years. Record each
  report's true year in the manifest and filename; do not relabel a report to
  force it into a single-year batch.
- Record the exact source URL in your local `metadata/companies.csv`.
- Review each source's terms before downloading or processing its report.
- Do not automate collection from NSE; its terms prohibit systematic automated collection.
- Keep complete annual-report PDFs local unless redistribution permission is confirmed.
- Do not commit extracted full-text corpora until their redistribution status is reviewed.

## Version 3 report staging

New reports must be downloaded into `raw/annual_reports/version_3/candidates/`.
A downloaded PDF is not training data until its source, identity, completeness,
and extraction quality have been checked.

Use the three staging folders as follows:

| Folder | Purpose | May be used for training? |
|---|---|---:|
| `candidates/` | Newly downloaded reports awaiting validation | No |
| `approved/` | Complete reports that passed source and extraction checks | Yes |
| `rejected/` | Invalid, duplicate, incomplete, or low-quality reports retained for review | No |

The existing Version 2 PDFs remain in their current location and must not be
moved or renamed while the Version 2 baseline is being preserved.

### Candidate-to-approved workflow

1. Download one complete English annual or integrated report from an official
   company investor-relations page.
2. Save it in `candidates/` using the format
   `company_name_fyYYYY_YY.pdf`.
3. Add its company, sector, financial year, source URL, and local filename to
   the private `metadata/companies.csv` manifest.
4. Confirm that the file opens, is a PDF, is not password-protected, and is not
   a duplicate of an existing report.
5. Run the project's PDF extraction check and review its page counts and
   possible scanned pages.
6. Confirm that the report contains substantive financial statements, notes,
   governance information, and management discussion.
7. Move a passing report to `approved/` and set its manifest status to
   `approved`.
8. Move a failing report to `rejected/` and record the rejection reason in the
   manifest. Do not silently delete it or include it in training.

The corpus builder accepts explicit `--pdf` arguments. Only paths inside the
`approved/` folder should be supplied when building the Version 3 corpus.

Version 3 is a multi-company, multi-sector, multi-year corpus. A different
financial year is not, by itself, a rejection reason. When multiple reports
from the same company are available, approve only the reports that add useful
coverage after checking repeated boilerplate and company-level corpus balance.

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
