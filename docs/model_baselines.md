# Domain Word2Vec Model Baselines

This document preserves the measurable baseline for each locally trained model.
Generated model files and source annual reports remain local and are excluded
from Git. Metrics recorded here allow later models to be compared against the
same published reference point.

## Version 2

Version 2 is the baseline trained before expanding the annual-report corpus.
Its local artifacts are stored in `artifacts/indian_financial/version_2/`.

### Training corpus

| Measurement | Value |
|---|---:|
| Validated documents | 6 |
| Pages processed | 2,817 |
| Phrase-aware sentences | 48,818 |
| Phrase-aware tokens | 1,467,039 |
| Raw vocabulary | 23,203 |
| Training-corpus SHA-256 | `be10c4e8722ebad8b33dc3dcbfb73c0b68bf370d78476d538a7815dd154b4750` |

The corpus hash identifies the exact phrase-aware training text. A future model
with a different hash was trained from different corpus contents.

### Training configuration

| Setting | Value |
|---|---:|
| Architecture | Skip-gram |
| Embedding dimensions | 50 |
| Context window | 3 |
| Negative samples | 5 |
| Minimum word frequency | 5 |
| Epochs | 5 |
| Initial learning rate | 0.025 |
| Random seed | 42 |
| Workers | 1 |

Using one worker and a fixed random seed makes repeated training runs
reproducible on the same software stack.

### Trained model

| Measurement | Value |
|---|---:|
| Retained vocabulary | 9,744 |
| Discarded low-frequency vocabulary | 13,459 |
| Embedding matrix shape | 9,744 x 50 |
| Technical validation | Passed |
| Finite vector values | Yes |
| Non-zero vectors | Yes |

### Target-term coverage

| Measurement | Result |
|---|---:|
| Overall | 121/142 (85.21%) |
| Single terms | 36/40 (90.00%) |
| Phrase terms | 85/102 (83.33%) |
| Core priority | 110/124 (88.71%) |
| Sector priority | 8/13 (61.54%) |
| Supplemental priority | 3/5 (60.00%) |

The weakest evaluated categories were banking at 61.54%, cash flow and
liquidity at 63.64%, and corporate actions at 68.75%. These gaps guide the
selection of additional reports for Version 3.

### Semantic benchmark

The version-controlled benchmark contains 30 related financial pairs and 30
unrelated control pairs. All 60 pairs were available in the Version 2
vocabulary.

| Measurement | Result |
|---|---:|
| Evaluated pairs | 60/60 |
| Skipped pairs | 0 |
| Related-pair average similarity | 0.7555 |
| Unrelated-pair average similarity | 0.3380 |
| Average separation margin | 0.4175 |
| Pairwise separation accuracy | 97.78% |
| Suspicious tokens detected | 0 |
| Evaluation warnings | 0 |

Pairwise separation measures how often a related pair scores above an
unrelated pair. It is a diagnostic of contextual separation on this curated
benchmark, not a claim of 97.78% accuracy on every financial-language task.

### Software versions

| Dependency | Version |
|---|---:|
| Gensim | 4.4.0 |
| NumPy | 2.5.1 |

### Local preserved artifacts

- `word2vec.model`: complete Gensim training model
- `vectors.kv`: vectors used for similarity and inference
- `training_report.json`: configuration, corpus hash, and training validation
- `evaluation_report.json`: coverage, semantic benchmark, and neighbour checks

These artifacts are ignored by Git because trained vocabulary can retain
company-specific tokens from the private local corpus.

## Version 3 acceptance conditions

These conditions were defined before collecting the expanded training corpus.
The Version 3 model must be evaluated with the same target taxonomy and frozen
60-pair semantic benchmark used for Version 2. Benchmark pairs must not be
changed in response to Version 3 scores.

### Critical gates

Version 3 is rejected if any critical gate fails.

| Gate | Required result |
|---|---:|
| Training completes without an exception | Yes |
| Vocabulary is non-empty | Yes |
| Embedding matrix has 50 dimensions | Yes |
| All vector values are finite | Yes |
| All retained vectors are non-zero | Yes |
| Fixed random seed is recorded | `42` |
| Training uses one worker | Yes |
| Training corpus SHA-256 is recorded | Yes |
| Evaluation warnings | 0 |
| Suspicious tokens detected | 0 |
| Frozen benchmark pairs evaluated | 60/60 |

### Coverage targets

Version 3 must improve coverage of important financial terminology, especially
the categories that were weak in Version 2.

| Measurement | Version 2 | Version 3 target |
|---|---:|---:|
| Overall target-term coverage | 85.21% | At least 90% |
| Core-priority coverage | 88.71% | At least 92% |
| Phrase-term coverage | 83.33% | At least 88% |
| Banking coverage | 61.54% | At least 75% |
| Cash-flow and liquidity coverage | 63.64% | At least 75% |
| Corporate-actions coverage | 68.75% | At least 75% |

Increasing vocabulary size alone does not satisfy these targets. A term counts
as covered only when its expected model token is present in the trained
vocabulary.

### Semantic benchmark safeguards

The larger corpus may make the embedding space broader, so Version 3 does not
need to beat every Version 2 similarity value. It must still preserve clear
separation between related and unrelated concepts.

| Measurement | Version 2 | Minimum Version 3 result |
|---|---:|---:|
| Related-pair average | 0.7555 | Greater than unrelated average |
| Unrelated-pair average | 0.3380 | Lower than related average |
| Average separation margin | 0.4175 | At least 0.38 |
| Pairwise separation accuracy | 97.78% | At least 95% |

A model that meets the coverage targets but falls below either semantic
threshold is not accepted automatically. Its corpus and neighbours must be
investigated and the Version 2 model remains the release candidate.

### Corpus-quality safeguards

Before training, the expanded corpus must satisfy all of the following:

- Every added report has passed PDF and extraction validation.
- No failed or duplicate document is included.
- Every report has its sector, financial year, and official source recorded in
  the private local manifest.
- Reports may span multiple financial years. Each report must retain its true
  year in both the manifest and filename.
- A report is not rejected only because its financial year differs from other
  reports in the corpus.
- Additional reports from a company already represented in the corpus require
  a duplicate-content and company-balance review before approval.
- No single document supplies more than 20% of all corpus tokens.
- The preprocessing and phrase-detection stages complete without warnings that
  invalidate the corpus.
- The generated corpus, phrase report, and quality report are preserved with
  the Version 3 artifacts.

The 20% document-share limit reduces the risk that one unusually long annual
report dominates the learned vocabulary and contexts.

### Manual neighbour review

Automated scores are necessary but do not fully measure semantic quality. The
nearest eight neighbours of these ten anchor terms must be manually reviewed:

1. `revenue`
2. `profit`
3. `net_profit`
4. `operating_profit`
5. `working_capital`
6. `liquidity`
7. `credit_risk`
8. `fair_value`
9. `dividend`
10. `cash_and_cash_equivalents`

At least eight of the ten anchor terms must have acceptable neighbour lists.
For an anchor to pass, at least five of its eight nearest neighbours must be
financially or contextually relevant on human review. Broken PDF fragments or
meaningless extraction artifacts cause that anchor to fail.

### Final selection rule

Version 3 becomes the release candidate only when:

1. every critical gate passes;
2. every coverage target is met;
3. both semantic benchmark safeguards are met;
4. every corpus-quality safeguard is met; and
5. the manual neighbour review passes.

If a condition fails, Version 2 is retained while the cause is investigated.
The benchmark must not be weakened and failed checks must not be hidden merely
to label Version 3 as an improvement.
