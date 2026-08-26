# Model artifacts

The trained Indian financial Word2Vec model will be stored under:

```text
artifacts/indian_financial/
|-- word2vec.model
|-- vectors.kv
`-- training_report.json
```

`word2vec.model` contains the complete trainable model. `vectors.kv` contains
the smaller query-only embeddings used by the application. The JSON report
records the corpus fingerprint, configuration, training statistics, and
technical validation checks.

Generated artifacts are ignored by Git. They should only be published after
corpus provenance, model quality, and file size have been reviewed.
