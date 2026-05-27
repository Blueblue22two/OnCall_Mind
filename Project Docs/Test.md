0. 数据集验证:
```bash
.venv/bin/python -m tests.evaluation.validate_dataset
`````

1. Basic test run:
```bash
RAG_MODE=basic .venv/bin/python -m tests.evaluation.evaluate_rag \
  --output reports/rag_basic_baseline_v1.1.2.json \
  --output-format both
```

2. Enhanced test run:
```bash
RAG_MODE=enhanced QUERY_PREPROCESSOR_TYPE=none RERANKER_TYPE=cross_encoder \
.venv/bin/python -m tests.evaluation.evaluate_rag \
  --output reports/rag_enhanced_baseline_v1.1.2.json \
  --output-format both
```

3. generate report:
```bash
.venv/bin/python -m tests.evaluation.compare_reports \
  --basic reports/basic.json \
  --enhanced reports/enhanced.json \
  --output reports/comparison.json
```