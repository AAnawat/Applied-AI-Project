# Topic categorization

For the isolated Typhoon QLoRA experiment and Colab notebook, see
[finetuning/README.md](finetuning/README.md). The web application is unchanged.

Assigns each Thai feedback response exactly one topic code, using a local model
served by [Ollama](https://ollama.com). The codes match `categories.code` in the database:

`welfare`, `compensation`, `management`, `workload`, `environment`, `career`, `communication`, `other`

```
categorization/
├── README.md
├── cmd/                command-line entry points only
│   ├── predict.py      categorize a feedback CSV
│   └── evaluate.py     score categorizers on an exam against its answer key
├── core/               logic shared by the commands
│   ├── taxonomy.py     category codes, names, Thai descriptions, tie-breaking rules
│   ├── examples.py     synthetic few-shot examples (one per category)
│   ├── llm.py          prompt building, Ollama client, JSON parsing (also used by the dashboard)
│   ├── feedback.py     rejects missing or blank feedback
│   ├── exam.py         loads an exam + answer key and pairs rows by id
│   ├── scoring.py      accuracy, F1, invalid rate, latency
│   └── paths.py        data/ and results/ locations, new run folders
├── data/
│   ├── exams/          <name>.csv: id, feedback   (what the model sees)
│   └── answer_keys/    <name>.csv: id, category   (human-assigned codes)
└── results/            one folder per run, never overwritten
```

Run every command from the repository root.

## Setup

```bash
ollama pull hf.co/typhoon-ai/typhoon2.5-qwen3-4b-gguf:Q4_K_M
```

The default is the tested original Typhoon2.5-Qwen3-4B Q4_K_M model.
The dashboard uses zero-shot categorization through this same client. It stops before
saving if any topic answer is invalid; the CLI below leaves invalid categories blank.
Dashboard Ollama host/model/timeout can be set in Streamlit secrets (see the root README).

## Predict

```bash
python -m categorization.cmd.predict --input feedback.csv
```

Adds a `category` column to any CSV with a `feedback` column and writes it to
`results/<input>_<model>_<timestamp>/categorized.csv` (or `--output path.csv`).
The model must answer JSON restricted to the codes above (temperature 0, thinking off).
Anything else is left blank, so no `category_predictions` row is saved for it.
No category confidence is produced.

Pass the output to the sentiment command, which keeps `category` and saves both:

```bash
python -m sentiment.predict --input results/<run>/categorized.csv --dry-run
```

## Evaluate

An evaluation set is two files with the same name:

| File | Columns | Rules |
|---|---|---|
| `data/exams/<name>.csv` | `id`, `feedback` | unique ids, no blank or duplicate feedback |
| `data/answer_keys/<name>.csv` | `id`, `category` | unique ids, codes from the list above |

Rows are matched by `id`, so either file can be in any order. Every id must appear in both.

```bash
python -m categorization.cmd.evaluate --exam climatelens_105
python -m categorization.cmd.evaluate --exam climatelens_105 --model qwen3:4b
```

It compares zero-shot and few-shot prompting on the same rows
(`--methods` picks a subset, `--limit N` scores the first N rows) and writes to
`results/<exam>_<model>_<timestamp>/`:

- `predictions.csv`: id, feedback, answer, each method's prediction and the model's raw reply
- `confusion_<method>.csv`: answer × predicted counts
- `metrics.json`: accuracy, macro/weighted F1, invalid-output rate, latency,
  per-class precision/recall and Ollama's reported model memory

Invalid outputs count as wrong.

### Writing an exam

- `climatelens_105` is 105 rows from `climatelens_template.csv` (ids 1-105): 15 per category,
  spread across sentiments like the file, with at most one version of each base sentence.
  Its answer key comes from that file's `category` column (names mapped to codes); it has no `other` rows.
- The few-shot examples in `core/examples.py` are synthetic; evaluation stops if an exam contains any of them.
- Do not build exams from `dashboard/mock_data.py`: it repeats a few dozen template sentences,
  so it is not realistic feedback.
- Label the answer key without looking at model predictions.
