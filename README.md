# brad_detection_arm

Train a character-level BiLSTM to map product titles (`GOOD_NAME`) to category codes (`ADG_CODE`) from an Excel sheet. Text is Armenian and Latin mixed. Run inference from the command line or look up example titles for a code straight from the spreadsheet.

## What you need

- Python 3.9 or newer
- `brand_task.xlsx` with columns `GOOD_NAME` and `ADG_CODE`
- After training: `bilstm_adg.pt` in the project folder (git ignores `*.pt`; generate locally)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

NumPy is pinned below 2.x for compatibility with common PyTorch wheels on older setups.

## Train

Default is a character BiLSTM on CPU or GPU if PyTorch sees CUDA.

```bash
python train_bilstm.py --early-stopping 10
```

Useful flags: `--epochs`, `--out` for another checkpoint path, `--best-metric` (default `top3`), `--focal-gamma` for imbalanced classes. For word vectors instead of characters, pass one or more `--fasttext-model` paths to `.bin` files (see `train_bilstm.py --help`).

Training writes `bilstm_adg.pt`, plus `.history.json` / `.meta.json` next to it (those patterns are gitignored).

## Predict (title → code)

```bash
python predict.py --text "product name here"
python predict.py --text "another" --topk 5
```

Requires `bilstm_adg.pt`.

## Look up titles for a code (Excel only, no model)

Reads `brand_task.xlsx` and prints rows for that `ADG_CODE`:

```bash
python predict.py --code 2202
python predict.py -c 401
```

Optional: `--max-examples`, `--keywords` (comma-separated substrings), `--data` for another xlsx path. Duplicate-looking lines in a code usually mean wrong labels in the sheet; filtering is your responsibility.

## Text preprocessing

Product strings are normalized before the model sees them: NFC, strip Excel junk (`_x000D_`, slashes, asterisks), lowercase, spacing, and a small brand alias map (e.g. `coca cola` / `կոկա կոլա` → `coca_cola`). Training and `predict.py` use the same function (`text_language.normalize_good_name`).

## Evaluation

With a checkpoint and `brand_task.xlsx`:

```bash
python visualize.py --out-dir figures
```

Produces validation metrics and plots under `figures/` (that folder is gitignored).

## Layout

| File | Role |
|------|------|
| `train_bilstm.py` | Training loop, checkpoint |
| `predict.py` | Inference and `--code` lookup |
| `text_language.py` | Normalization and brand aliases |
| `fasttext_embeddings.py` | Word + FastText path only |
| `visualize.py` | Metrics and confusion-style plots |

## License

Add one if this repo goes public.
