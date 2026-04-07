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

## Clean the Excel (optional)

`clean_brand_task.py` applies simple ML-oriented cleaning to `GOOD_NAME` (whitespace, quotes, punctuation, keep Armenian + Latin + digits + `/` etc.) and normalizes `ADG_CODE` to integers. Optional columns **`BRAND`** and **`INDUSTRY`** are recognized case-insensitively (`brand`, `Industry`, …); if missing, empty columns are added. Output column order: `GOOD_NAME`, `BRAND`, `INDUSTRY`, `ADG_CODE`, then any other columns.

```bash
python clean_brand_task.py
python clean_brand_task.py --input brand_task.xlsx --output brand_task_cleaned.xlsx
python clean_brand_task.py --drop-invalid-code   # drop rows with bad codes
```

Train on the cleaned file: `python train_bilstm.py --data brand_task_cleaned.xlsx` (training still uses `GOOD_NAME` → `ADG_CODE` only; extra columns are kept in the file for analysis and for `predict.py --code`).

### Rich cleaning + brand/industry (`clean_data.py`)

Runs the dictionary-based brand extraction and industry rules; reads **`brand_task.xlsx`** (or **`brand_task.csv`** if the xlsx is missing) and writes **`cleaned_product_data_with_brands.csv`** with columns: `cleaned_product_name`, `adg_code`, `brand`, `industry`.

```bash
python clean_data.py
```

To use with `train_bilstm.py`, either rename columns to `GOOD_NAME` / `ADG_CODE` or point training at a small wrapper that reads this CSV.

This pipeline is separate from `text_language.normalize_good_name` used inside `train_bilstm.py` at load time — use one strategy consistently (either clean the sheet first and relax duplicate logic, or rely on in-loader normalization only).

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
