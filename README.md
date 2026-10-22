# Product-review sentiment analysis with Naive Bayes

A restored graduate project by **Ronak Mistry** exploring TF-IDF text features and a
Multinomial Naive Bayes classifier on product reviews.

## Method and corrections

The original notebook scraped reviews, labeled them using TextBlob polarity, then trained
a classifier to reproduce those labels. It fitted TF-IDF before splitting and described
ROC AUC as accuracy. The restoration separates those concepts:

1. Read a local CSV, remove empty reviews, and deduplicate identical trimmed text before splitting.
2. Use provided binary labels or explicitly generate TextBlob pseudo-labels (polarity >= 0 is positive).
3. Make a seeded, stratified 80/20 split. Fit TF-IDF and Naive Bayes **only on training reviews**.
4. Report accuracy, ROC AUC from positive-class probabilities, class-level metrics, a confusion
   matrix, and the training-majority baseline on held-out reviews.

The classifier still uses TF-IDF plus MultinomialNB with alpha=1.0. Scikit-learn's built-in
English stopword list replaces the old downloaded NLTK list; it is not identical and may
remove negation words. This is documented as a restoration difference, not a benchmark gain.
## Setup and verification

Use Python **3.12** in an isolated environment. CPU execution is the default.
Run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python scripts/execute_notebooks.py
```

On Windows, activate with `.venv\Scripts\activate`. The notebook runner starts a fresh
kernel for each supported root notebook and writes executed copies and HTML to
`artifacts/notebooks/`. Open the HTML locally to read the rendered walkthrough.
Root notebooks default to generated fixtures and perform no downloads.
Direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt`
records the complete tested environment. The lock targets CPython 3.12 on macOS
Apple Silicon and Linux; optional platform packages use environment markers.
CI installs the lock, runs fixture tests, executes notebooks, and uploads the rendered results.

## Evidence and preservation

**No real-data reproduction has been performed in this restoration.** Fixture metrics
validate code paths only. They do not establish model quality or recover historical results.
See [verification](docs/verification.md) for checks and limitations, and
[inventory](docs/inventory.json) for the starting commit, original files, and missing resources.
The original notebook sources and saved outputs are preserved in [historical/](historical/).
They use obsolete APIs and are not supported execution entry points.

Generated data, features, models, reports, and notebook outputs stay under ignored
`artifacts/` or `data/` directories. Review dataset permissions before adding any data to Git.

## Quick run

```bash
python sentiment.py smoke --output artifacts/smoke
```

Forty generated positive/negative template sentences with manually assigned synthetic labels
exercise fitting, held-out evaluation, plotting, and a saved-pipeline round trip. The fixture
is deliberately easy; its scores do not establish performance on real reviews.

## Supply the recovered reviews later

The original `Final_Amazon_Reviews.csv` is absent. Supply a CSV with:

- `body`: review text (required).
- `label`: optional binary 0/1 labels (0 negative, 1 positive). Without this column, the
  pipeline uses TextBlob polarity to generate pseudo-labels and records that fact prominently.

At least ten distinct nonempty reviews are required, with at least two examples per class.
Contradictory labels for identical text are rejected. Class coverage should be checked on
real data. Related or near-duplicate reviews can still leak information across an exact-text
split; group them before use if the recovered data includes such relationships.

```bash
python sentiment.py train --data data/Final_Amazon_Reviews.csv --output artifacts/reviews
python sentiment.py predict --model artifacts/reviews/pipeline.joblib --text "A useful product"
```

No live scraping is run. The original scraping code is preserved only in the historical
notebook. No NLTK corpus downloads are needed for this supported workflow.

Runs write `pipeline.joblib`, `metadata.json`, `results.json`, `predictions.csv`, and
`confusion.png`. Reports record the seed, split indices, data fingerprint, label source,
configuration and versions. Only load trusted locally generated joblib artifacts; the saved
pipeline should be used with the pinned environment. Label source is stored beside it.

## Historical results and limitations

The original notebook saves ROC AUC approximately **0.8998** and describes accuracy ranges
of 85–98%. These are **unverified historical claims**, not new measurements. The original
pre-split TF-IDF fitting and TextBlob-generated targets prevent treating that score as an
independent assessment of sentiment accuracy.

Metrics on pseudo-labels measure agreement with TextBlob, including its errors. Independent
human labels are needed before claiming real sentiment accuracy. Class imbalance and the
chosen neutral-as-positive rule also affect interpretation.

## References and attribution

Original source, saved outputs, and Git history preserve authorship; the existing
[MIT license](LICENSE) is retained.
References: [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html),
[TextBlob sentiment](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis).
