"""Offline review analysis; live scraping is preserved only as historical source."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


def fixture():
    rows = []
    for i in range(20):
        rows.append({'body': f'I love this wonderful useful product example{i}', 'label': 1})
        rows.append({'body': f'I hate this broken disappointing product example{i}', 'label': 0})
    return pd.DataFrame(rows)


def prepare(frame):
    if 'body' not in frame:
        raise ValueError('CSV must contain a body column with review text; optional label column must contain 0 or 1')
    frame = frame.copy().dropna(subset=['body'])
    frame['body'] = frame.body.astype(str).str.strip()
    frame = frame[frame.body != ''].copy()
    if 'label' in frame:
        source = 'provided binary labels; independence must be verified by the data owner'
        if not frame.label.isin([0, 1]).all():
            raise ValueError('label must contain only 0 or 1 with no missing values')
    else:
        from textblob import TextBlob
        # Match the coursework's rule: neutral (zero polarity) is positive.
        frame['label'] = [int(TextBlob(text).sentiment.polarity >= 0) for text in frame.body]
        source = 'TextBlob pseudo-labels; zero polarity is positive; metrics measure agreement with this rule'
    if (frame.groupby('body').label.nunique() > 1).any():
        raise ValueError('Identical review text has contradictory labels')
    frame = frame.drop_duplicates('body').reset_index(drop=True)
    counts = frame.label.value_counts()
    if len(frame) < 10 or len(counts) != 2 or counts.min() < 2:
        raise ValueError('Need at least 10 distinct reviews and at least 2 examples of each binary class')
    return frame, source


def split(frame, seed=42):
    # Deduplication occurs before splitting; TF-IDF is fitted only after splitting.
    return train_test_split(np.arange(len(frame)), test_size=.2, random_state=seed, stratify=frame.label)


def build_pipeline():
    return Pipeline([('tfidf', TfidfVectorizer(lowercase=True, strip_accents='ascii', stop_words='english')),
                     ('classifier', MultinomialNB())])


def run(smoke=True, data=None, output='artifacts/smoke', seed=42):
    if not smoke and (not data or not Path(data).is_file()):
        raise FileNotFoundError('Supply --data with the recovered review CSV. No scraping or dataset download is performed.')
    frame, label_source = prepare(fixture() if smoke else pd.read_csv(data))
    train, test = split(frame, seed)
    model = build_pipeline()
    model.fit(frame.body.iloc[train], frame.label.iloc[train])
    probabilities = model.predict_proba(frame.body.iloc[test])[:, 1]
    predictions = model.predict(frame.body.iloc[test])
    actual = frame.label.iloc[test]
    majority = int(frame.label.iloc[train].mode().iloc[0])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / 'pipeline.joblib')
    restored = joblib.load(output / 'pipeline.joblib')
    np.testing.assert_allclose(probabilities, restored.predict_proba(frame.body.iloc[test])[:, 1])
    report = {'status': 'fixture-only; not research evidence' if smoke else 'real-data rerun',
              'label_source': 'manually assigned synthetic fixture labels' if smoke else label_source,
              'configuration': {'seed': seed, 'test_fraction': .2, 'stop_words': 'scikit-learn english',
                                'model': 'TF-IDF + MultinomialNB(alpha=1.0)'},
              'dataset_sha256': None if smoke else hashlib.sha256(Path(data).read_bytes()).hexdigest(),
              'split_indices': {'train': train.tolist(), 'test': test.tolist()},
              'split_sizes': {'train': len(train), 'test': len(test)},
              'metrics': {'accuracy': float(accuracy_score(actual, predictions)),
                          'roc_auc': float(roc_auc_score(actual, probabilities)),
                          'majority_baseline_accuracy': float(accuracy_score(actual, np.full(len(test), majority)))},
              'confusion_matrix': confusion_matrix(actual, predictions, labels=[0, 1]).tolist(),
              'classification_report': classification_report(actual, predictions, labels=[0, 1], output_dict=True, zero_division=0),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'metadata.json').write_text(json.dumps({'label_source': report['label_source'],
        'configuration': report['configuration'], 'environment': report['environment']}, indent=2) + '\n')
    pd.DataFrame({'body': frame.body.iloc[test], 'actual': actual, 'predicted': predictions,
                  'positive_probability': probabilities}).to_csv(output / 'predictions.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.asarray(report['confusion_matrix'])
    ax.imshow(matrix, cmap='Blues')
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicted', ylabel='Actual',
           title='Synthetic fixture: not model quality' if smoke else 'Held-out label agreement')
    fig.tight_layout(); fig.savefig(output / 'confusion.png'); plt.close(fig)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'train', 'predict'])
    parser.add_argument('--data')
    parser.add_argument('--output', default='artifacts/smoke')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model')
    parser.add_argument('--text')
    args = parser.parse_args()
    try:
        if args.command == 'predict':
            if not args.model or not args.text:
                parser.error('predict requires --model and --text')
            pipeline = joblib.load(args.model)
            print(json.dumps({'label': int(pipeline.predict([args.text])[0]),
                              'positive_probability': float(pipeline.predict_proba([args.text])[0, 1])}))
        else:
            print(json.dumps(run(args.command == 'smoke', args.data, args.output, args.seed), indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'{error}\n')
