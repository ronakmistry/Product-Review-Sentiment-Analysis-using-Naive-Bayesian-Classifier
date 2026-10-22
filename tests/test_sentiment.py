import numpy as np
import pandas as pd
import pytest
from sentiment import fixture, prepare, split, build_pipeline, run


def test_split_before_vectorization():
    frame, _ = prepare(fixture())
    train, test = split(frame)
    assert not set(train).intersection(test)
    frame.loc[test, 'body'] += ' holdoutonlytoken'
    model = build_pipeline().fit(frame.body.iloc[train], frame.label.iloc[train])
    assert 'holdoutonlytoken' not in model['tfidf'].vocabulary_
    assert np.array_equal(test, split(frame)[1])


def test_duplicates_invalid_labels_and_empty_text():
    frame = fixture()
    cleaned, _ = prepare(pd.concat([frame, frame.iloc[:2]]))
    assert len(cleaned) == len(frame)
    contradictory = pd.concat([frame, pd.DataFrame([{'body': frame.body.iloc[0], 'label': 0}])])
    with pytest.raises(ValueError, match='contradictory'):
        prepare(contradictory)
    with pytest.raises(ValueError, match='body column'):
        prepare(pd.DataFrame({'text': ['hello']}))
    with pytest.raises(ValueError, match='at least 10'):
        prepare(pd.DataFrame({'body': ['  ', None], 'label': [0, 1]}))
    frame.loc[0, 'label'] = 2
    with pytest.raises(ValueError, match='only 0 or 1'):
        prepare(frame)


def test_pseudo_labels_are_explained():
    frame, source = prepare(fixture().drop(columns='label'))
    assert 'pseudo-labels' in source
    assert set(frame.label) == {0, 1}


def test_smoke_round_trip_and_real_missing(tmp_path):
    result = run(output=tmp_path)
    assert 'fixture-only' in result['status']
    assert set(result['metrics']) == {'accuracy', 'roc_auc', 'majority_baseline_accuracy'}
    assert sum(map(sum, result['confusion_matrix'])) == result['split_sizes']['test']
    with pytest.raises(FileNotFoundError, match='No scraping'):
        run(smoke=False, output=tmp_path)
