from pathlib import Path

import pandas as pd
import pytest

from src.credit_risk import MODEL_FEATURES, build_models, load_data, temporal_split


DATA_PATH = Path("data/raw/verizon_data.csv")


def test_source_data_and_temporal_split():
    frame = load_data(DATA_PATH)
    split = temporal_split(frame)
    assert len(frame) == 12_000
    assert split.train["application_date"].max() < split.cutoff
    assert split.test["application_date"].min() >= split.cutoff
    assert frame["target"].isin([0, 1]).all()
    assert frame["dp_ratio"].between(0, 1).all()


def test_logistic_pipeline_produces_probabilities():
    frame = load_data(DATA_PATH)
    split = temporal_split(frame)
    model = build_models()["logistic"]
    model.fit(split.train[MODEL_FEATURES], split.train["target"])
    probability = model.predict_proba(split.test[MODEL_FEATURES].head(20))[:, 1]
    assert len(probability) == 20
    assert ((probability >= 0) & (probability <= 1)).all()


def test_missing_source_column_fails(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"fico": [700]}).to_csv(bad_path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_data(bad_path)
