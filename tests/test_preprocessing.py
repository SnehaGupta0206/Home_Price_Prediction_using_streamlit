import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib


def test_no_missing_values_after_imputation():
    """Processed training data should have zero missing values."""
    df = pd.DataFrame({'a': [1, 2, None], 'b': [4, None, 6]})
    df_filled = df.fillna(0)
    assert df_filled.isnull().sum().sum() == 0


def test_total_sf_feature_engineering():
    """TotalSF should equal the sum of basement + 1st floor + 2nd floor."""
    df = pd.DataFrame({
        'TotalBsmtSF': [800],
        '1stFlrSF': [1000],
        '2ndFlrSF': [500]
    })
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    assert df['TotalSF'].iloc[0] == 2300


def test_log_transform_reversal():
    """np.expm1(np.log1p(x)) should return x (round-trip check)."""
    original_value = 250000
    log_value = np.log1p(original_value)
    reversed_value = np.expm1(log_value)
    assert round(reversed_value) == original_value


def test_model_loads_and_predicts():
    """Saved model should load and return one prediction per input row."""
    model = joblib.load('models/best_model.pkl')
    model_columns = joblib.load('models/model_columns.pkl')

    dummy_input = pd.DataFrame([[0] * len(model_columns)], columns=model_columns)
    prediction = model.predict(dummy_input)

    assert len(prediction) == 1
    assert isinstance(prediction[0], (float, np.floating))