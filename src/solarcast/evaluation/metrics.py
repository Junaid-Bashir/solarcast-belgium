"""Deterministic MW metrics on explicit, complete finite pairs."""

import math

import numpy as np
import numpy.typing as npt
import pandas as pd


def _vector(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    if isinstance(values, pd.Series):
        if (
            not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)
            or pd.api.types.is_complex_dtype(values.dtype)
        ):
            raise ValueError("Metric inputs must be real numeric vectors")
        vector = values.to_numpy(dtype=float, na_value=np.nan)
    else:
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Metric inputs must be real numeric vectors")
        vector = np.asarray(raw, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Metric inputs must be nonempty finite one-dimensional vectors")
    return vector


def _errors(actual: npt.ArrayLike, prediction: npt.ArrayLike) -> npt.NDArray[np.float64]:
    actual_values, predictions = _vector(actual), _vector(prediction)
    if actual_values.shape != predictions.shape:
        raise ValueError("actual and prediction must have identical lengths")
    with np.errstate(over="raise", invalid="raise"):
        try:
            return predictions - actual_values
        except FloatingPointError as error:
            raise ValueError("Prediction-minus-actual overflow") from error


def mae(actual: npt.ArrayLike, prediction: npt.ArrayLike) -> float:
    """Mean absolute prediction error in MW; caller must select complete finite pairs."""
    errors = np.abs(_errors(actual, prediction))
    scale = float(errors.max())
    return 0.0 if scale == 0 else float(np.mean(errors / scale)) * scale


def rmse(actual: npt.ArrayLike, prediction: npt.ArrayLike) -> float:
    """Root mean squared prediction error in MW, with scaled squaring to avoid overflow."""
    errors = _errors(actual, prediction)
    scale = float(np.abs(errors).max())
    return 0.0 if scale == 0 else float(np.sqrt(np.mean((errors / scale) ** 2))) * scale


def bias(actual: npt.ArrayLike, prediction: npt.ArrayLike) -> float:
    """Mean prediction minus actual in MW: positive means overforecasting."""
    errors = _errors(actual, prediction)
    scale = float(np.abs(errors).max())
    return 0.0 if scale == 0 else float(np.mean(errors / scale)) * scale


def rmse_skill(
    actual: npt.ArrayLike, prediction: npt.ArrayLike, reference_prediction: npt.ArrayLike
) -> float:
    """1 - RMSE(model)/RMSE(reference) on identical pairs; NaN if reference RMSE is zero."""
    model_error = rmse(actual, prediction)
    reference_error = rmse(actual, reference_prediction)
    return math.nan if reference_error == 0 else 1.0 - model_error / reference_error
