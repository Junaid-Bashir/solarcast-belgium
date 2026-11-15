"""Small exception vocabulary for ingestion failures."""


class EliaAPIError(RuntimeError):
    """An Elia request failed or returned an invalid response envelope."""


class EliaSchemaError(ValueError):
    """An Elia record does not match the documented source contract."""


class PVDataValidationError(ValueError):
    """Normalized observations violate the PV data contract."""
