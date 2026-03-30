"""Parser utilities for Qualibrate instrument data."""

import datetime

import xarray as xr


def repetition_data(
    ds: xr.Dataset, repetition_dim: str = "qubit"
) -> list[xr.Dataset]:
    """
    Split a Dataset along a repetition dimension into a list of Datasets.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the repetition dimension.
    repetition_dim : str
        Name of the dimension to split along (default: "qubit").

    Returns
    -------
    list[xr.Dataset]
        One Dataset per index along *repetition_dim*.
    """
    n_qubits = ds.sizes[repetition_dim]
    output_data = []
    for qubit_idx in range(n_qubits):
        data = ds.isel(**{repetition_dim: qubit_idx})
        output_data.append(data)
    return output_data


def parse_timestamp(ts: str) -> datetime.datetime:
    """
    Parse an ISO-8601 timestamp string, ignoring timezone offset.

    Parameters
    ----------
    ts : str
        Timestamp string, e.g. "2026-03-30T09:52:18.123456+08:00".

    Returns
    -------
    datetime.datetime
        Parsed datetime (timezone-naive).
    """
    if "+" in ts:
        ts = ts.split("+")[0]
    return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
