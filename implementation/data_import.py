import numpy as np
import pandas as pd


def csv_import(targets: list[str, ...], path: str, dtype: np.dtype, remove_dup: bool = False) -> tuple[list[str, ...], np.array]:
    """
    Function reading the data CSV, returning a NumPy array with the relevant classes encoded.
    :param targets: A list which must contain the letters considered as target.
    :param path: The path of the CSV file to be read, containing the letters features.
    :param dtype: The required NumPy data type for the output array.
    :param remove_dup: Boolean to check whether to eliminate the duplicate rows.
    :return: A tuple with two elements. The first is a list of the targets indexed by their encoding.
     The second is a NumPy array with the standardized features and the relevant targets encoded in the last column.
    """
    out_array = pd.read_csv(path)  # Read CSV
    out_array = out_array.loc[out_array['Y'].isin(targets)]  # Filter out observations related to rel. variables
    ord_targets, out_array.iloc[:, 0] = np.unique(out_array.iloc[:, 0], return_inverse=True)  # Encode target
    if remove_dup:
        out_array.drop_duplicates(inplace=True)  # drop duplicates (there are some)
    out_array = out_array.to_numpy(dtype=dtype)
    out_array[:, [0, -1]] = out_array[:, [-1, 0]]  # Swap columns (target last)

    out_array[:, :-1] = (out_array[:, :-1]-np.mean(out_array[:, :-1], axis=0))/np.std(out_array[:, :-1])  # standardize

    return list(ord_targets), out_array

