import os
import pandas as pd
import glob


def list_files_with_exts(filepath_pattern, ext_list, recursive=True):
    if isinstance(filepath_pattern, list):
        matches = []
        for fp in filepath_pattern:
            matches += list_files_with_exts(fp, ext_list, recursive=recursive)
        return sorted(set(matches))

    else:
        # make sure extensions all start with "." and are lowercase
        ext_list = ["." + ext.strip(".").lower() for ext in ext_list]

        if os.path.isdir(filepath_pattern):
            filepath_pattern = os.path.join(filepath_pattern, "*")

        # find all matches (recursively)
        matches = glob.glob(filepath_pattern)
        if recursive:
            for match in list(matches):
                matches += glob.glob(os.path.join(match, "**"), recursive=True)

        # filter matches by extension
        matches = [
            match
            for match in matches
            if os.path.splitext(match)[1].lower() in ext_list
        ]
        return matches


def _deeplabcut_loader(filepath, name):
    """Load tracking results from deeplabcut csv or hdf5 files."""
    ext = os.path.splitext(filepath)[1]
    if ext == ".h5":
        df = pd.read_hdf(filepath)
    if ext == ".csv":
        df = pd.read_csv(filepath, header=[0, 1, 2], index_col=0)

    coordinates, confidences = {}, {}
    bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    if "individuals" in df.columns.names:
        for ind in df.columns.get_level_values("individuals").unique():
            ind_df = df.xs(ind, axis=1, level="individuals")
            arr = ind_df.to_numpy().reshape(len(ind_df), -1, 3)
            coordinates[f"{name}_{ind}"] = arr[:, :, :-1]
            confidences[f"{name}_{ind}"] = arr[:, :, -1]
    else:
        arr = df.to_numpy().reshape(len(df), -1, 3)
        coordinates[name] = arr[:, :, :-1]
        confidences[name] = arr[:, :, -1]

    return coordinates, confidences, bodyparts


def _name_from_path(filepath, path_in_name, path_sep, remove_extension):
    """Create a name from a filepath.

    Either return the name of the file (with the extension removed) or return
    the full filepath, where the path separators are replaced with `path_sep`.
    """
    if remove_extension:
        filepath = os.path.splitext(filepath)[0]
    if path_in_name:
        return filepath.replace(os.path.sep, path_sep)
    else:
        return os.path.basename(filepath)