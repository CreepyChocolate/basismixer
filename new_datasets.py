import numpy as np
import shutil
import os
import glob
import re

rng = np.random.default_rng()
sep = os.sep


def make_random_dataset(path, keep_artists=1):

    os.makedirs(path, exist_ok=True)

    xml_scores = glob.glob(os.path.join("asap-dataset", "**", "*.musicxml"), recursive=True)

    piece_dict = {os.path.dirname(score): [] for score in xml_scores}

    for piece in piece_dict:

        artist_matches = glob.glob(os.path.join(piece, "*.match"))

        artist_names = []
        for match in artist_matches:
            match = match.split(sep)
            artist_names.append(match[-1][:-(len("*.match"))])

        piece_dict[piece] = artist_names

        # copy folder structure
        piece_path = piece.split(sep)
        piece_path = sep.join(piece_path[1:])

        os.makedirs(os.path.join(path, piece_path), exist_ok=True)

        # choose random artists and their corresponding files
        n_available_artists = len(piece_dict[piece])
        sample_size = keep_artists if n_available_artists >= keep_artists else n_available_artists

        random_artists = rng.choice(piece_dict[piece], sample_size, replace=False)

        available_files = glob.glob(os.path.join(piece, "*"))

        pattern_list = [".*musicxml"]
        for artist in random_artists:
            pattern_list.append(f".*{artist}.*\.match")

        r = re.compile("|".join(pattern_list))
        allowed_files = list(filter(r.match, available_files))

        # copy to new folder
        for f in allowed_files:
            if os.path.isfile(f):
                shutil.copy(f, os.path.join(path, piece_path))
            elif os.path.isdir(f):
                directory = f.split(sep)[-1]
                shutil.copytree(f, os.path.join(path, piece_path, directory), dirs_exist_ok=True)
            else:
                raise ValueError(f"{f} is neither a file nor a directory")

    return_dict = {}

    for key in piece_dict:
        new_key = sep.join(key.split(sep)[1:])
        new_path = os.path.join(path, new_key)

        xml = glob.glob(os.path.join(new_path, "*.musicxml"))
        matches = glob.glob(os.path.join(new_path, "*.match"))

        key_dict = {"xml_files": xml, "match_files": matches}

        return_dict[new_path] = key_dict

    return return_dict


if __name__ == "__main__":
    make_random_dataset("new_asap_test", keep_artists=100)
