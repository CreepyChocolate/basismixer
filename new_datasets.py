import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
import partitura as pt
from basismixer.performance_codec import get_performance_codec

import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self,
                 match_paths,
                 seq_len, param_names=["beat_period", "timing", "articulation_log", "velocity_trend", "velocity_dev"],
                 feat_names="all"):

        self.data = []
        self.target_data = None
        self.param_names = param_names
        self.parameter_dict = {name: [] for name in param_names}

        print("Processing score data...")
        for match_file in tqdm(match_paths):

            try:

                xml = glob.glob(os.path.join(os.path.dirname(match_file), "*.musicxml"))[0]

                score = pt.load_score(xml)
                score = pt.score.merge_parts(score)
                score = pt.score.unfold_part_maximal(score, update_ids=True)

                nid_dict = dict((n.id, i) for i, n in enumerate(score.notes_tied))

                pt.score.expand_grace_notes(score)

                basis, bf_names = pt.musicanalysis.make_note_feats(score, feat_names, force_fixed_size=True)

                performance, alignment = pt.load_match(match_file)

                parameter_names = param_names

                pc = get_performance_codec(parameter_names)

                targets, snote_ids, unique_onset_idxs = pc.encode(
                    part=score,
                    ppart=performance[0],
                    alignment=alignment,
                    return_u_onset_idx=True
                )

                matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])
                basis_matched = basis[matched_subset_idxs]

            except Exception as e:
                print(match_file)
                print(e)
                continue

            padding_len = len(targets) % seq_len

            for name in param_names:
                new_targets = targets[name]
                t_padding_array = np.zeros(shape=seq_len - padding_len)
                new_targets = np.concatenate((new_targets, t_padding_array))
                new_targets = np.split(new_targets, len(new_targets) / seq_len)

                self.parameter_dict[name] += [target for target in new_targets]

            bm_padding_array = np.zeros(shape=(seq_len - padding_len, basis_matched.shape[1]))
            new_basis = np.vstack((basis_matched, bm_padding_array))
            new_basis = np.split(new_basis, len(new_basis) / seq_len)

            self.data += [basis for basis in new_basis]

    def choose_parameter(self, parameter_name):
        self.target_data = self.parameter_dict[parameter_name]

    def __getitem__(self, idx):

        x = self.data[idx]
        y = self.target_data[idx]

        return x, y

    def __len__(self):
        return len(self.data)


matches = glob.glob(os.path.join("asap-dataset", "Bach", "Fugue", "bwv_846", "*.match"), recursive=True)

custom_dataset = MyDataset(matches, seq_len=50)

for name in ["beat_period", "timing", "articulation_log", "velocity_trend", "velocity_dev"]:
    custom_dataset.choose_parameter(name)
    torch.save(custom_dataset, f"{name}_data.pt")