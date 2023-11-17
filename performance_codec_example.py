import numpy as np
import partitura as pt
from basismixer.performance_codec import PerformanceCodec, get_performance_codec

xml_fn = "asap-dataset\\Bach\\Fugue\\bwv_854\\xml_score.musicxml"

match_fn = "asap-dataset\\Bach\\Fugue\\bwv_854\\LuA01M.match"


score = pt.load_score(xml_fn)

performance, alignment = pt.load_match(match_fn)

parameter_names = ["velocity_trend", "beat_period"]


pc = get_performance_codec(parameter_names)

expressive_parameters, snote_ids, unique_onset_idxs = pc.encode(
    part=score[0],
    ppart=performance[0],
    alignment=alignment,
    return_u_onset_idx=True
)