# Helper functions for week4 exercises
import numpy as np
from flygym.vision import Retina

def crop_hex_to_rect(visual_input):
    ommatidia_id_map = Retina().ommatidia_id_map
    rows = [np.unique(row) for row in ommatidia_id_map]
    max_width = max(len(row) for row in rows)
    rows = np.array([row for row in rows if len(row) == max_width])[:, 1:] - 1
    cols = [np.unique(col) for col in rows.T]
    min_height = min(len(col) for col in cols)
    cols = [col[:min_height] for col in cols]
    rows = np.array(cols).T
    return visual_input[..., rows, :].max(-1)
