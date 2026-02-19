### Pretrain-Experiments Data Insertion ###

import logging
from typing import List, Optional, Tuple

import h5py

log = logging.getLogger(__name__)


class InsertionMapReader:
    """Minimal reader for optimized insertion map HDF5 files.

    Maps indices to [(position, [token_ids]), ...].
    Used by the data loader to apply token insertions during training.

    The HDF5 file contains 5 flat arrays (optimized format):
        /keys, /tuple_offsets, /positions, /token_offsets, /tokens
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self._f = None
        with h5py.File(hdf5_path, "r") as f:
            keys = f["keys"][:]
        self._key_to_idx = {int(k): i for i, k in enumerate(keys)}
        log.info(
            "Loaded insertion map from '%s' with %d indices",
            hdf5_path,
            len(keys),
        )

    def has_index(self, index: int) -> bool:
        return index in self._key_to_idx

    def get_all_indices(self) -> list:
        return list(self._key_to_idx.keys())

    def load(self, index: int) -> Optional[List[Tuple[int, List[int]]]]:
        if not self.has_index(index):
            return None
        if self._f is None:
            self._f = h5py.File(self.hdf5_path, "r")
        idx = self._key_to_idx[index]
        t_start = int(self._f["tuple_offsets"][idx])
        t_end = int(self._f["tuple_offsets"][idx + 1])
        result = []
        for t in range(t_start, t_end):
            pos = int(self._f["positions"][t])
            tok_start = int(self._f["token_offsets"][t])
            tok_end = int(self._f["token_offsets"][t + 1])
            tokens = self._f["tokens"][tok_start:tok_end].tolist()
            result.append((pos, tokens))
        return result

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

### End Pretrain-Experiments Data Insertion ###
