### Pretrain-Experiments Data Insertion ###

import logging
import threading
from typing import List, Optional, Tuple

import h5py

log = logging.getLogger(__name__)


class InsertionMapReader:
    """Minimal reader for optimized insertion map HDF5 files.

    Maps indices to [(position, [token_ids]), ...].
    Used by the data loader to apply token insertions during training.

    The HDF5 file contains 5 flat arrays (optimized format):
        /keys, /tuple_offsets, /positions, /token_offsets, /tokens

    Thread-safe: h5py is not thread-safe, so each thread gets its own file
    handle via threading.local().  Handles are opened lazily and excluded
    from pickle so that DataLoader worker processes each get their own.
    """

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self._local = threading.local()
        with h5py.File(hdf5_path, "r") as f:
            keys = f["keys"][:]
            self.num_indices = len(keys)
            self.num_tuples = f["positions"].shape[0]
            self.total_tokens = f["tokens"].shape[0]
        self._key_to_idx = {int(k): i for i, k in enumerate(keys)}
        log.info(
            "InsertionMapReader initialized from '%s': %d sequences with insertions, "
            "%d total tokens to insert",
            hdf5_path,
            self.num_indices,
            self.total_tokens,
        )

    def has_index(self, index: int) -> bool:
        return index in self._key_to_idx

    def get_all_indices(self) -> list:
        return list(self._key_to_idx.keys())

    def _get_file(self):
        """Get a per-thread h5py file handle, opening lazily on first access."""
        f = getattr(self._local, "f", None)
        if f is None:
            f = h5py.File(self.hdf5_path, "r")
            self._local.f = f
        return f

    def load(self, index: int) -> Optional[List[Tuple[int, List[int]]]]:
        if not self.has_index(index):
            return None
        f = self._get_file()
        idx = self._key_to_idx[index]
        t_start = int(f["tuple_offsets"][idx])
        t_end = int(f["tuple_offsets"][idx + 1])
        result = []
        for t in range(t_start, t_end):
            pos = int(f["positions"][t])
            tok_start = int(f["token_offsets"][t])
            tok_end = int(f["token_offsets"][t + 1])
            tokens = f["tokens"][tok_start:tok_end].tolist()
            result.append((pos, tokens))
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_local"] = None  # thread-local state cannot be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()
        # each thread will open its own file handle lazily via _get_file()

### End Pretrain-Experiments Data Insertion ###
