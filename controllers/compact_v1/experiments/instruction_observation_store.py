"""Lossless saved RGB-D access with a bounded resident image cache."""
from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path

import numpy as np


class ObservationStore(MutableMapping):
    def __init__(self, directory, observation_type, max_bytes=64 * 1024**2):
        if max_bytes < 0:
            raise ValueError("Observation cache budget must be nonnegative")
        self.directory = Path(directory)
        self.observation_type = observation_type
        self.max_bytes = max_bytes
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.loads = 0
        self._paths = {}
        self._cache = OrderedDict()

    @staticmethod
    def _size(observation):
        return sum(value.nbytes for value in vars(observation).values()
                   if isinstance(value, np.ndarray))

    def _remember(self, key, observation):
        if key in self._cache:
            self.resident_bytes -= self._size(self._cache.pop(key))
        size = self._size(observation)
        while self._cache and self.resident_bytes + size > self.max_bytes:
            _, old = self._cache.popitem(last=False)
            self.resident_bytes -= self._size(old)
        if size <= self.max_bytes:
            self._cache[key] = observation
            self.resident_bytes += size
            self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    def __setitem__(self, key, observation):
        if key != observation.frame_id or key in self._paths:
            raise ValueError("Saved observation IDs must be unique and match the frame")
        path = self.directory / f"observation_{key:05d}.npz"
        # The pilot records the immutable observation before memory integration.
        # Do not write a second copy or evict images that have not been saved.
        if not path.is_file():
            raise FileNotFoundError(path)
        self._paths[key] = path
        self._remember(key, observation)

    def __getitem__(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        with np.load(self._paths[key], allow_pickle=False) as data:
            observation = self.observation_type(**{
                name: data[name].item() if data[name].ndim == 0 else data[name]
                for name in data.files
            })
        if observation.frame_id != key:
            raise ValueError("Saved observation ID differs from its index")
        self.loads += 1
        self._remember(key, observation)
        return observation

    def __contains__(self, key):
        return key in self._paths

    def __delitem__(self, key):
        del self._paths[key]
        if key in self._cache:
            self.resident_bytes -= self._size(self._cache.pop(key))

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)

    def stats(self):
        return dict(saved_observations=len(self), resident_observations=len(self._cache),
                    resident_bytes=self.resident_bytes, peak_resident_bytes=self.peak_resident_bytes,
                    budget_bytes=self.max_bytes, disk_loads=self.loads)
