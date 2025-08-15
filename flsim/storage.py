"""Simple in-memory IPFS simulator."""
from typing import Dict


class IPFSSim:
    def __init__(self):
        self.storage: Dict[str, object] = {}
        self.counter = 0

    def save(self, obj) -> str:
        cid = f"Qm{self.counter:08d}"
        self.storage[cid] = obj
        self.counter += 1
        return cid

    def load(self, cid: str):
        return self.storage[cid]

__all__ = ["IPFSSim"]
