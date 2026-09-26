"""Dependency-free local embedding fallback.

Feature hashing is deliberately simple but persistent and semantic enough for a small
personal memory store. Swap this class for sentence-transformers/Ollama embeddings later.
"""
from __future__ import annotations
import hashlib, math, re

class HashEmbedding:
    dimensions = 256
    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"[\w'-]+", text.lower()):
            # Small normalizer for common personal-memory wording. A production
            # embedder can replace this module without changing the store API.
            if word.startswith("prefer"):
                word = "prefer"
            elif len(word) > 4 and word.endswith("s"):
                word = word[:-1]
            index = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % self.dimensions
            vector[index] += 1.0
        magnitude = math.sqrt(sum(x*x for x in vector))
        return [x / magnitude for x in vector] if magnitude else vector

def cosine(a: list[float], b: list[float]) -> float: return sum(x*y for x, y in zip(a,b))
