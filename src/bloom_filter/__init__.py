import math
from typing import Hashable


class BloomFilter:
    def __init__(self, n: int, fpr: float):
        """
        Construct a Bloom filter with a FPR approximately equal to `fpr` when it
        contains n items.

        The FPR of the Bloom filter will not be exactly fpr due to rounding.
        Inspect the `true_fpr` attribute of this object to get the true value.
        """
        self.n = n
        self.fpr = fpr
        self._validate_constructor_args()

        # FPR of a Bloom filter containing n items in m bits with k hash
        # functions is = [1 - e^(-kn/m)]^k.
        #
        # By calculus, FPR is minimized by choosing k = (m/n) ln 2.
        #
        # Plugging this choice back into the equation for FPR and rearranging
        # for m and rounding up yields:
        self.m = math.ceil(-n * math.log(fpr) / math.log(2) ** 2)
        self.k = math.ceil((self.m / n) * math.log(2))

        self.true_fpr = (1 - math.exp(-self.k * self.n / self.m)) ** self.k

        self._bits = [False] * self.m

    def _validate_constructor_args(self):
        if self.n <= 0 or not isinstance(self.n, int):
            raise ValueError(f"n must be a positive integer. n={self.n}")
        if not 0 < self.fpr < 1 or not isinstance(self.fpr, float):
            raise ValueError(
                f"False positive rate must be a positive float between 0 and 1. FPR={self.fpr}"
            )

    def _hash(self, x: Hashable, i: int) -> int:
        """
        Quick and dirty hash functions.

        Bloom filter hash functions need not be cryptographic, but we do need k
        independent hash functions, each of which distributes values uniformly
        over the interval 0 to m-1.

        For the purposes of this demo, I use the Python builtin `hash` to
        construct k hash functions.

        The `i`th of these hash functions is a linear combination of the regular
        hash builtin applied to the input, and the hash builtin applied to the
        input salted with an arbitrary string of text.
        """
        if i < 0 or i >= self.k:
            raise ValueError(
                f"This bloom filter defines hash functions 0..{self.k - 1}. "
                f"Attempted to compute {self.k}th hash function."
            )

        h1 = hash(x) % self.m
        h2 = hash((x, "salt")) % self.m

        return (h1 + i * h2) % self.m

    def add(self, x: Hashable) -> None:
        for hash_i in range(self.k):
            self._bits[self._hash(x, hash_i)] = True

    def __contains__(self, x: Hashable) -> bool:
        return all(self._bits[self._hash(x, hash_i)] for hash_i in range(self.k))
