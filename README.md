# Usage

To create a Bloom filter with a false positive rate of 5% when occupied by 1000
items:

```
$ uv run python

>>> from bloom_filter import BloomFilter
>>> b = BloomFilter(n=1000, fpr=0.05)
>>> b.add("foo")
>>> print("foo" in b)  # prints True
>>> print("bar" in b)  # prints False (probably!)
```

# Tests

```
$ uv run pytest
```
