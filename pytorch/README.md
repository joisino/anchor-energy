# Fast and Robust Comparison of Probability Measures in Heterogeneous Spaces (PyTorch)

It shoud be noted that pytorch implementation is slow (O(n^3) even for anchor energy.)

## How to use

`python distance_comparison.py [type] [opt] [file1] [file2]`

This outputs the distance (AE or AW).

List of types
* AE: Anchor Energy
* AW: Anchor Wasserstein

Opt should specify epsilon for AW.

Note that this outputs AE itself, while the cpp implementation outputs E_{h ~ A, g ~ B][OT(h, g)].

### Input file format

The input file should be consisted of n lines, where n is the number of points.

```
d_11 d_12 d_13 ... d_1n
d_21 d_22 d_23 ... d_2n
...
d_n1 d_n2 d_n3 ... d_nn
```

d_ij is the distance between point i and j.

Note that the format is slightly different from that in the cpp implementation. To use point matrices, use `AnchorFeature` function instead of `AnchorFeatureDistance`.

### Example

```
$ python ./distance_comparison.py AE ../samplein/1.txt ../samplein/2.txt
tensor(9.9270)
$ python ./distance_comparison.py AW 0.001 ../samplein/1.txt ../samplein/2.txt
tensor(12.8486)
```
