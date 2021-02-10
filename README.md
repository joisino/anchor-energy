# Fast and Robust Comparison of Probability Measures in Heterogeneous Spaces

Paper: https://arxiv.org/abs/2002.01615

This repository provides C++ implementations of our proposed methods and baselines. It also provides Python wrappers of them.

## Build

No dependencies. Just run `make`.

```
$ make
```

## How to use

`distance_comparison [type] [opt] [file1] [file2]`

This outputs two values the distance (e.g., AE, AW) and elapsed time in milliseconds.

List of types
* AE: Anchor Energy
* RAE: Robust Anchor Energy
* AW: Anchor Wasserstein
* RAW: Robust Anchor Wasserstein
* GW: Gromov Wasserstein
* RGW: Robust Gromov Wasserstein

Opt should specify hyperparameters. Epsilon for AW and RAW, epsilon and tau for GW and RGW.

It should be noted that AE and RAW return E_{h ~ A, g ~ B][OT(h, g)]. So, to compute AE correctly, run x: `distance_comparison AE A B`, y: `distance_comparison AE A A`, and z: `distance_comparison AE B B`, and compute `2 * x - y - z`.

### Input file format

The input file should be consisted of n+1 lines, where n is the number of points.

```
n
d_11 d_12 d_13 ... d_1n
d_21 d_22 d_23 ... d_2n
...
d_n1 d_n2 d_n3 ... d_nn
```

d_ij is the distance between point i and j.

To specify the weights of points, change `prob` option in the `load` function in `distance_comparison.cpp`, build the program, and specify the weights in the second line, i.e., 

```
n
w_1 w_2 w_3 ... w_n
d_11 d_12 d_13 ... d_1n
d_21 d_22 d_23 ... d_2n
...
d_n1 d_n2 d_n3 ... d_nn
```

where w_i is the weight of point i.

### Example

```
$ ./distance_comparison AE samplein/1.txt samplein/2.txt
19.842169 0.000000
$ ./distance_comparison AW 0.001 samplein/1.txt samplein/2.txt
14.689693 11.000000
```

## Matching

Run `make matching`.

```
$ make matching
$ ./matching AE graphs/G1.txt graphs/G2.txt
646 13 5 7 16 5 278 513 514 640 19 515 516 517 518 ...
```
This outputs the matching vector. To access the AEM matching, please access `P` of `matching.cpp`.

### Input file format

The input file should be consisted of m+1 lines, where m is the number of edges.

```
n m
a_1 b_1
a_2 b_2
a_3 b_3
...
a_m b_m
```

where n is the number of nodes, and a_i and b_i indicates there exists and edge between node a_i and b_i. Node indices are 0-indexed.

To use other formats, please modify `load` function in `matching.cpp` (e.g., by replacing one in `distance_comparison.cpp`)

## Python Wrapper

The python wrapper requires Boost.Python and cmake (>= 3.12). You can install them by

```
$ sudo apt install python3 build-essential python3-dev cmake libboost-all-dev
```

Be sure that libboost_python is compatible with your python version. You can build the wrapper by

```
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
$ python3 -c "import anchor; help(anchor.anchor_energy)" 
```

For example, you can use `sample.py`.

```
$ PYTHONPATH=./build python3 sample.py
AE 19.842169                                                                            
RAE 0.119391                                                                            
AW 14.689693                    
RAW 0.052937                                                                            
GW 1060.477979
AEM ...
AWM ...
GWM ...
```

When you use the wrapper, be sure that `anchor.so` is installed in the search path for modules.


## Pytorch Implementation

An alternative pytorch implementation is avaibale in `pytorch` directory. It is differentiable but slow.


## Feedback and Contact

Please feel free to contact me at r.sato AT ml.ist.i.kyoto-u.ac.jp, or to open issues.


## Citation

```
@article{sato2020fast,
  author    = {Ryoma Sato and
               Marco Cuturi and
               Makoto Yamada and
               Hisashi Kashima},
  title     = {Fast and Robust Comparison of Probability Measures in Heterogeneous
               Spaces},
  journal   = {arXiv},               
  year      = {2020},
}
```