# CheegerConvexificationBounds.jl

Package to compute strong lower bounds on the edge expansion of an undirected graph $G = (V,E)$.
The edge expansion of $G$ is defined as
```math
h(G) = \min_{\emptyset \neq S \subset V} \frac{\lvert \partial S \rvert}{\min \{ \lvert S \rvert, \lvert S \setminus V \rvert\}}
```
where $\lvert \partial S \rvert$ denotes the size of the cut induced by the set of vertices $S$, i.e.,
```math
\partial S = \{ \{i,j\} \in E(G) \mid i \in S, j \in V \setminus S\}.
```

### Installation
To enter the package mode in Julia press ```]``` and to exit press ```backspace```.
```julia
pkg> add https://github.com/melaniesi/CheegerConvexificationBounds.jl.git
```

### Example
```julia
julia> using CheegerConvexificationBounds
julia> L = CheegerConvexificationBounds.GrevlexGrlexLaplacian.grlex(4);
julia> alpha_start = 1; alpha_min = 1e-6; alpha_scale = 0.6;
julia> ncutsnew_max = 500; ncuts_max = 10000;
julia> params = Parameters(alpha_start, alpha_min, alpha_scale,
                            ncutsnew_max, ncuts_max, eps_correction=0.001);
julia> result = lowerboundCheegerConvexification(L, params; diag_constraint=false);
                                                             ┏━━━━━━━━━━━━━━━━━━━━━━━━┓                             
                                                             ┃        C  U  T  S      ┃                             
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃    primal          dual            fopt          alpha     ┃  total   new   removed ┃   time_elapsed   iteration ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┗━━━━━━━━━━━━━━━━━━━━━━━━┛━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
   1.4654362       8.5094201       7.2196347    1.0000000000       0                           0.01 s        1
   1.3103050       1.5353373       1.3824974    0.6000000000       0                           0.02 s        2
   1.1377973       1.4449348       1.2112816    0.3600000000       0                           0.03 s        3
   1.0936916       1.1706934       1.1025107    0.2160000000       0                           0.06 s        4
   1.0818725       1.1128854       1.0874053    0.1296000000       0     27      0             0.07 s        5
   1.0638181       1.1087441       1.0733555    0.0777600000      27      0      0             0.10 s        6
   1.0415724       1.0959189       1.0527777    0.0466560000      27      0      0             0.11 s        7
   1.0071489       1.0899655       1.0233413    0.0279936000      27      0      0             0.11 s        8
   1.0042538       0.9997662       1.0058093    0.0167961600      27      0      0             0.14 s        9
   0.9997872       0.9997709       1.0023822    0.0100776960      27      0      0             0.14 s       10
   1.0000674       1.0000078       1.0000040    0.0060466176      27      0      0             0.14 s       11
   0.9993603       0.9999537       0.9999999    0.0036279706      27      0      0             0.14 s       12
   0.9993348       0.9999573       0.9999999    0.0021767823      27      0      0             0.14 s       13
   0.9973768       0.9999614       0.9999999    0.0013060694      27      0      0             0.14 s       14
   0.9975411       0.9999708       0.9999998    0.0007836416      27      0      0             0.14 s       15
   0.9940505       0.9999799       0.9999998    0.0004701850      27      0      0             0.14 s       16
   1.0004581       1.0000409       1.0000000    0.0002821110      27      0      0             0.14 s       17
   1.0003305       0.9999986       0.9999999    0.0001692666      27      0      0             0.14 s       18
   0.9995228       0.9999988       0.9999999    0.0001015600      27      0      0             0.14 s       19
   1.0003225       1.0000005       0.9999999    0.0000609360      27      0      0             0.15 s       20
   1.0003943       1.0000004       0.9999999    0.0000365616      27      0      0             0.15 s       21
   1.0000587       0.9999997       0.9999999    0.0000219370      27      0      0             0.15 s       22
   0.9996225       0.9999999       0.9999999    0.0000131622      27      0      0             0.15 s       23
   1.0006803       1.0000001       0.9999999    0.0000078973      27      0      0             0.15 s       24
   0.9996703       0.9999998       0.9999999    0.0000047384      27      0      0             0.15 s       25
   0.9998174       0.9999999       0.9999999    0.0000028430      27      0      0             0.15 s       26
   0.9974775       0.9999998       0.9999999    0.0000017058      27      0      0             0.15 s       27
   0.9982830       1.0000000       0.9999999    0.0000010235      27      0      0             0.15 s       28
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃   DNN lower bound:                      0.99999978                                                               ┃
┃   time:                                 0.148 s                                                                  ┃
┃   iterations:                          28                                                                        ┃
┃   added cuts:                          27                                                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

```

More examples can be found in the folder [`examples/`](examples/) of this project.

### References
This package is part of the publication

Timotej Hrga, Melanie Siebenhofer, Angelika Wiegele. (2024). _Connectivity via convexity: Bounds on the edge expansion in graphs._ [submitted for publication].
