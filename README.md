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
To enter the package mode press ```]``` and to exit press ```backspace```
```julia
pkg> add https://github.com/melaniesi/CheegerConvexificationBounds.jl.git
```

### Example
```julia
julia> using CheegerConvexificationBounds
julia> L = CheegerConvexificationBounds.GrevlexGrlexLaplacian.grevlex(7);
julia> alpha_start = 1; alpha_min = 1e-6; alpha_scale = 0.6;
julia> ncutsnew_max = 500; nuts_max = 10000;
julia> params = Parameters(alpha_max, alpha_min, alpha_scale,
                            ncutsnew_max, ncuts_max, eps_correction=0.001);
julia> result = lowerboundCheegerConvexification(L, params; diag_constraint=false);
```

More examples can be found in the folder [`examples/`](examples/) of this project.