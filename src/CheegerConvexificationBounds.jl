# ==========================================================================
#   CheegerConvexificationBounds.jl -- Package to compute lower bounds on
#                                      the Cheeger constant of a graph.
# --------------------------------------------------------------------------
#   Copyright (C) 2024 Melanie Siebenhofer <melaniesi@edu.aau.at>
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see https://www.gnu.org/licenses/.
# ==========================================================================

module CheegerConvexificationBounds

include("BoundsMosek.jl")
include("GrevlexGrlexLaplacian.jl")
include("RudyGraphIO.jl")

using DataStructures
using Dates
using LinearAlgebra
using LBFGSB
using MKL
using Printf

export Parameters, lowerboundCheegerConvexification

"""
    Parameters(alpha_start, alpha_min, alphascale, max_newcuts, max_cutstotal, niterations_nocutsstart=5,
               niterations_end=100)

Data structure for the parameters of the algorithm to compute a lower bound of the edge expansion.

# Arguments:
- `alpha_start`:   Start value of the parameter α
- `alpha_min`:     Minimum value of the parameter α, iterations are stopped as soon as alpha is smaller than `alpha_min`.
- `alphascale`:    Scaling factor of α.
- `max_newcuts`:   The maximum number of new cuts added after one iteration.
- `max_cutstotal`: The maximum number of cuts in total.

# Optional Arguments:
- `niterations_nocutsstart=5`: The number of iterations before cuts are added.
- `niterations_end=500`:       The maximum number of iterations at the end where no new cuts are added and
                               α isn't decreased (so it is minimal).

# Keyword Arguments
- `min_newcuts_samealpha=50`:  The minimum number of new cuts to use the same α and not reduce it.
- `eps_triviol=1e-3`:          Threshold of violation of an inequalitie to be added as cut.
- `eps_tripurge=1e-5`:         Cuts with dual value `eps_tripurge` are removed.
- `nopurge_lastrounds=false`:  If `nopurge_lastrounds=true`, no cuts are removed during the extra iterations at the end.
- `eps_correction=0.01`:       Maximum of correction by safe dual bound procedure to stop the extra iterations at the end.
- `lbfgsb_factr=1e8`:          Parameter `factr` for the L-BFGS-B solver.
- `lbfgsb_factr_last=1e8`:     Parameter `factr` for solving the inner problem when α has its minimal value.
- `lbfgsb_maxiter=2000`:       Parameter `maxiter` for the L-BFGS-B solver.
- `lbfgsb_m=10`:               Parameter `m` for the L-BFGS-B solver.
"""
struct Parameters
    alpha_start::Float64
    alpha_min::Float64
    alphascale::Float64
    max_newcuts::Int64             # maximum number of new cuts added after one iteration
    max_cutstotal::Int64           # maximum number of cuts in total
    eps_triviol::Float64           # min violation of inequalities to be added as cut
    niterations_nocutsstart::Int64 # number of iterations before cuts are added
    niterations_end::Int64         # number of iterations at the end where no new cuts are added and alpha isn't decreased (so it is minimal)
    min_newcuts_samealpha::Int64   # minimum number of new cuts to use the same alpha
    eps_tripurge::Float64          # remove triangle inequality if dual value < eps_tripurge
    nopurge_lastrounds::Bool       # should in the last iterations also no cuts be purged?
    eps_correction::Float64        # maximum correction to stop algorithm
    lbfgsb_factr::Float64          # parameter value factr in L-BFGSB solver
    lbfgsb_factr_last::Float64     # value of parameter lbfgsb_factr in last two iterations
    lbfgsb_maxiter::Int64
    lbfgsb_m::Int64
    Parameters(alpha_start, alpha_min, alphascale, max_newcuts, max_cutstotal, niterations_nocutsstart=5,
               niterations_end=500; min_newcuts_samealpha=50, eps_triviol=1e-3, eps_tripurge=1e-5, nopurge_lastrounds=false, eps_correction=0.01,
               lbfgsb_factr=1e8, lbfgsb_factr_last=1e8, lbfgsb_maxiter=2000, lbfgsb_m=10) =
                    any(<(0), [alpha_start, alpha_min, alphascale, max_newcuts, max_cutstotal, niterations_end, niterations_nocutsstart,
                               lbfgsb_factr, lbfgsb_factr_last, lbfgsb_maxiter]) || lbfgsb_m < 3 || lbfgsb_m > 20 ? 
                    error("invalid parameter settings") :
                    new(alpha_start, alpha_min, alphascale, max_newcuts, max_cutstotal, eps_triviol, niterations_nocutsstart, niterations_end,
                        min_newcuts_samealpha, eps_tripurge, nopurge_lastrounds, eps_correction, lbfgsb_factr, lbfgsb_factr_last, lbfgsb_maxiter, lbfgsb_m)
end

"""
    TriangleIneq(i, j, k, type)

Data structure that represents the scaled BQP triangle inequalities.
It holds that `i < j < k` and the inequality is defined by `type`.

- `type = 1`: inequality represented is `Yij + Yik - Yjk ≤ yi`
- `type = 2`: inequality represented is `Yij + Yjk - Yik ≤ yj`
- `type = 3`: inequality represented is `Yik + Yjk - Yij ≤ yk`
"""
struct TriangleIneq
    i::Int
    j::Int
    k::Int
    type::Int
end

"""
    getLbig(L)

Return the cost matrix for the SDP relaxation of dimension 2n + 3.

Cost matrix formed by the Laplacian matrix `L` of the graph and of the form
```
                 ╭                 ╮
                 │  L    0     0   │
    Lbig = 0.5 * │  0    L     0   │ .
                 |  0    0     0   |
                 ╰                 ╯  

"""
function getLbig(L)
    n = size(L, 1)
    return Symmetric([1//2 * L     zeros(n, (n + 3))  ;
                    zeros(n,n)  1/2 * L  zeros(n,3);
                    zeros(3,(2 * n + 3))]);
end


"""
    projection_PSD_cone(M)

Compute the projection of the matrix `M`
onto the cone of positive semidefinite
matrices.

Returns `U, d`, the projection can be computed
as `U * diagm(d) * U'`.
"""
function projection_PSD_cone(M)
    ev, U = eigen(Symmetric(M))
    ind1 = findfirst(>(0), ev)
    if isnothing(ind1)
        return zeros(size(M,1),1), [0]
    end
    idx = ind1:length(ev)
    U = U[:,idx]
    return U, ev[idx]
end

"""
    vec2mat!(S, x)

Write values of vector `x` into the upper triangular part of matrix `S`.

The entries of `S` are filled column-wise from top to bottom, i.e.,

```
S[1,1] = x[1]
S[1,2] = x[2]
S[2,2] = x[3]
S[1,3] = x[4]
S[2,3] = x[5]
S[3,3] = x[6]
       .
       .
       .
S[n,n] = x[n(n+1)/2]
```
"""
function vec2mat!(S, x)
    dim_S = size(S,1)
    ind = 1
    for j = 1:dim_S
        for i = 1:j
            S[i,j] = x[ind]
            ind += 1
        end
    end
end

"""
    get_Aadjointnu!(Atnu, nu, n; diag_constraint=true)

Compute A'(ν) of (DNN-PFRC) and store it in `Atnu`.

If `diag_constraint=true`, the constraint `diag(Y^{11}) = y^1, diag(Y^{22}) = y^2` is included in A.
Otherwise only the constraints `e'y1 = 1` and `diag(Y12) = 0` are included.
"""
function get_Aadjointnu!(Atnu, nu, n; diag_constraint=true)
    @assert size(Atnu,1) == 2*n + 3
    fill!(Atnu, 0)
    Atnu[1:n,end] .= 0.5 * nu[1]  # constraint e'y^1 = 1
    for i = 1:n
        Atnu[i,(n + i)] = 0.5 * nu[1 + i] # constraint diag(Y^{12}) = 0
    end
    if diag_constraint  # constraint diag(Y^{11}) = y^1, diag(Y^{22}) = y^2
        Atnu[1:n,end] += 0.5 * nu[(n + 2):(2*n + 1)]
        Atnu[(n + 1):2*n,end] = 0.5 * nu[(2*n + 2):end]
        for i = 1:2*n
            Atnu[i,i] = - nu[1 + n + i]
        end
    end
end


"""
    get_Badjointmu!(Btmu::Matrix, mu::Vector, violated_triangles, n::Int; diag_constraint=true)

Compute B'(μ) of (DNN-PFRC) and store it in `Btmu`.

In B are the scaled BQP triangle cuts. The cuts are stored in `violated_triangles::Vector{TriangleIneq}`.

# Keyword Arguments:
- `diag_constraint=true`: If `diag_constraint=true`, the inequality constraints are formed as
                          `0.5 * (Y[i,j] + Y[j,i] + Y[i,k] + Y[k,i] - Y[j,k] - Y[k,j]) ≤ 1/3 * (Y[i,i] + 2 * y[i])`,
                          else it is written as
                          `0.5 * (Y[i,j] + Y[j,i] + Y[i,k] + Y[k,i] - Y[j,k] - Y[k,j]) ≤ y[i]`.
"""
function get_Badjointmu!(Btmu::Matrix, mu::Vector, violated_triangles, n::Int; diag_constraint=true)
    @assert size(Btmu,1) == 2*n + 3
    @assert length(mu) == length(violated_triangles)
    fill!(Btmu, 0)
    if diag_constraint
        for (ind,triv) in enumerate(violated_triangles)
            i,j,k,t = triv.i, triv.j, triv.k, triv.type
            entry = mu[ind]
            if t == 1
                Btmu[i,j] += 0.5 * entry
                Btmu[i,k] += 0.5 * entry
                Btmu[j,k] -= 0.5 * entry
                Btmu[i,i] -= entry / 3.0
                Btmu[i,end] -= entry / 3.0
            elseif t == 2
                Btmu[i,j] += 0.5 * entry
                Btmu[j,k] += 0.5 * entry
                Btmu[i,k] -= 0.5 * entry
                Btmu[j,j] -= entry / 3.0
                Btmu[j,end] -= entry / 3.0
            elseif t == 3
                Btmu[i,k] += 0.5 * entry
                Btmu[j,k] += 0.5 * entry
                Btmu[i,j] -= 0.5 * entry
                Btmu[k,k] -= entry / 3.0
                Btmu[k,end] -= entry / 3.0
            end
        end
    else
        for (ind,triv) in enumerate(violated_triangles)
            i,j,k,t = triv.i, triv.j, triv.k, triv.type
            entry = 0.5 * mu[ind]
            if t == 1
                Btmu[i,j] += entry
                Btmu[i,k] += entry
                Btmu[j,k] -= entry
                Btmu[i,end] -= entry
            elseif t == 2
                Btmu[i,j] += entry
                Btmu[j,k] += entry
                Btmu[i,k] -= entry
                Btmu[j,end] -= entry
            elseif t == 3
                Btmu[i,k] += entry
                Btmu[j,k] += entry
                Btmu[i,j] -= entry
                Btmu[k,end] -= entry
            end
        end
    end
end

"""
    augmLagrangeFct(x, n, Lbig, V, normR², R, alpha; diag_constraint=true)

Compute the value of the augmented Lagragian function and the gradient with respect to `x`, i.e., ν and S.

The objectiv function is
```
Fα(ν,S) = ν[1] - 1/(2α) ‖(V'(A'ν + S - Lbig)V + αR)+ ‖² + α/2 ‖R‖² 
```

# Keyword Arguments:
- `diag_constraint=true`: If `diag_constraint=true`, the diagonal constraint `diag(Y11) = y1` and `diag(Y22) = y2` is
                          included in the formulation.
"""
function augmLagrangeFct(x, n, Lbig, V, normR², R, alpha; diag_constraint=true)
    # n = diag_constraint ? Int(-5 + sqrt(11 + 2 * length(x))) >> 1 : Int(sqrt(2*(1 + length(x))) - 4) >> 1
    n_eqconstraints = diag_constraint ? 3*n + 1 : n + 1
    dim_S = 2*n + 3

    nu = x[1:n_eqconstraints]
    Atnu = Symmetric(zeros(dim_S, dim_S))
    get_Aadjointnu!(parent(Atnu), nu, n, diag_constraint=diag_constraint)
    S = Symmetric(zeros(dim_S, dim_S))
    vec2mat!(parent(S), x[(n_eqconstraints + 1):end])
    
    M = V' * ((Atnu + S - Lbig) * V) + alpha * R
    U, ev = projection_PSD_cone(M)
    
    # function value
    fval = -nu[1] + sum(ev.^2)/(2*alpha) - 0.5*alpha * normR²

    # gradient
    VU = V*U
    temp = 1/alpha * VU * Diagonal(ev) * VU'
    g = zeros(n_eqconstraints + dim_S * (dim_S + 1) >> 1)
    g[1] = sum(temp[1:n,end]) - 1
    ind = 2
    for i = 1:n
        g[ind] = temp[i,(n + i)]
        ind += 1
    end
    if diag_constraint
        for i = 1:2*n
            g[ind] = temp[i,end] - temp[i,i]
            ind += 1
        end
    end

    for j = 1:dim_S
        for i = 1:j-1
            g[ind] = 2 * temp[i,j]
            ind += 1
        end
        g[ind] = temp[j,j]
        ind += 1
    end
    return fval, g
end


"""
    augmLagrangeFct_withtriangles(x, n, Lbig, V, normR², R, alpha, violated_triangles::Vector{TriangleIneq})

Compute the value of the augmented Lagragian function and the gradient with respect to `x`, i.e., ν, μ, and S.

The objectiv function is
```
Fα(ν,μ,S) = ν[1] - 1/(2α) ‖(V'(A'ν - B'μ + S - Lbig)V + αR)+ ‖² + α/2 ‖R‖² 
```

# Keyword Arguments:
- `diag_constraint=true`: If `diag_constraint=true`, the diagonal constraint `diag(Y11) = y1` and `diag(Y22) = y2` is
                          included in the formulation.
"""
function augmLagrangeFct_withtriangles(x, n, Lbig, V, normR², R, alpha, violated_triangles; diag_constraint=true)
    # n = diag_constraint ? Int(-5 + sqrt(11 + 2 * length(x))) >> 1 : Int(sqrt(2*(1 + length(x))) - 4) >> 1
    n_eqconstraints = diag_constraint ? 3*n + 1 : n + 1
    n_triangles = length(violated_triangles)
    dim_S = 2*n + 3

    nu = x[1:n_eqconstraints]
    Atnu = Symmetric(zeros(dim_S, dim_S))
    get_Aadjointnu!(parent(Atnu), nu, n, diag_constraint=diag_constraint)
    Btmu = n_triangles > 0 ? Symmetric(zeros(dim_S, dim_S)) : missing
    if n_triangles > 0
        get_Badjointmu!(parent(Btmu), x[(n_eqconstraints + 1):(n_eqconstraints + n_triangles)], violated_triangles,
                        n, diag_constraint=diag_constraint)
    end
    S = Symmetric(zeros(dim_S, dim_S))
    vec2mat!(parent(S), x[(n_eqconstraints + n_triangles + 1):end])
    
    M = n_triangles > 0 ? V' * ((Atnu - Btmu + S - Lbig) * V) + alpha * R : V' * ((Atnu + S - Lbig) * V) + alpha * R
    U, ev = projection_PSD_cone(M)
    
    # function value
    fval = -nu[1] + sum(ev.^2)/(2*alpha) - 0.5*alpha * normR²

    # gradient
    VU = V*U
    temp = 1/alpha * VU * Diagonal(ev) * VU'
    g = zeros(n_eqconstraints + length(violated_triangles) + dim_S * (dim_S + 1) >> 1)
    g[1] = sum(temp[1:n,end]) - 1
    ind = 2
    # derivative w: diag(Y12) = 0
    for i = 1:n
        g[ind] = temp[i,(n + i)]
        ind += 1
    end
    if diag_constraint
        # derivative w: diag constraint
        for i = 1:2*n
            g[ind] = temp[i,end] - temp[i,i]
            ind += 1
        end
        # derivative u: triangle inequalities
        for t in violated_triangles
            if t.type == 1
                g[ind] = -temp[t.i,t.j] -temp[t.i,t.k] + temp[t.j,t.k] + 1.0/3*(temp[t.i,t.i] + 2*temp[t.i,end])
            elseif t.type == 2
                g[ind] = -temp[t.i,t.j] -temp[t.j,t.k] + temp[t.i,t.k] + 1.0/3*(temp[t.j,t.j] + 2*temp[t.j,end])
            elseif t.type == 3
                g[ind] = -temp[t.i,t.k] -temp[t.j,t.k] + temp[t.i,t.j] + 1.0/3*(temp[t.k,t.k] + 2*temp[t.k,end])
            end
            ind += 1
        end        
    else
        # derivative u: triangle inequalities
        for t in violated_triangles
            if t.type == 1
                g[ind] = -temp[t.i,t.j] -temp[t.i,t.k] + temp[t.j,t.k] + temp[t.i,end]
            elseif t.type == 2
                g[ind] = -temp[t.i,t.j] -temp[t.j,t.k] + temp[t.i,t.k] + temp[t.j,end]
            elseif t.type == 3
                g[ind] = -temp[t.i,t.k] -temp[t.j,t.k] + temp[t.i,t.j] + temp[t.k,end]
            end
            ind += 1
        end
    end
    # derivative S
    for j = 1:dim_S
        for i = 1:j-1
            g[ind] = 2 * temp[i,j]
            ind += 1
        end
        g[ind] = temp[j,j]
        ind += 1
    end
    return fval, g
end

"""
    compute_safedualbound(Z_tilde, V, btnu, ubλmaxY)

Compute a safe dual bound from the output `Z_tilde` and `btnu`.

# Arguments:
- `Z_tilde`: Matrix computed as `V'(Lbig - A'ν + B'μ - S)V`.
- `V`:       Matrix used for facial reduction of problem.
- `btnu`:    Value b'ν from algorithm.
- `ubλmaxY`: Upper bound on the eigenvalue of an optimal solution Y of the problem.
"""
function compute_safedualbound(Z_tilde, V, btnu, ubλmaxY)
    VZnewVt = Symmetric(V * Z_tilde * V')
    ev = eigvals(VZnewVt)
    return btnu + ubλmaxY * sum(filter(<(0), ev))
end

"""
    add_newviolatedtriangles!(violated_triangles::Vector{TriangleIneq}, Y, max_newcuts, 
                              istart=1, iend=(size(Y,1)-3)>>1; )

Separate new violated inequalities from `Y` and add to `violated_triangles`.

# Arguments:
- `violated_triangles::Vector{TriangleIneq}`: A list of already added BQP inequalities.
- `Y::Matrix`                               : Matrix to separate the cuts from.
- `max_newcuts`                             : Only the most `max_newcuts` violated inequalities are added.
- Indices on the matrix `Y` for separation  : The cuts are separated from `Y[istart:iend,istart:iend]` only.
    - `istart=1`
    - `iend=(size(Y,1) - 3)>>1`     

# Keyword Arguments:
- `eps_viol=1e-4`:   Threshold on violation of inequality to be considered as violated.
"""
function add_newviolatedtriangles!(violated_triangles::Vector{TriangleIneq}, Y, max_newcuts,  istart=1, iend=(size(Y,1)-3)>>1; eps_viol=1e-4)
    if max_newcuts == 0 return 0; end
    new_cuts = PriorityQueue{TriangleIneq, Float64}()
    n_addedcuts = 0
    new_found = false
    new_cut = undef
    new_viol = 0
    for i = istart:(iend - 2)
        for j = (i + 1):(iend - 1)
            for k = (j + 1):iend
                if Y[j,i] + Y[k,i] - Y[j,k] - Y[end,i] > eps_viol
                    new_found = true
                    new_cut = TriangleIneq(i,j,k,1)
                    new_viol = Y[j,i] + Y[k,i] - Y[j,k] - Y[end,i]
                elseif Y[i,j] + Y[k,j] - Y[k,i] - Y[end,j] > eps_viol
                    new_found = true
                    new_cut = TriangleIneq(i,j,k,2)
                    new_viol = Y[i,j] + Y[k,j] - Y[k,i] - Y[end,j]
                elseif Y[i,k] + Y[j,k] - Y[i,j] - Y[end,k] > eps_viol
                    new_found = true
                    new_cut = TriangleIneq(i,j,k,3)
                    new_viol = Y[i,k] + Y[j,k] - Y[i,j] - Y[end,k]
                end
                if new_found
                    if n_addedcuts < max_newcuts
                        enqueue!(new_cuts, new_cut, new_viol)
                        n_addedcuts += 1
                    elseif peek(new_cuts)[2] < new_viol # if smallest violation so far smaller than the one of new_cut
                        dequeue!(new_cuts)
                        enqueue!(new_cuts, new_cut, new_viol)
                    end
                    new_found = false
                end
            end
        end
    end
    if n_addedcuts > 0
        len_old = length(violated_triangles)
        push!(violated_triangles, collect(keys(new_cuts))...)
        unique!(violated_triangles)
        n_addedcuts = length(violated_triangles) - len_old
    end
    return n_addedcuts
end


"""
    remove_violatedtriangles!(violated_triangles::Vector, xcur::Vector, lowerbounds::Vector, n_eqconstraints::Int)

Remove cuts from `violated_triangles` based on dual values stored in `xcur`.

# Arguments:
- `violated_triangles::Vector`: A list of cuts of length `k`, the ones with small dual value in `xcur` are removed.
- `xcur`: Dual variables corresponding to `violated_triangles` start at index `n_eqconstraints + 1`. The dual variables
          corresponding to removed cuts are also removed from `xcur`.
- `lowerbounds`: Vector of length `k + (2n+3)(2n+4)/2`, entries corresponding to cuts start at index 1,
                 corresponding values from `lowerbounds` are removed as well.
- `n_eqconstraints`: The number of equality constraints of the formulation.

# Keyword Arguments:
- `eps_purge=1e-8`:   Cuts with dual value less than `eps_purge` get removed.
"""
function remove_violatedtriangles!(violated_triangles::Vector, xcur::Vector, lowerbounds::Vector, n_eqconstraints::Int; eps_purge=1e-8)
    indices_toremove = findall(<(eps_purge), view(xcur, (n_eqconstraints + 1):(n_eqconstraints + length(violated_triangles))))
    deleteat!(violated_triangles, indices_toremove)
    indices_toremove .+= n_eqconstraints
    deleteat!(xcur, indices_toremove)
    deleteat!(lowerbounds, indices_toremove)
    return length(indices_toremove)
end


"""
    lowerboundCheegerConvexification(L, params::Parameters)

Compute a lower bound on the Cheeger constant of a graph.

# Arguments:
- `L::Matrix`: Laplacian matrix representing the graph.
```
- `params::Parameters`:  Parameters for the algorithm.

# Keyword Arguments:
- `diag_constraint=true`: Indicate whether the diagonal constraint `diag(Y11) = y1, diag(Y22) = y2` is added.

# Output:
A dictionary with the following fields is returned.
- `lb`              :  Lower bound obtained by the algorithm (best safe dual bound within the last + additional iterations)
- `time-wc`         :  Wall clock time (in seconds) needed to compute the lower bound.
- `iterations`      :  Total number of iterations.
- `ncuts`           :  Number of cuts added (and not removed).
- `foptlast`        :  Value of Fα(ν,μ,S) in the last iteration.
- `Y`               :  Value of matrix `Y` at the end of algorithm.
- `R`               :  Value of matrix `R` at the end of algorithm.
- `ν`               :  Value of vector `ν` at the end of algorithm.
- `Z`               :  Value of matrix `Z` at the end of algorithm.
- `S`               :  Value of matrix `S` at the end of algorithm.
- `iterations_extra`:  The number of additional iterations performed at the end.
"""
function lowerboundCheegerConvexification(L, params::Parameters; diag_constraint=true)
    start_time = now()
    n = size(L,1)
    Lbig = getLbig(L)
    
    dim_S = 2*n + 3
    len_vecS = dim_S * (dim_S + 1) >> 1
    n_eqconstraints = diag_constraint ? 3*n + 1 : n + 1
    ubλmaxY = n + floor(n/2)^2 #- 2*(floor(n/2) - 1)

    V = vcat(Matrix(I,n,n), -Matrix(I,n,n), -ones(1,n), ones(1,n), zeros(1,n))
    V = hcat(V, vcat(zeros(n), ones(n), floor(n/2), -1, 1))
    V = Matrix(qr(V).Q)
    R = Symmetric(Matrix{Float64}(I, (n + 1), (n + 1)))
    R[n+1,n+1] = n/4
    normR² = dot(R,R)
    Y = undef
    lowerbound = -Inf
    ν = undef
    μ = undef
    S = Symmetric(zeros(dim_S,dim_S))
    Z = undef
    Atν = Symmetric(zeros(dim_S, dim_S))
    Btμ = Symmetric(zeros(dim_S, dim_S))
    tmp = undef
    VtLV = V' * Lbig * V

    alpha_start = params.alpha_start; alpha_min = params.alpha_min; alphascale = params.alphascale;
    continue_iterations = true
    max_newcuts = params.max_newcuts; max_cutstotal = params.max_cutstotal; eps_triviol=params.eps_triviol;
    eps_tripurge = params.eps_tripurge
    min_newcuts_samealpha = params.min_newcuts_samealpha
    violated_triangles = Vector{TriangleIneq}()
    n_cuts = length(violated_triangles)
    n_newcuts = 0
    n_remcuts = 0
    purgecuts = true
    eps_correction = params.eps_correction
    
    it_addcuts = params.niterations_nocutsstart
    n_itsend = params.niterations_end
    extra_iterations = 0
    lastrounds = false

    lbfgsb_factr = params.lbfgsb_factr
    lbfgsb_factr_last = params.lbfgsb_factr_last
    factr_val = lbfgsb_factr
    lbfgsb_maxiter = params.lbfgsb_maxiter
    lbfgsb_m = params.lbfgsb_m

    iterations_outer = 0
    iterations_total = 0

    alpha = alpha_start
    xcur = fill(Cdouble(0), n_eqconstraints + n_cuts + len_vecS)
    lowerbounds = vcat(fill(-Inf, n_eqconstraints), fill(Cdouble(0), n_cuts + len_vecS))
    fopt = undef

    println("                                                             ┏━━━━━━━━━━━━━━━━━━━━━━━━┓                             ") 
    println("                                                             ┃        C  U  T  S      ┃                             ")
    println("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓") 
    println("┃    primal          dual            fopt          alpha     ┃  total   new   removed ┃   time_elapsed   iteration ┃")
    println("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┗━━━━━━━━━━━━━━━━━━━━━━━━┛━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛") 

    iterations_outer += 1
    iterations_inner = 0
    alpha = alpha_start
    while continue_iterations #alpha > alpha_min
        iterations_inner += 1
        iterations_total += 1
        f = x -> augmLagrangeFct_withtriangles(x, n, Lbig, V, normR², R, alpha, violated_triangles, diag_constraint=diag_constraint)
        fopt, xopt = lbfgsb(f, xcur, lb=lowerbounds, ub=Inf, m=lbfgsb_m, factr=factr_val, pgtol=1e-5, iprint=-1, maxiter=lbfgsb_maxiter)
        fopt = -fopt
        ν = xopt[1:n_eqconstraints]
        get_Aadjointnu!(parent(Atν), ν, n, diag_constraint=diag_constraint)
        μ = xopt[(n_eqconstraints + 1):(n_eqconstraints + n_cuts)]
        get_Badjointmu!(parent(Btμ), μ, violated_triangles, n, diag_constraint=diag_constraint)
        vec2mat!(parent(S), xopt[(n_eqconstraints + n_cuts + 1):end])
        xcur = xopt
        tmp =  V' * (Lbig - Atν + Btμ - S) * V
        U, ev = projection_PSD_cone(tmp - alpha * R)
        Z = U * Diagonal(ev) * U'
        tmp2 = 1/alpha * (tmp - Z)
        axpy!(-1, tmp2, parent(R)) # R = R + 1 * tmp
        normR² = dot(R,R)
        # alpha *= alphascale
        time_elapsed_s = Dates.value(Millisecond(now() - start_time)) / 10^3
        primal = dot(VtLV, R)
        dual = ν[1]
        # search for new violated triangles
        Y = Symmetric(V*R*V')
        if iterations_inner ≥ it_addcuts
            max_newcuts_cur = lastrounds ? 0 : max(0, min(max_newcuts, max_cutstotal - length(violated_triangles)))
            n_remcuts = purgecuts ? remove_violatedtriangles!(violated_triangles, xcur, lowerbounds, n_eqconstraints, eps_purge=eps_tripurge) : 0
            n_newcuts = add_newviolatedtriangles!(violated_triangles, Y, max_newcuts_cur, 1, n, eps_viol=eps_triviol)
            @printf("%12.7f    %12.7f    %12.7f    %12.10f   %5d   %4d   %4d       %10.2f s    %5d\n",
                    primal, dual, fopt, alpha, n_cuts, n_newcuts, n_remcuts, time_elapsed_s, iterations_inner)

            if n_newcuts > 0
                xcur = [xcur[1:(n_eqconstraints + n_cuts)]; zeros(n_newcuts); xcur[(n_eqconstraints + n_cuts + 1):end]]
                append!(lowerbounds, zeros(n_newcuts))
            end
            n_cuts = length(violated_triangles)
            if n_newcuts < min_newcuts_samealpha
                # reduce alpha
                if alpha * alphascale < alpha_min
                    lastrounds = true
                else
                    alpha *= alphascale
                end
            end
        else
            @printf("%12.7f    %12.7f    %12.7f    %12.10f   %5d                     %10.2f s    %5d\n",
                    primal, dual, fopt, alpha, n_cuts, time_elapsed_s, iterations_inner)
            
            if alpha * alphascale < alpha_min
                lastrounds = true
            else
                alpha *= alphascale
            end
        end
        if lastrounds
            # check valid lower bound
            lb_tmp = compute_safedualbound(tmp, V, ν[1], ubλmaxY)   
            lb_correction = ν[1] - lb_tmp
            if lb_tmp > lowerbound
                lowerbound = lb_tmp
            end
            # stop if correction is small enough or max number of iterations at the end reached
            if lb_correction < eps_correction || extra_iterations >= n_itsend    
                continue_iterations = false
            else
                extra_iterations += 1
                factr_val = lbfgsb_factr_last
                if params.nopurge_lastrounds || n_itsend == 0 # no need to purge cuts in last iteration
                    purgecuts = false
                end
            end
        end
    end
    time_total_s = Dates.value(Millisecond(now() - start_time)) / 10^3
    n_cuts = length(violated_triangles) - n_newcuts
    println("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓") 
    @printf("┃   DNN lower bound:               %17.8f                                                               ┃\n", lowerbound)
    @printf("┃   time:                          %12.3f s                                                                  ┃\n", time_total_s)
    @printf("┃   iterations:                    %8d                                                                        ┃\n", iterations_total)
    @printf("┃   added cuts:                    %8d                                                                        ┃\n", n_cuts)
    println("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛") 

    results = Dict{String, Any}("lb" => lowerbound, "time-wc" => time_total_s,
                                "iterations" => iterations_total, "ncuts" => n_cuts, "foptlast" => fopt)
    results["Y"] = Y;
    results["R"] = R;
    results["nu"] = ν;
    results["Z"] = Z;
    results["S"] = S;
    results["iterations_extra"] = extra_iterations;

    return results
end

end
