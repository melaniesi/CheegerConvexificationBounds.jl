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

module BoundsMosek

using JuMP
using LinearAlgebra
using Mosek
using MosekTools

export SDPRelax_dim_2np3, SDPRelax_dim_np1


"""
     SDPRelax_dim_2np3(L)

Compute the SDP relaxation of dimension `2n + 3` with Mosek and return the objective,
the optimizer and the time needed for computation.

The SDP relaxation is
```
   min  0.5 (⟨L,Y11⟩ + ⟨L,Y22⟩)
   st.  e'y1 = 1
        tr(CYC' - Cyd' - dy'C' + ρdd') = 0
        diag(Y12) = 0
        ╭      ╮
        │ Y  y │
        │ y' ρ │ ∈ PSD(2n+3)
        ╰      ╯ 
            ╭                  ╮
            │Y11  Y12  Y13  Y14│
        Y = │Y12' Y22  Y23  Y24│ 
            |Y13' Y23' Y33  Y34|
            |Y14' Y24' Y34' Y44|
            ╰                  ╯
        y' = ( y1'  y2'  y3'  y4' )

            ╭                  ╮
            │ en'  0n'  1     0│
        C = │ en'  0n'  0    -1│ 
            | In   In   0n   0n|
            ╰                  ╯
        d' = ( ⌊n/2⌋   1   en' )

```

# Arguments:
- `L::Matrix`: the Laplacian matrix of the graph with `n` vertices

# Keyword Arguments:
- `dnn=false`: If `dnn=true`, then the constraints Y ≥ 0 and y ≥ 0 are added.
- `bqp_a=false`: If `bqp_a=true`, the redundant constraints Yij ≤ yi are added.
- `bqp_b=false`: If `bqp_b=true`, the constraints Yij + Yik - Yjk ≤ yi are added.
- `bqp_c=false`: If `bqp_c=true`, the redundant constraints yi + yj - Yij ≤ ρ are added.
- `bqp_d=false`: If `bqp_d=true`, the constraints yi + yj + yk - Yij - Yik - Yjk ≤ ρ are added.
- `diagcon=false`: If `diagcon=true`, the constraints diag(Y11) = y1 and diag(Y22) = y2 are added.
"""
function SDPRelax_dim_2np3(L; dnn=false, bqp_a=false, bqp_b=false, bqp_c=false, bqp_d=false, diagcon=false)
     n = size(L,1)

     m = Model(Mosek.Optimizer)
     Ybar = @variable(m, Ybar[1:(2 * n + 3),1:(2 * n + 3)], PSD)
     C = [ones(1,n) zeros(1,n)  1  0;
          ones(1,n) zeros(1,n)  0 -1;
          I(n) I(n) zeros(n,1) zeros(n,1)]
     d = [floor(n/2); 1; ones(n)]

     @objective(m, Min, dot(L, Ybar[2:(n + 1),2:(n + 1)]))

     @constraint(m, sum(Ybar[1,2:(n + 1)]) == 1)
     @constraint(m, tr(C*Ybar[2:end,2:end]*C' - C * Ybar[2:end,1] * d' - d * Ybar[2:end,1]' * C' + Ybar[1,1] * d * d') == 0)
     @constraint(m, [i=2:(n + 1)], Ybar[i,i+n] == 0)

     if diagcon
          @constraint(m, [i=2:(2 * n + 1)], Ybar[i,i] == Ybar[i,1]) # @constraint(m, diag(Ybar[2:(2 * n + 1),2:(2 * n + 1)]) .== Ybar[2:(2 * n + 1),1])
     end
     
     if dnn @constraint(m, [j=1:(2 * n + 3),i=(j + 1):(2 * n + 3)], Ybar[i,j] ≥ 0) end

     if bqp_a
          @constraint(m, [j=2:(2 * n + 1), i=(j + 1):(2 * n + 1)], Ybar[i,j] <= Ybar[i,i])
          @constraint(m, [j=2:(2 * n + 1), i=(j + 1):(2 * n + 1)], Ybar[i,j] <= Ybar[j,j])
     end
     if bqp_b
          # roles of i and j can be interchanged → set i < j
          @constraint(m, [l=2:(2 * n + 1), i=2:(2 * n + 1), j=(i + 1):(n + 1)],
                         Ybar[i,l] + Ybar[j,l] - Ybar[i,j] - Ybar[l,l] ≤ 0)
     end
     if bqp_c
          @constraint(m, [j=2:(2 * n + 1), i=(j + 1):(n + 1)],
                         Ybar[i,i] + Ybar[j,j] - Ybar[i,j] - Ybar[1,1] ≤ 0)
     end
     if bqp_d
          @constraint(m, [l=2:(2 * n + 1), j=(l + 1):(2 * n + 1), i=(j + 1):(n + 1)],
               Ybar[i,i] + Ybar[j,j] + Ybar[l,l] - Ybar[i,j] - Ybar[i,l] - Ybar[j,l] - Ybar[1,1] ≤ 0)
     end
     # @constraint(m, Ybar[2*n+2,2*n+3] == Ybar[2*n+2,2*n+2] - Ybar[1,2*n+2] * floor(n/2))
     optimize!(m)
     return objective_value(m), value.(Ybar), solve_time(m)
end


# Xbaropt = 1 / Ybaropt[1] * Ybaropt
"""
     SDPRelax_dim_np1(L)

Compute the basic SDP relaxation with Mosek and return the objective, the optimizer and the time needed for computation.

The basic SDP relaxation is
```
   min  ⟨L,Y⟩
   st.  e'y = 1
        1/⌊n/2⌋ ≤ ρ ≤ 1
        1 ≤ ⟨E,Y⟩ ≤ ⌊n/2⌋
        diag(Y) = y
        ╭     ╮
        │Y  y │
        │y' ρ │ ∈ PSD(n+1).
        ╰     ╯ 
```

# Arguments:
- `L::Matrix`: the Laplacian matrix of the graph with `n` vertices

# Keyword Arguments:
- `dnn=false`: If `dnn=true`, then the constraint Y ≥ 0 gets added.
- `bqp_a=false`: If `bqp_a=true`, the constraints Yij ≤ yi are added.
- `bqp_b=false`: If `bqp_b=true`, the constraints Yij + Yik - Yjk ≤ yi are added.
- `bqp_c=false`: If `bqp_c=true`, the constraints yi + yj - Yij ≤ ρ are added.
- `bqp_d=false`: If `bqp_d=true`, the constraints yi + yj + yk - Yij - Yik - Yjk ≤ ρ are added.

"""
function SDPRelax_dim_np1(L; dnn=false, bqp_a=false, bqp_b=false, bqp_c=false, bqp_d=false)
     n = size(L, 1)
     model = Model(Mosek.Optimizer)

     @variable(model, Y[1:n+1,1:n+1], PSD)

     @objective(model, Min, LinearAlgebra.dot(L,Y[2:n+1,2:n+1]))

     @constraint(model, 1 / (floor(n / 2)) ≤ Y[1,1] ≤ 1)
     @constraint(model, sum(Y[1,2:n+1]) == 1)
     @constraint(model, sum(Y[2:n+1,2:n+1]) <= floor(n/2))
     @constraint(model, LinearAlgebra.diag(Y[2:n+1,2:n+1]) .== Y[1,2:n+1])
     if dnn @constraint(model, [i=1:n+1, j=i+1:n+1], Y[i,j] >= 0) end
     if bqp_a
          @constraint(model, [i=2:n+1, j=i+1:n+1], Y[i,j] <= Y[i,i])
          @constraint(model, [i=2:n+1, j=i+1:n+1], Y[i,j] <= Y[j,j])
     end
     if bqp_b
          # roles of i and j can be interchanged → set i < j
          @constraint(model, [l=2:n+1, i=2:n+1, j=i+1:n+1],
                         Y[i,l] + Y[j,l] - Y[i,j] - Y[l,l] ≤ 0)
     end
     if bqp_c
          @constraint(model, [i=2:n+1, j=i+1:n+1],
                         Y[i,i] + Y[j,j] - Y[i,j] - Y[1,1] ≤ 0)
     end
     if bqp_d
          @constraint(model, [i=2:n+1, j=i+1:n+1, l=j+1:n+1],
               Y[i,i] + Y[j,j] + Y[l,l] - Y[i,j] - Y[i,l] - Y[j,l] - Y[1,1] ≤ 0)
     end
     optimize!(model)
     return objective_value(model), value.(Y), solve_time(model)
end




end # module