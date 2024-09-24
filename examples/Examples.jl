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

using CheegerConvexificationBounds
using CheegerConvexificationBounds.BoundsMosek

# Example with grevlex instance
L = CheegerConvexificationBounds.GrevlexGrlexLaplacian.grevlex(7);

# SDP relaxation of dimension n + 1
lowerbound1, Yopt1, time1 = SDPRelax_dim_np1(L, dnn=true, bqp_b=false);
lowerbound1

# SDP relaxation of dimension 2n + 3
lowerbound2, Yopt2, time2 = SDPRelax_dim_2np3(L, dnn=true, bqp_b=false, diagcon=true);
lowerbound2

# Lower bound from DNN-PFR solved with augmented Lagrangian approach
paramsdnn = Parameters(1, 1e-6, 0.6, 0, 0, Int(maxintfloat()), 500,
                       eps_triviol=1e-3, min_newcuts_samealpha=50, eps_tripurge=1e-5, eps_correction=0.01,
                       lbfgsb_maxiter=2000, lbfgsb_factr=1e8, lbfgsb_factr_last=1e8);
result3 = lowerboundCheegerConvexification(Lbig, paramsdnn; diag_constraint=false);
result3["lb"]
# Lower bound from DNN-PFRC solved with augmented Lagrangian approach
params = Parameters(1, 1e-6, 0.6, 500, 10000, 5, 500,
                    eps_triviol=1e-3, min_newcuts_samealpha=50, eps_tripurge=1e-5, eps_correction=0.001,
                    lbfgsb_maxiter=2000, lbfgsb_factr=1e8, lbfgsb_factr_last=1e8);
result4 = lowerboundCheegerConvexification(Lbig, params; diag_constraint=false);
result4["lb"]



# Example with instance from Rudy file
L = CheegerConvexificationBounds.RudyGraphIO.laplacian_from_RudyFile("rand01-7-32-0.dat");
result = lowerboundCheegerConvexification(Lbig, params; diag_constraint=false);
result["lb"]