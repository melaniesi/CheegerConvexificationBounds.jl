using CheegerConvexificationBounds

using LinearAlgebra
using ForwardDiff
using Test

@testset "projection-psd" begin
    A = [1 0 0 0;
        0 -3 0 0;
        0 0 0 0
        0 0 0 4]
    B = copy(A); B[2,2] = 0
    U, ev = CheegerConvexificationBounds.projection_PSD_cone(A)
    @test U * Diagonal(ev) * U' ≈ B

    A = [-1 0 0 0;
        0 -3 0 0;
        0 0 -2 0
        0 0 0 -4]
    U, ev = CheegerConvexificationBounds.projection_PSD_cone(A)
    @test U * Diagonal(ev) * U' ≈ zeros(4,4)

    A = [1 0 0 0;
        0 3 0 0;
        0 0 2 0
        0 0 0 4]
    U, ev = CheegerConvexificationBounds.projection_PSD_cone(A)
    @test U * Diagonal(ev) * U' ≈ A
end


@testset "vec2mat" begin
    v = [1;1;2;1;2;3;1;2;3;4]
    S = Symmetric(zeros(4,4))
    CheegerConvexificationBounds.vec2mat!(parent(S), v)
    @test S == [1 1 1 1;
                1 2 2 2;
                1 2 3 3;
                1 2 3 4]
    CheegerConvexificationBounds.vec2mat!(parent(S), 1:10)
    @test S == [1 2 4  7;
                2 3 5  8;
                4 5 6  9;
                7 8 9 10]
end

@testset "add_newviolatedtriangles" begin
    Y = [0.5554221252973426 0.5085575306519106 0.8021925662586997 0.7649969733695833 0.1494820623075549 0.4320637264510716#=
    =#  0.6343951705436697 0.9499042743579864 0.3567050463587397 0.45574341331864543 0.0869838286684731; #=
    =#  0.9629880214886796 0.33925323743712155 0.2639978630561296 0.08437401827576363 0.6406363175152172 0.1907590915808347#=
    =#  0.852135680427663 0.5890887324089625 0.2714005979051255 0.6691241887965166 0.2823498321667868; #=
    =#  0.5038208722771225 0.5853170022132967 0.6551015148436167 0.11013037230809841 0.24231756772068058 0.8902155995224191#=
    =#  0.1802269970226429 0.22064871440108125 0.3977562186238984 0.19790860705185653 0.1760345419451752;#=
    =#  0.16932981179872553 0.22255963648558663 0.8031199496769714 0.7232249037077482 0.832131647767224 0.27031479349926424#=
    =#  0.32257732857674737 0.07377999866389873 0.13668027146533712 0.851386869637665 0.9551324424847997;#=
    =#  0.03779743195552954 0.407622070033528 0.5250182009719789 0.7642235890852082 0.2622560289659198 0.2418094107505746 #=
    =#  0.32049405867417635 0.1084828175822985 0.20736198538005923 0.3586291756576745 0.5312388131793415;#=
    =#  0.4738407703262314 0.13645637947587153 0.9315127670208145 0.38249235390126823 0.5022374709983656 0.2839698636761624#=
    =#  0.08076678907815438 0.14086576015883445 0.33640701467307976 0.5122530870396156 0.34372759892328464; #=
    =#  0.5809012268598549 0.4773352698184986 0.029272501307489374 0.9208211388003024 0.6438963977414329 0.8153053884018232#=
    =#  0.017582241517617003 0.20407078442578486 0.8234406149487259 0.7254436395441984 0.8198716316843712; #=
    =#  0.2427602674328101 0.1759740262655083 0.9576033819513267 0.8464012806018375 0.13513949970826966 0.1598980111250078#=
    =#  0.05311187124744188 0.36834059584317846 0.9331515187021692 0.21961701147910062 0.9441209455284574; #=
    =#  0.4749029392201338 0.8269450618871154 0.2529964861061701 0.9031177846807593 0.09735306502759411 0.486024729136993#=
    =#  0.36735918857738226 0.6880298441847156 0.9349552169632983 0.6417118592709101 0.4508924979933592; #=
    =#  0.6000832795781865 0.6800391948322017 0.27578586984226283 0.609640298795315 0.08176394089507277 0.7605969507083138#=
    =#  0.21238298094823493 0.2728967910665886 0.050029810560065835 0.12399368021414725 0.22937907753695264;#=
    =#  0.6000437463854386 0.9890150019957731 0.6046854604748492 0.5425824377587808 0.022513036919810325 0.20033564599930498#=
    =#  0.397187765771135 0.5475950118225494 0.9885251943453405 0.9137057893687682 0.9713501709811019]
    t1 = CheegerConvexificationBounds.TriangleIneq(1, 2, 3, 1)
    t2 = CheegerConvexificationBounds.TriangleIneq(1, 2, 4, 1)
    t3 = CheegerConvexificationBounds.TriangleIneq(1, 3, 4, 2)
    t4 = CheegerConvexificationBounds.TriangleIneq(2, 3, 4, 2)

    violated_triangles = Vector{CheegerConvexificationBounds.TriangleIneq}()
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 100)
    @test n_added == 4
    @test all(in(violated_triangles), [t1,t2,t3,t4])
    @test length(violated_triangles) == 4
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 100)
    @test n_added == 0

    violated_triangles = Vector{CheegerConvexificationBounds.TriangleIneq}()
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 3)
    @test n_added == length(violated_triangles) == 3
    @test all(in(violated_triangles), [t1,t2,t3])

    violated_triangles = Vector{CheegerConvexificationBounds.TriangleIneq}()
    push!(violated_triangles, CheegerConvexificationBounds.TriangleIneq(1,2,3,3))
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 100)
    @test n_added == 4
    @test length(violated_triangles) == 5

    violated_triangles = Vector{CheegerConvexificationBounds.TriangleIneq}()
    push!(violated_triangles, t1)
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 100)
    @test n_added == 3 && length(violated_triangles) == 4

    violated_triangles = Vector{CheegerConvexificationBounds.TriangleIneq}()
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 0)
    @test n_added == length(violated_triangles) == 0

    violated_triangles = Vector{CheegerConvexificationBounds.TriangleIneq}()
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 100, 2)
    @test n_added == 1
    @test t4 in violated_triangles

    violated_triangles = Vector{CheegerConvexificationBounds.TriangleIneq}()
    n_added = CheegerConvexificationBounds.add_newviolatedtriangles!(violated_triangles, Y, 100, 2, 3)
    @test n_added == 0
end

@testset "Aadjointnu" begin
    n = 10
    dim_mat = 2n + 3
    matrices_A = []
    # A1
    Ai = zeros(dim_mat,dim_mat)
    for i = 1:n
        Ai[i,end] = 1/2
        Ai[end,i] = 1/2
    end
    push!(matrices_A, Ai)

    for i = 1:n
        Ai = zeros(dim_mat,dim_mat)
        Ai[i,n+i] = 1/2
        Ai[n+i,i] = 1/2
        push!(matrices_A, Ai)
    end

    for i = 1:2*n
        Ai = zeros(dim_mat,dim_mat)
        Ai[i,end] = 1/2
        Ai[end,i] = 1/2
        Ai[i,i] = -1
        push!(matrices_A, Ai)
    end

    nu = rand(n + 1)
    Atnu = Symmetric(zeros(dim_mat,dim_mat))
    CheegerConvexificationBounds.get_Aadjointnu!(parent(Atnu), nu, n, diag_constraint=false)
    Atnu_2 = sum(nu .* matrices_A[1:(n+1)])
    @test Atnu == Atnu_2

    nu = rand(3*n + 1)
    CheegerConvexificationBounds.get_Aadjointnu!(parent(Atnu), nu, n, diag_constraint=true)
    Atnu_3 = sum(nu .* matrices_A)
    @test Atnu == Atnu_3
end

@testset "Badjointmu" begin
    n = 10
    dim_mat = 2n + 3

    violated_triangles = [CheegerConvexificationBounds.TriangleIneq(1,2,3,1),
                            CheegerConvexificationBounds.TriangleIneq(2,4,9,2),
                            CheegerConvexificationBounds.TriangleIneq(3,4,10,3)];

    B1 = zeros(dim_mat, dim_mat)
    B2 = copy(B1)
    B3 = copy(B1)
    B1[1,2] = B1[2,1] = B1[1,3] = B1[3,1] = 0.5;
    B1[2,3] = B1[3,2] = -0.5;
    B1[1,1] = B1[1,dim_mat] = B1[dim_mat,1] = -1.0/3;
    B2[2,4] = B2[4,2] = B2[4,9] = B2[9,4] = 0.5;
    B2[2,9] = B2[9,2] = -0.5;
    B2[4,4] = B2[4,dim_mat] = B2[dim_mat,4] = -1.0/3;
    B3[3,10] = B3[10,3] = B3[4,10] = B3[10,4] = 0.5;
    B3[3,4] = B3[4,3] = -0.5;
    B3[10,10] = B3[10,dim_mat] = B3[dim_mat,10] = -1.0/3;
    matrices_B = [B1, B2, B3]

    mu = [2,5,3]
    Btmu = sum(mu .* matrices_B)

    Btmu1 = Symmetric(zeros(dim_mat, dim_mat))
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu1), mu, violated_triangles, n, diag_constraint=true)
    @test Btmu1 ≈ Btmu

    Btmu2 = Symmetric(rand(dim_mat,dim_mat))
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu2), mu, violated_triangles, n, diag_constraint=true)
    @test Btmu2 ≈ Btmu

    Btmu3 = Symmetric(zeros(dim_mat, dim_mat))
    @test_throws AssertionError CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu3), mu, [], n, diag_constraint=true)
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu3), [], [], n)
    @test Btmu3 ≈ zeros(dim_mat, dim_mat)

    # diag_constraint = false
    B1 = zeros(dim_mat, dim_mat)
    B2 = copy(B1)
    B3 = copy(B1)
    B1[1,2] = B1[2,1] = B1[1,3] = B1[3,1] = 0.5;
    B1[2,3] = B1[3,2] = -0.5;
    B1[1,dim_mat] = B1[dim_mat,1] = -0.5;
    B2[2,4] = B2[4,2] = B2[4,9] = B2[9,4] = 0.5;
    B2[2,9] = B2[9,2] = -0.5;
    B2[4,dim_mat] = B2[dim_mat,4] = -0.5;
    B3[3,10] = B3[10,3] = B3[4,10] = B3[10,4] = 0.5;
    B3[3,4] = B3[4,3] = -0.5;
    B3[10,dim_mat] = B3[dim_mat,10] = -0.5;
    matrices_B = [B1, B2, B3]
    mu = [2,6,90]
    Btmu = sum(mu .* matrices_B)
    Btmu4 = Symmetric(zeros(dim_mat,dim_mat))
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu4), mu, violated_triangles, n, diag_constraint=false)
    @test Btmu4 ≈ Btmu
end

@testset "Aadjointnu" begin
    n = 10
    dim_mat = 2n + 3
    matrices_A = []
    # A1
    Ai = zeros(dim_mat,dim_mat)
    for i = 1:n
        Ai[i,end] = 1/2
        Ai[end,i] = 1/2
    end
    push!(matrices_A, Ai)

    for i = 1:n
        Ai = zeros(dim_mat,dim_mat)
        Ai[i,n+i] = 1/2
        Ai[n+i,i] = 1/2
        push!(matrices_A, Ai)
    end

    for i = 1:2*n
        Ai = zeros(dim_mat,dim_mat)
        Ai[i,end] = 1/2
        Ai[end,i] = 1/2
        Ai[i,i] = -1
        push!(matrices_A, Ai)
    end

    nu = rand(n + 1)
    Atnu = Symmetric(zeros(dim_mat,dim_mat))
    CheegerConvexificationBounds.get_Aadjointnu!(parent(Atnu), nu, n, diag_constraint=false)
    Atnu_2 = sum(nu .* matrices_A[1:(n+1)])
    @test Atnu == Atnu_2

    nu = rand(3*n + 1)
    CheegerConvexificationBounds.get_Aadjointnu!(parent(Atnu), nu, n, diag_constraint=true)
    Atnu_3 = sum(nu .* matrices_A)
    @test Atnu == Atnu_3
end

@testset "Badjointmu" begin
    n = 10
    dim_mat = 2n + 3

    violated_triangles = [CheegerConvexificationBounds.TriangleIneq(1,2,3,1),
                            CheegerConvexificationBounds.TriangleIneq(2,4,9,2), 
                            CheegerConvexificationBounds.TriangleIneq(3,4,10,3)];

    B1 = zeros(dim_mat, dim_mat)
    B2 = copy(B1)
    B3 = copy(B1)
    B1[1,2] = B1[2,1] = B1[1,3] = B1[3,1] = 0.5;
    B1[2,3] = B1[3,2] = -0.5;
    B1[1,1] = B1[1,dim_mat] = B1[dim_mat,1] = -1.0/3;
    B2[2,4] = B2[4,2] = B2[4,9] = B2[9,4] = 0.5;
    B2[2,9] = B2[9,2] = -0.5;
    B2[4,4] = B2[4,dim_mat] = B2[dim_mat,4] = -1.0/3;
    B3[3,10] = B3[10,3] = B3[4,10] = B3[10,4] = 0.5;
    B3[3,4] = B3[4,3] = -0.5;
    B3[10,10] = B3[10,dim_mat] = B3[dim_mat,10] = -1.0/3;
    matrices_B = [B1, B2, B3]

    mu = [2,5,3]
    Btmu = sum(mu .* matrices_B)

    Btmu1 = Symmetric(zeros(dim_mat, dim_mat))
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu1), mu, violated_triangles, n, diag_constraint=true)
    @test Btmu1 ≈ Btmu

    Btmu2 = Symmetric(rand(dim_mat,dim_mat))
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu2), mu, violated_triangles, n, diag_constraint=true)
    @test Btmu2 ≈ Btmu

    Btmu3 = Symmetric(zeros(dim_mat, dim_mat))
    @test_throws AssertionError CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu3), mu, [], n, diag_constraint=true)
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu3), [], [], n)
    @test Btmu3 ≈ zeros(dim_mat, dim_mat)

    # diag_constraint = false
    B1 = zeros(dim_mat, dim_mat)
    B2 = copy(B1)
    B3 = copy(B1)
    B1[1,2] = B1[2,1] = B1[1,3] = B1[3,1] = 0.5;
    B1[2,3] = B1[3,2] = -0.5;
    B1[1,dim_mat] = B1[dim_mat,1] = -0.5;
    B2[2,4] = B2[4,2] = B2[4,9] = B2[9,4] = 0.5;
    B2[2,9] = B2[9,2] = -0.5;
    B2[4,dim_mat] = B2[dim_mat,4] = -0.5;
    B3[3,10] = B3[10,3] = B3[4,10] = B3[10,4] = 0.5;
    B3[3,4] = B3[4,3] = -0.5;
    B3[10,dim_mat] = B3[dim_mat,10] = -0.5;
    matrices_B = [B1, B2, B3]
    mu = [2,6,90]
    Btmu = sum(mu .* matrices_B)
    Btmu4 = Symmetric(zeros(dim_mat,dim_mat))
    CheegerConvexificationBounds.get_Badjointmu!(parent(Btmu4), mu, violated_triangles, n, diag_constraint=false)
    @test Btmu4 ≈ Btmu
end


@testset "augmLagrangeFct" begin
    function augmLagrangeFct2_fval(x, n, Lbig, V, R, alpha; diag_constraint=true)
        # n = diag_constraint ? Int(-5 + sqrt(11 + 2 * length(x))) >> 1 : Int(sqrt(2*(1 + length(x))) - 4) >> 1
        nconstraints = diag_constraint ? 3*n + 1 : n + 1
        dim_S = 2*n + 3
        nu = x[1:nconstraints]
        Atnu = Symmetric(zeros(dim_S, dim_S))
        matrices_A = []
        # A1
        Ai = zeros(dim_S,dim_S)
        for i = 1:n
            Ai[i,end] = 1/2
            Ai[end,i] = 1/2
        end
        push!(matrices_A, Ai)
        for i = 1:n
            Ai = zeros(dim_S,dim_S)
            Ai[i,n+i] = 1/2
            Ai[n+i,i] = 1/2
            push!(matrices_A, Ai)
        end
        for i = 1:2*n
            Ai = zeros(dim_S,dim_S)
            Ai[i,end] = 1/2
            Ai[end,i] = 1/2
            Ai[i,i] = -1
            push!(matrices_A, Ai)
        end
        Atnu = sum(nu .* matrices_A[1:nconstraints])

        Smatrices = []
        for j = 1:dim_S
            for i = 1:j
                Si = zeros(dim_S,dim_S)
                Si[i,j] = 1
                Si[j,i] = 1
                push!(Smatrices, Si)
            end
        end
        S = sum(x[nconstraints + 1:end] .* Smatrices)
        
        M = V' * ((Atnu + S - Lbig) * V) + alpha * R
        _, ev = CheegerConvexificationBounds.projection_PSD_cone(Symmetric(M))
        # function value
        return -nu[1] + sum(ev.^2)/(2*alpha) - 0.5*alpha * dot(R,R)
    end

    Lbig = Symmetric(1/2 * [2 -1 -1 0 0 0 0 0 0 0 0;
                            -1 2 0 -1 0 0 0 0 0 0 0; 
                            -1 0 2 -1 0 0 0 0 0 0 0; 
                            0 -1 -1 2 0 0 0 0 0 0 0; 
                            0 0 0 0 2 -1 -1 0 0 0 0; 
                            0 0 0 0 -1 2 0 -1 0 0 0;
                            0 0 0 0 -1 0 2 -1 0 0 0; 
                            0 0 0 0 0 -1 -1 2 0 0 0; 
                            0 0 0 0 0 0 0 0 0 0 0; 
                            0 0 0 0 0 0 0 0 0 0 0; 
                            0 0 0 0 0 0 0 0 0 0 0])
    n = 4
    V = vcat(Matrix(I,n,n), -Matrix(I,n,n), -ones(1,n), ones(1,n), zeros(1,n))
    V = hcat(V, vcat(zeros(n), ones(n), floor(n/2), -1, 1))
    V = Matrix(qr(V).Q)
    R = Symmetric(Matrix{Float64}(I, (n + 1), (n + 1)))
    alpha = 1
    dim_S = 2*n + 3
    len_vecS = dim_S * (dim_S + 1) >> 1

    diag_constraint = false
    nconstraints = diag_constraint ? 3*n + 1 : n + 1
    xcur = fill(Cdouble(0), nconstraints + len_vecS)
    lowerbounds = vcat(fill(-Inf, nconstraints), fill(Cdouble(0), len_vecS))
    f = x -> CheegerConvexificationBounds.augmLagrangeFct(x, n, Lbig, V, dot(R,R), R, alpha, diag_constraint=diag_constraint)
    fxcur, gxcur = f(xcur)
    @test fxcur ≈ -3/2
    @test gxcur ≈ [-0.5555555555555556, 0.04166666666666663, 0.04166666666666655, 0.04166666666666666, 0.041666666666666755, 0.06944444444444448,
                    0.13888888888888898, 0.06944444444444452, 0.13888888888888892, 0.13888888888888895, 0.06944444444444445, 0.13888888888888887,
                    0.1388888888888889, 0.13888888888888884, 0.06944444444444439, 0.08333333333333326, 0.0833333333333332, 0.08333333333333331,
                    0.08333333333333338, 0.12500000000000008, 0.08333333333333318, 0.0833333333333331, 0.0833333333333332, 0.0833333333333333,
                    0.2500000000000001, 0.125, 0.08333333333333326, 0.08333333333333323, 0.08333333333333331, 0.0833333333333334, 0.25000000000000006,
                    0.24999999999999997, 0.12499999999999996, 0.08333333333333337, 0.08333333333333334, 0.08333333333333344, 0.08333333333333351,
                    0.25000000000000006, 0.24999999999999997, 0.24999999999999994, 0.12499999999999996, -0.11111111111111133, -0.11111111111111141,
                    -0.11111111111111112, -0.11111111111111091, 0.3333333333333337, 0.33333333333333365, 0.33333333333333337, 0.33333333333333315,
                    0.44444444444444464, 0.33333333333333354, 0.3333333333333336, 0.33333333333333337, 0.3333333333333332, -2.220446049250313e-16,
                    -3.885780586188048e-16, -5.551115123125783e-17, 2.7755575615628914e-16, -0.666666666666667, 0.5000000000000002, 0.2222222222222222,
                    0.2222222222222222, 0.22222222222222224, 0.22222222222222227, 0.3333333333333334, 0.33333333333333326, 0.3333333333333333,
                    0.3333333333333334, 0.22222222222222232, 0.33333333333333337, 0.2777777777777778] # computed with ForwardDiff
    # computation with ForwardDiff
    f2 = x -> augmLagrangeFct2_fval(x, n, Lbig, V, R, alpha, diag_constraint=diag_constraint)
    @test fxcur ≈ f2(xcur)
    @test gxcur ≈ ForwardDiff.gradient(f2, xcur)

    diag_constraint = true
    nconstraints = diag_constraint ? 3*n + 1 : n + 1
    xcur = fill(Cdouble(0), nconstraints + len_vecS)
    lowerbounds = vcat(fill(-Inf, nconstraints), fill(Cdouble(0), len_vecS))
    fxcur, gxcur = f(xcur)
    @test fxcur ≈ -3/2
    @test gxcur ≈ [-0.5555555555555556, 0.04166666666666663, 0.04166666666666655, 0.04166666666666666, 0.041666666666666755, 0.04166666666666663, 
                    0.04166666666666659, 0.041666666666666685, 0.04166666666666674, 0.04166666666666663, 0.04166666666666663, 0.041666666666666685, 
                    0.041666666666666755, 0.06944444444444448, 0.13888888888888898, 0.06944444444444452, 0.13888888888888892, 0.13888888888888895, 
                    0.06944444444444445, 0.13888888888888887, 0.1388888888888889, 0.13888888888888884, 0.06944444444444439, 0.08333333333333326, 
                    0.0833333333333332, 0.08333333333333331, 0.08333333333333338, 0.12500000000000008, 0.08333333333333318, 0.0833333333333331, 
                    0.0833333333333332, 0.0833333333333333, 0.2500000000000001, 0.125, 0.08333333333333326, 0.08333333333333323, 0.08333333333333331, 
                    0.0833333333333334, 0.25000000000000006, 0.24999999999999997, 0.12499999999999996, 0.08333333333333337, 0.08333333333333334, 
                    0.08333333333333344, 0.08333333333333351, 0.25000000000000006, 0.24999999999999997, 0.24999999999999994, 0.12499999999999996, 
                    -0.11111111111111133, -0.11111111111111141, -0.11111111111111112, -0.11111111111111091, 0.3333333333333337, 0.33333333333333365, 
                    0.33333333333333337, 0.33333333333333315, 0.44444444444444464, 0.33333333333333354, 0.3333333333333336, 0.33333333333333337, 
                    0.3333333333333332, -2.220446049250313e-16, -3.885780586188048e-16, -5.551115123125783e-17, 2.7755575615628914e-16, -0.666666666666667, 
                    0.5000000000000002, 0.2222222222222222, 0.2222222222222222, 0.22222222222222224, 0.22222222222222227, 0.3333333333333334, 
                    0.33333333333333326, 0.3333333333333333, 0.3333333333333334, 0.22222222222222232, 0.33333333333333337, 0.2777777777777778] # computed with ForwardDiff
    @test fxcur ≈ f2(xcur)
    @test gxcur ≈ ForwardDiff.gradient(f2, xcur)
end

@testset "augmLagrangeFct_withtriangles" begin
    function augmLagrangeFct2_fval(x, n, Lbig, V, R, alpha, violated_triangles; diag_constraint=true)
        # n = diag_constraint ? Int(-5 + sqrt(11 + 2 * length(x))) >> 1 : Int(sqrt(2*(1 + length(x))) - 4) >> 1
        nconstraints = diag_constraint ? 3*n + 1 : n + 1
        dim_S = 2*n + 3
        nu = x[1:nconstraints]
        Atnu = Symmetric(zeros(dim_S, dim_S))
        matrices_A = []
        # A1
        Ai = zeros(dim_S,dim_S)
        for i = 1:n
            Ai[i,end] = 1/2
            Ai[end,i] = 1/2
        end
        push!(matrices_A, Ai)
        for i = 1:n
            Ai = zeros(dim_S,dim_S)
            Ai[i,n+i] = 1/2
            Ai[n+i,i] = 1/2
            push!(matrices_A, Ai)
        end
        for i = 1:2*n
            Ai = zeros(dim_S,dim_S)
            Ai[i,end] = 1/2
            Ai[end,i] = 1/2
            Ai[i,i] = -1
            push!(matrices_A, Ai)
        end
        Atnu = sum(nu .* matrices_A[1:nconstraints])

        mu = x[(nconstraints + 1):(nconstraints + length(violated_triangles))]
        matrices_B = []
        if diag_constraint
            for t in violated_triangles
                i, j, k, type = t.i, t.j, t.k, t.type
                Bj = zeros(dim_S, dim_S)
                if type == 1
                    Bj[i,j] = Bj[j,i] = Bj[i,k] = Bj[k,i] = 0.5
                    Bj[j,k] = Bj[k,j] = -0.5
                    Bj[i,i] = Bj[i,end] = Bj[end,i] = -1.0/3
                elseif t.type == 2
                    Bj[i,j] = Bj[j,i] = Bj[j,k] = Bj[k,j] = 0.5
                    Bj[i,k] = Bj[k,i] = -0.5
                    Bj[j,j] = Bj[j,end] = Bj[end,j] = -1.0/3
                elseif t.type == 3
                    Bj[i,k] = Bj[k,i] = Bj[j,k] = Bj[k,j] = 0.5
                    Bj[j,i] = Bj[i,j] = -0.5
                    Bj[k,k] = Bj[k,end] = Bj[end,k] = -1.0/3
                end
                push!(matrices_B, Bj)
            end
        else
            for t in violated_triangles
                i, j, k, type = t.i, t.j, t.k, t.type
                Bj = zeros(dim_S, dim_S)
                if type == 1
                    Bj[i,j] = Bj[j,i] = Bj[i,k] = Bj[k,i] = 0.5
                    Bj[j,k] = Bj[k,j] = Bj[i,end] = Bj[end,i] = -0.5
                elseif t.type == 2
                    Bj[i,j] = Bj[j,i] = Bj[j,k] = Bj[k,j] = 0.5
                    Bj[i,k] = Bj[k,i] = Bj[j,end] = Bj[end,j] = -0.5
                elseif t.type == 3
                    Bj[i,k] = Bj[k,i] = Bj[j,k] = Bj[k,j] = 0.5
                    Bj[j,i] = Bj[i,j] = Bj[k,end] = Bj[end,k] = -0.5
                end
                push!(matrices_B, Bj)
            end
        end
        Btmu = sum(mu .* matrices_B)

        Smatrices = []
        for j = 1:dim_S
            for i = 1:j
                Si = zeros(dim_S,dim_S)
                Si[i,j] = 1
                Si[j,i] = 1
                push!(Smatrices, Si)
            end
        end
        S = sum(x[nconstraints + length(violated_triangles) + 1:end] .* Smatrices)
        
        M = V' * ((Atnu - Btmu + S - Lbig) * V) + alpha * R
        _, ev = CheegerConvexificationBounds.projection_PSD_cone(Symmetric(M))
        # function value
        return -nu[1] + sum(ev.^2)/(2*alpha) - 0.5*alpha * dot(R,R)
    end

    Lbig = Symmetric(1/2 * [2 -1 -1 0 0 0 0 0 0 0 0;
                            -1 2 0 -1 0 0 0 0 0 0 0; 
                            -1 0 2 -1 0 0 0 0 0 0 0; 
                            0 -1 -1 2 0 0 0 0 0 0 0; 
                            0 0 0 0 2 -1 -1 0 0 0 0; 
                            0 0 0 0 -1 2 0 -1 0 0 0;
                            0 0 0 0 -1 0 2 -1 0 0 0; 
                            0 0 0 0 0 -1 -1 2 0 0 0; 
                            0 0 0 0 0 0 0 0 0 0 0; 
                            0 0 0 0 0 0 0 0 0 0 0; 
                            0 0 0 0 0 0 0 0 0 0 0])
    n = 4
    V = vcat(Matrix(I,n,n), -Matrix(I,n,n), -ones(1,n), ones(1,n), zeros(1,n))
    V = hcat(V, vcat(zeros(n), ones(n), floor(n/2), -1, 1))
    V = Matrix(qr(V).Q)
    R = Symmetric(Matrix{Float64}(I, (n + 1), (n + 1)))
    alpha = 1
    dim_S = 2*n + 3
    len_vecS = dim_S * (dim_S + 1) >> 1

    violated_triangles = [CheegerConvexificationBounds.TriangleIneq(1,2,3,1),
                            CheegerConvexificationBounds.TriangleIneq(2,4,8,2),
                            CheegerConvexificationBounds.TriangleIneq(3,4,6,3)]

    diag_constraint = false
    nconstraints = diag_constraint ? 3*n + 1 : n + 1
    xcur = fill(Cdouble(0), nconstraints + length(violated_triangles) + len_vecS)
    lowerbounds = vcat(fill(-Inf, nconstraints), fill(Cdouble(0), length(violated_triangles) + len_vecS))
    f = x -> CheegerConvexificationBounds.augmLagrangeFct_withtriangles(x, n, Lbig, V, dot(R,R), R, alpha, violated_triangles, diag_constraint=diag_constraint)
    fxcur, gxcur = f(xcur)

    # computation with ForwardDiff
    f2 = x -> augmLagrangeFct2_fval(x, n, Lbig, V, R, alpha, violated_triangles, diag_constraint=diag_constraint)
    @test fxcur ≈ f2(xcur)
    @test gxcur ≈ ForwardDiff.gradient(f2, xcur)

    diag_constraint = true
    nconstraints = diag_constraint ? 3*n + 1 : n + 1
    xcur = fill(Cdouble(0), nconstraints + length(violated_triangles) + len_vecS)
    lowerbounds = vcat(fill(-Inf, nconstraints), fill(Cdouble(0), length(violated_triangles) + len_vecS))
    fxcur, gxcur = f(xcur)
    @test fxcur ≈ f2(xcur)
    @test gxcur ≈ ForwardDiff.gradient(f2, xcur)

    violated_triangles = []
    fxcur1, gxcur1 = CheegerConvexificationBounds.augmLagrangeFct_withtriangles(xcur, n, Lbig, V, dot(R,R), R, alpha, violated_triangles, diag_constraint=diag_constraint) 
    fxcur2, gxcur2 = CheegerConvexificationBounds.augmLagrangeFct(xcur, n, Lbig, V, dot(R,R), R, alpha, diag_constraint=diag_constraint)
    @test fxcur1 ≈ fxcur2
    @test gxcur1 ≈ gxcur2
end

@testset "remove_violatedtriangles" begin
    violated_triangles = [CheegerConvexificationBounds.TriangleIneq(1,2,3,1),
                CheegerConvexificationBounds.TriangleIneq(2,4,8,2), CheegerConvexificationBounds.TriangleIneq(3,4,6,3),
                CheegerConvexificationBounds.TriangleIneq(4,5,10,2), CheegerConvexificationBounds.TriangleIneq(5,6,9,2),
                CheegerConvexificationBounds.TriangleIneq(6,7,11,3)]

    n_eqconstraints = 4
    xcur = [collect(1:n_eqconstraints); 1; 2; 0; 3.5; 1e-12; 100; zeros(5)]
    lowerbounds = [fill(-Inf, n_eqconstraints); zeros(length(violated_triangles) + 5)]
    @test CheegerConvexificationBounds.remove_violatedtriangles!(violated_triangles, xcur, lowerbounds, n_eqconstraints, eps_purge=1e-8) == 2
    @test xcur ≈ [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.5, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test lowerbounds ≈ [-Inf, -Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test CheegerConvexificationBounds.TriangleIneq(3,4,6,3) ∉ violated_triangles
    @test CheegerConvexificationBounds.TriangleIneq(5,6,9,2) ∉ violated_triangles

    violated_triangles = [CheegerConvexificationBounds.TriangleIneq(1,2,3,1), CheegerConvexificationBounds.TriangleIneq(2,4,8,2), 
                CheegerConvexificationBounds.TriangleIneq(3,4,6,3), CheegerConvexificationBounds.TriangleIneq(4,5,10,2), 
                CheegerConvexificationBounds.TriangleIneq(5,6,9,2), CheegerConvexificationBounds.TriangleIneq(6,7,11,3)]

    xcur = [collect(1:n_eqconstraints); 1; 2; 3; 3.5; 10; 100; zeros(5)]
    lowerbounds = [fill(-Inf, n_eqconstraints); zeros(length(violated_triangles) + 5)]
    @test CheegerConvexificationBounds.remove_violatedtriangles!(violated_triangles, xcur, lowerbounds, n_eqconstraints, eps_purge=1e-8) == 0
    @test xcur ≈ [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 3.5, 10, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test lowerbounds ≈ [-Inf, -Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test length(violated_triangles) == 6

    xcur = [collect(1:n_eqconstraints); 1e-10; 1e-10; 1e-10; 1e-10; 1e-10; 1e-10; zeros(5)]
    @test CheegerConvexificationBounds.remove_violatedtriangles!(violated_triangles, xcur, lowerbounds, n_eqconstraints, eps_purge=1e-8) == 6
    @test xcur ≈ [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test lowerbounds ≈ [-Inf, -Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0]

    @test CheegerConvexificationBounds.remove_violatedtriangles!(violated_triangles, xcur, lowerbounds, n_eqconstraints, eps_purge=1e-8) == 0
    @test xcur ≈ [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test lowerbounds ≈ [-Inf, -Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0]
end
