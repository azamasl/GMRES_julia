using JLD2, Random, LinearAlgebra
include("sp_function.jl")

ran_seed = 223;
                      # 0: random bilinear, 1: random quadratic Convex-Concave
prob_type = 1
eq_spaced = 1         # 1: creates a random problem with the equi-spaced eigenvals in the final hessian,
n, m      = 500, 500       # primal dual dimensions
scale     = 1e-1;         # scaling down the entires of the random matrices to avoid overflow
scaleDiag = 1e-1
ts = m + n            # total size
max_it =  ts+2
prt = 1               # 0 don't print grad norm at every iterations; 1: prints


function main()
                      # Creating the random problme instance
    Random.seed!(ran_seed)
    xstar = randn(n)
    ystar = randn(m)

    x0 = randn(n)
    y0 = randn(m)

    TYPE = ""
    B = zeros(n, n)
    C = zeros(m, m)
    A = random_scaled(m, n, scale)
    if prob_type == 1      #convex-concave problem
        println("The Problem is strongly convex-concave quadratic\n")
        if m == n && eq_spaced == 1
            println("Creat a Convex-Concave problem with equi-spaced eigen in A")
            diagInd = diagind(B)
            B[diagInd] .= 1                          #(n/2.0) #B is multiple of the I
            diagInd = diagind(C)
            C[diagInd] .= 1                          #(n/2.0) #C is multiple of the I
            A = my_special_random_square_matrix(m)   #A has equi-spaced eignevalues, from 1 to m, this makes nabF to have similar eigenvalues.

        else
            B = random_PD(n, scaleDiag)
            C = random_PD(m, scaleDiag)
        end
        TYPE = "Strongly convex-concave,"
    elseif prob_type == 0     #bilinear problem
        println("The Problem is bilinear")
        if m == n && eq_spaced == 1
            println("Creat a Bilinear problem with equi-spaced eigen A")
            A = my_special_random_square_matrix(m)
        end
        TYPE = "Bilinear,"
    else
        println("Unspecified problem type")
    end

    nbF = [
        B A'
        -A C
    ]
    @show nbFc0 = cond(Array(nbF), 2)
    @show nbFc = myCond(nbF)
    filename = "$prob_type-$ts-$eq_spaced"
    display(filename)
    @save "output/problem-" * filename * ".jld2" TYPE prob_type ran_seed max_it stepsize m n scale scaleDiag c1 A B C xstar ystar x0 y0  prt  nbFc ts
end

main()
