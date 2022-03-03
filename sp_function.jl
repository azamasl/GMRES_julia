struct Saddle_Point
    # L = 0.5(x-xstar)'*B*(x-xstar) +(y-ystar)'*A*(x-xstar) -0.5(y-ystar)'*C*(y-ystar)
    B::Array{Float64,2} # PSD matrix
    A::Array{Float64,2}
    C::Array{Float64,2} # PSD matrix
    xstar::Array{Float64,1}
    ystar::Array{Float64,1}
    #∇F::Array{Float64,2} # PSD matrix
end

struct fun_return #a better name for this struc whould be alg_return
    x_sol::Array{Float64}
    y_sol::Array{Float64}
    iter::Int
    nfs::Array{Float64}
    val::Float64
    ng::Float64
end


struct ObjectiveFunction
    L::Function # objective function
    ∇xL::Function # (sub)gradient of objective
    ∇yL::Function # (sub)gradient of objective
    #∇F::Array{Float64,2}
    ∇F::Function
end

struct Obje_Fun2D
    L::Function # objective function
    ∇xL::Function # (sub)gradient of objective
    ∇yL::Function # (sub)gradient of objective
    ∇F::Function
end


"saddle-point function L(x,y)= x'Bx+y'Ax-y'Cy with (sub)gradient ∇L(x,y)."
function saddle_point_objective(sp::Saddle_Point)
    B, A, C, xstar, ystar = sp.B, sp.A, sp.C, sp.xstar, sp.ystar
    function L(x, y)
        return 0.5 * (x - xstar)' * B * (x - xstar) +
               (y - ystar)' * A * (x - xstar) -
               0.5 * (y - ystar)' * C * (y - ystar)
    end
    function ∇xL(x, y)
        return B * (x - xstar) + A' * (y - ystar)
    end
    function ∇yL(x, y)
        return A * (x - xstar) - C' * (y - ystar)
    end
    function ∇F(x, y)
        ∇F = [
            sp.B sp.A'
            -sp.A sp.C
        ]
        return ∇F
    end
    return ObjectiveFunction(L, ∇xL, ∇yL, ∇F)
end



## ########################## Resource Allocation problem
struct Res_Alloc #problem definition struct
    a::Array{Float64,1} #r.v. vector
    b::Array{Float64,1}
    c::Array{Float64,1}
    d::Array{Float64,1}
end

struct Obje_FunRA #return struct
    L::Function #Lagrangian
    ∇xL::Function
    ∇yL::Function
    ∇F::Function
    f::Function #objective function
end

function sp_RA_objective(sp::Res_Alloc)
    a, b, c, d = sp.a, sp.b, sp.c, sp.d
    # L = \sum f_i(x_i) + y*(\sum x_i -1)
    #f_i = 0.5*a_i*(x_i-c_i)^2 + log(1+ e^{b_i(x_i - d_i)})
    function f(x)
        return sum(0.5 .* a .* ((x .- c) .^ 2) .+ log1p.(exp.(b .* (x .- d))))
    end
    function L(x, y)
        "Note L is the lagrangian and we picked R.H.S. in constraint equal 0."
        return f(x) + y[1] * (sum(x))
    end
    function ∇xL(x, y)
        e2x_d = exp.(b .* (x .- d))
        return a .* (x .- c) .+ (b .* e2x_d) ./ (1 .+ e2x_d) .+ y .* ones(n, 1)
    end
    function ∇yL(x, y)
        ret = zeros(1, 1)
        ret[1, 1] = sum(x)
        return ret
    end
    function ∇F(x, y)
        e2x_d = exp.(b .* (x .- d))
        vones = ones(n, 1)
        di = a .+ (b .^ 2) .* e2x_d ./ ((1 .+ e2x_d) .^ 2)
        Lxx = Diagonal(di)
        return Matrix([
            Lxx vones
            -vones' zeros(1, 1)
        ])
    end
    return Obje_FunRA(L, ∇xL, ∇yL, ∇F, f)
end


## ####################################################

struct Analytic_Center_Problem #problem definition struct
    A::Array{Float64,2}
    b::Array{Float64,1}
end

struct Analytic_Center_ObjectiveFunGrad #return struct
    L::Function #Lagrangian
    ∇xL::Function
    ∇yL::Function
    ∇F::Function
    f::Function #objective function
end

function Analytic_Center_ObjeGrad(sp::Analytic_Center_Problem)
    A, b = sp.A, sp.b
    # f = -\sum log(x_i)
    # L = -\sum log(x_i) + y*(Ax-b)
    function f(x)
        return -sum(log.(x))
    end
    function L(x, y)
        return f(x) + y' * (A * x - b)
    end
    function ∇xL(x, y)
        return (-1 ./ x) .+ A' * y
    end
    function ∇yL(x, y)
        return A * x - b
    end
    function ∇F(x, y)
        xp2 = x .^ 2
        di = (1 ./ xp2)
        Lxx = Diagonal(di)        
        m = length(b)
        return Matrix([
            Lxx A'
            -A zeros(m, m)
        ])
    end
    return Analytic_Center_ObjectiveFunGrad(L, ∇xL, ∇yL, ∇F, f)
end




######################################################################################
#          Augxiliary functions
######################################################################################
function myCond(M)                 # Compute the condtion number as the ratio of numerically nonzero singularvalues
    U, s, V = svd(M)               # s is a vector and sorted descendingly: ORDER is important
    i = length(s)
    while s[i] < 1e-8
        i = i - 1
    end
    s1 = s[1]
    sn = s[i]
    conM = s1 / sn
end

"Generate random PSD matrix"
function random_PSD(N, scale)
    S = randn(N, N) / (N^0.25)
    S = scale * (S * S')                               #scaling S down to avoid overflow for larger matrices
    return S
end

"Generate random PD matrix "
function random_PD(N, scale)
    S = randn(N, N) / (N^0.25)
    S = scale * (S * S')                                #scaling S down to avoid overflow for larger matrices
    eig_vals, eig_vecs = eigen(S)
                                                        #adding 1e-6 to small eigenvalues to make sure S is PD
    eig_vals[findall(eig_vals .< 1E-6)] =
        eig_vals[findall(eig_vals .< 1E-6)] .+ 1E-6
    S = eig_vecs * Diagonal(eig_vals) * eig_vecs'
    return S
end

"Creats a random PD matrix with eigs = 1:N "
function my_special_random_square_matrix(N)
    S = randn(N, N)
    S = (S + S') / sqrt(2)
    eig_vals, eig_vecs = eigen(S)
    return eig_vecs * Matrix(Diagonal(1:N)) * eig_vecs'
end


function mysign(X)
    sig = zeros(size(X))
    for i = 1:length(sig)
        if X[i] > 0
            sig[i] = 1
        else
            sig[i] = -1
        end
    end
    return sig
end
