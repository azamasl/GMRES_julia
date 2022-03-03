using CPUTime, JLD2, LinearAlgebra, Convex, Random, Plots
include("sp_function.jl")
include("myGMRES.jl")

#Loading the problem
# A problem with equaly spaced eigenvalues.
filename = "1-1000-0"
prob = load("output/problem-" * filename * ".jld2")

n = prob["n"];
m = prob["m"];
B = prob["B"];
C = prob["C"];
A = prob["A"];

xstar = prob["xstar"];
ystar = prob["ystar"];
x0 = prob["x0"];
y0 = prob["y0"];

sp = Saddle_Point(B, A, C, xstar, ystar)
obj = saddle_point_objective(sp)
nbFc = prob["nbFc"]
nabF = obj.∇F([], [])

zstar = [xstar; ystar]
rhs = nabF * zstar                     # The r.h.s in the linear system we need to solve (i.e. b in Ax=b)
rhs = rhs[:, :]                        # convert a vector to a matrix

prt = 0;
max_it = prob["max_it"];



function save_gmres()
    zfirst, ress, zrec = reg_gmres(nabF, rhs, [zeros(n,1);zeros(m,1)], max_it)        #[x0; y0]
    nfs = []
    for zk in eachcol(zrec)
        xk = zk[1:n];yk = zk[n+1:n+m];
        nff = LinearAlgebra.norm([obj.∇xL(xk, yk); -obj.∇yL(xk, yk)])
        append!(nfs, nff)
    end
    @save "output/" * filename * "-gmres.jld2" zlast nbFc ress rhs nfs
end


function save_qr_gmres()
    zlast,zrec = QR_GMRES(nabF, b) ;
    nfs = []
    for zk in eachcol(zrec)
        xk = zk[1:n];yk = zk[n+1:n+m];
        nff = LinearAlgebra.norm([obj.∇xL(xk, yk); -obj.∇yL(xk, yk)])
        append!(nfs, nff)
    end
    @save "output/" * filename * "-qr_gmres.jld2"  zlast zrec nfs  nbFc
end


function save_lssec()
    x_sol, y_sol, iter, nfs, val, ng =
        secant_inv(x0, y0, obj, dummy, sp, max_it, prt, 1, F_tol)
    @save "output/" * filename * "-lssec.jld2" x_sol y_sol iter nfs val ng nbFc F_tol max_it
end



tim= @elapsed save_gmres();
sol = load("output/"*filename*"-gmres.jld2");
sol["time"] = tim;
@save "output/"*filename*"-gmres.jld2" sol

tim = @elapsed save_lssec();
sol = load("output/" * filename * "-qr_gmres.jld2");
sol["time"] = tim;
@save "output/" * filename * "-qr_gmres.jld2" sol;
