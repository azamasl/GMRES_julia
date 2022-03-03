using Plots, JLD2, Printf
ymin, ymax = 1E-10, 1E5


#Loading the problem
filename = "1-1000-0"
prob = load("output/problem-" * filename * ".jld2")

TYPE = prob["TYPE"];
F_tol = prob["F_tol"]
lb = Dict()
lb["gmres"] = "GMRES";
lb["qr_gmres"] = "QR-GMRES";

if n == m                              #plot eigenvalus of nabF
    λ = eigvals(nabF)
    scatter(
        real.(λ),
        imag.(λ),
        markershape = :circle,
        markersize = 4,
        legend = false,
    )
    scatter!(
        xlabel = "Real part",
        ylabel = "Imaginary part",
        aspect_ratio = :equal,
    )
end

method_data = load("output/" * filename * "-gmres.jld2");
sol = method_data["sol"];
plot(
    range(1, size(sol["ress"], 1), step = 1),
    sol["ress"],
    yscale = :log10,
    label = lb["gmres"],
    color = :cyan,
)
ti = sol["time"];
println("GMRES time = $ti s");

method_data = load("output/" * filename * "-qr_gmres.jld2");
sol = method_data["sol"];
plot!(
    range(1, size(sol["nfs"], 1), step = 1),
    sol["nfs"],
    yscale = :log10,
    label = lb["qr_gmres"],
    color = :red,
)
ti = sol["time"];
println("QR-GMRES time = $ti s");


sF = @sprintf(", cond(∇F)= %2.1f", sol["nbFc"])
plot!(
    xlabel = "Iteration",
    ylabel = "||F||",
    title = string(TYPE, " m = $m, n=$n, sF", tol=$F_tol),
    titlefont = font(8),
    legend = :topright,
    legendfont = font(7),
    ylims = (ymin, ymax),
)


#savefig("plots/" * filename * ".png")
