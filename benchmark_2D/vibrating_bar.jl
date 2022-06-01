include("../src/simulation.jl")

using Plots

L = 1
H = 0.05
nix = 30
niy = 3
nppc = 4
degree = 1
E = 3e4;
ρ = 2e3;
ν = 0
v0 = 0.8

Δt = 1e-5
tN = 5000
tend = Δt * tN


mpmgrid = MPMGrid((0, L), nix, degree, (0, H), niy, degree)
particles = initialize_uniform_particles(mpmgrid, ρ, (ni -1) * nppc, (niy - 1) * nppc)
# mpmgrid = MPMGrid((0, L), ni, degree)
# particles = initialize_uniform_particles(mpmgrid, ρ, (ni - 1) * nppc)
dirichlet = get_boundary_indices(mpmgrid; fix_left = true, fix_right=true)#, fix_top=true, fix_bottom=true)
model = initialize_model_2D(ρ, E, ν, Δt, tN; 
            dirichlet = dirichlet,
            constitutive_model = hooke_large_deformation,
            runparams = RunParameters(true, false, false, :strain))
spline_storage = initialize_spline_storage(particles, mpmgrid)
compute_bspline_values!(spline_storage, particles.position, mpmgrid.splines)

particles.velocity[:, 1] = v0 * sin.(π * particles.position[:, 1] / L) 

tend = run(model, particles, mpmgrid, spline_storage);


initial_position = particles.position - particles.displacement
xinit = initial_position[:, 1]
yinit = initial_position[:, 2]


# particles are stored in tensor formation
idx = sortperm(xinit)
# idx =  kron(1:((niy-1)*nppc), 1:((nix-1)*nppc))

u_true = v0 / w * sin.(π * xinit / L) * sin(w * tend)
σ_true = v0 * E / sqrt(E / ρ) * sin(w * tend) * cos.(π * xinit / L)
figux = plot(xinit[idx], particles.displacement[idx, 1], title="2D run vibbar", xlabel="x [m]", ylabel="u_x [m]", label="BSMPM")
plot!(figux, xinit[idx], u_true[idx], label="Analytic")

figs = plot(xinit[idx], particles.σ[idx, 1], title="2D run vibbar", xlabel="x [m]", ylabel="σ_{xx} [Pa]", label="BSMPM")
plot!(figs, xinit[idx], σ_true[idx], label="Analytic")

figuy = plot(xinit[idx], particles.displacement[idx, 2], title="2D run vibbar", xlabel="x [m]", ylabel="u_y [m]", label="BSMPM")

figso = plot(xinit[idx], particles.σ[idx, 2:4], title="2D run vibbar", xlabel="x [m]", ylabel="σ [Pa]", labels=["σ_{xy}" "σ_{yx}" "σ_{yy}"])


display(figux)
display(figuy)
display(figs)
display(figso)