include("../src/simulation.jl")

using Plots

L = 1
ni = 50
nppc = 10
degree = 1
E = 3e4;
ρ = 2e3;
ν = 0
v0 = 0.8

Δt = 1e-5
tN = 5000
tend = Δt * tN


# mpmgrid = MPMGrid((0, L), ni, degree, (0, L), ni, degree)
# particles = initialize_uniform_particles(mpmgrid, ρ, 100, 100)
mpmgrid = MPMGrid((0, L), ni, degree)
particles = initialize_uniform_particles(mpmgrid, ρ, (ni - 1) * nppc)
dirichlet = get_boundary_indices(mpmgrid; fix_left = true, fix_right=true)#, fix_top=true, fix_bottom=true)
model = initialize_model_1D(ρ, E, Δt, tN; 
        dirichlet = dirichlet, 
        constitutive_model = hooke_linear_elastic_deform,
        runparams = RunParameters(true, false, false, :deformation))
spline_storage = initialize_spline_storage(particles, mpmgrid)
compute_bspline_values!(spline_storage, particles.position, mpmgrid.splines)


particles.velocity = v0 * sin.(π * particles.position / L) 

tend = run(model, particles, mpmgrid, spline_storage);


w = π * sqrt(E/ρ) / L
xinit = particles.position - particles.displacement
u_true = v0 / w * sin.(π * xinit / L) * sin(w * tend)
σ_true = v0 * E / sqrt(E / ρ) * sin(w * tend) * cos.(π * xinit / L)
figu = plot(xinit, particles.displacement)
plot!(figu, xinit, u_true)

figs = plot(xinit, particles.σ)
plot!(figs, xinit, σ_true)

display(figu)
display(figs)