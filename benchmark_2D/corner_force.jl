using Plots
include("../src/simulation.jl")


L = 1
H = 1
nix = 10
niy = 10
nppc = 12
degree = 2
E = 1e2
ρ = 1e3
Δt = 1e-4
tN = 345
tend = Δt * tN
τ = 1
α = (L*τ)/(ρ * π)
ν = 0.4
corners = [ (0.0, 0.0),
            (L, 0.0),
            (L, H),
            (0.0, H)]

function corner_force(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::BasisSplineStorage) where np
    global L, τ, α, traction_storage, oppY
    current_topright_corner_particle = (particles.position[end, :].+sqrt(particles.volume[end])/2)'
    F = 40
    compute_bspline_values!(traction_storage, current_topright_corner_particle, mpmgrid.splines)
    return traction_storage.B' * [F F]
end

mpmgrid = MPMGrid((0, L*2), nix, degree, (0, H*2), niy, degree)
particles = initialize_uniform_particles(ρ, corners, (nix - 1) * nppc, (niy - 1) * nppc) 
dirichlet = get_boundary_indices(mpmgrid, fix_left=true, fix_bottom=true)
model = initialize_model_2D(ρ, E, ν, Δt, tN; 
        dirichlet = dirichlet, 
        traction_force = corner_force,
        constitutive_model = hooke_linear_elastic_deform,
        runparams = RunParameters(true, false, false, :deformation))
spline_storage = initialize_spline_storage(particles, mpmgrid)
traction_storage = initialize_spline_storage(1, mpmgrid)
tend = run(model, particles, mpmgrid, spline_storage);

initial_position = particles.position - particles.displacement
xinit = reshape(initial_position[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
yinit = reshape(initial_position[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
displ = reshape(particles.displacement[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
vmstress= 0.5 * ((particles.σ[:, 1] - particles.σ[:, 4]).^2 + 6 * particles.σ[:, 2].^2)
stress = reshape(particles.σ[:, 1], (nix - 1) * nppc, (niy-1)*nppc)

# # yinit = initial_position[:, 2]

# x_true = 0:0.001:L
# u_true = u_analytic.(x_true, tend, L, α)
# σ_true = σ_analytic.(x_true, tend, L, τ)
figu = plot(title="2D run corner ux", xlabel="x [m]", ylabel="u_x [m]", legend=false, label="Analytic")
figs = plot(title="2D run corner sxx", xlabel="x [m]", ylabel="σ_{xx} [Pa]", legend=false, label="Analytic")

for i = 1:size(xinit, 2)
    plot!(figu, xinit[:, i], displ[:, i], label="BSMPM at y=$(yinit[1,i])")
    plot!(figs, xinit[:, i], stress[:, i], label="BSMPM at y=$(yinit[1,i])")
end

magDisp = vec(sqrt.(sum(particles.displacement.^2, dims=2)))
figmagu = plot(particles.position[:, 1], particles.position[:, 2], magDisp, st=:surface, camera=(0,90))
fig_vonmis = plot(particles.position[:, 1], particles.position[:, 2], vmstress, st=:surface, camera=(0,90))

display(figu)
display(figs)
display(figmagu)
display(fig_vonmis)
display(scatter(particles.position[:, 1], particles.position[:, 2], legend=false, title="Particle positions as t=$(tend)s", xlabel="x [m]", ylabel="y [m]"))