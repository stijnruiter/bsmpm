using Plots
include("../src/simulation.jl")


L = 1
H = 0.05
nix = 21
niy = 3
nppc = 4
degree = 2
E = 100
ρ = E
Δt = 1e-4
tN = 5000
tend = Δt * tN
τ = 1
α = (L*τ)/(ρ * π)
ν = 0

function u_analytic(x::Real, t::Real, L::Real, α::Real) 
    u = 0;
    t = t % (4*L)
    ω = π/L
    if 0 <= t < (L-x)
        u = 0;
    elseif (L-x) <= t < (L+x)
        u = α*(1+cos(ω*(t+x)))
    elseif (L+x) <= t < (3*L-x)
        u = α*(cos(ω*(t+x)) - cos(ω*(t-x)))
    elseif (3*L-x) <= t < (3*L+x)
        u = α*(-1-cos(ω*(t-x)))
    elseif (3*L-x) <= t < (4*L)
        u = 0
    end
    return u
end

function σ_analytic(x::Real, t::Real, L::Real, τ::Real)
    σ = 0;
    t = t % (4*L)
    ω = π/L

    if 0 <= t < (L-x)
        σ = 0;
    elseif (L-x) <= t < (L+x)
        σ = -τ*sin(ω*(t+x))
    elseif (L+x) <= t < (3*L-x)
        σ = -τ*(sin(ω*(t+x)) + sin(ω*(t-x)))
    elseif (3*L-x) <= t < (3*L+x)
        σ = -τ*sin(ω*(t-x))
    elseif (3*L-x) <= t < (4*L)
        σ = 0
    end
    return σ
end

oppY = oppBspline(0, H, niy, degree)

function rhs_traction(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::AbstractBasisSplineStorage) where np
    global L, τ, α, traction_storage, oppY
    current_rhs_boundary = L + u_analytic(L, t, L, α)
    current_rhs_traction = σ_analytic(L, t, L, τ)
    compute_bspline_values!(traction_storage, [current_rhs_boundary], mpmgrid.splines[1])
    return [kron(oppY, traction_storage.B')*current_rhs_traction zeros(ndof(mpmgrid))]
end

mpmgrid = MPMGrid((0, L), nix, degree, (0, H), niy, degree)
particles = initialize_uniform_particles(mpmgrid, ρ, (nix - 1) * nppc, (niy - 1) * nppc) 
mpmgrid = MPMGrid((0, L + u_analytic(L, 1, L, α)), nix, degree, (0, H), niy, degree)
dirichlet = get_boundary_indices(mpmgrid; fix_left = true)#, fix_top=true, fix_bottom=true)
model = initialize_model_2D(ρ, E, ν, Δt, tN; 
        dirichlet = dirichlet, 
        traction_force = rhs_traction,
        constitutive_model = hooke_linear_elastic)
spline_storage = initialize_spline_storage(particles, mpmgrid)
traction_storage = initialize_spline_storage(1, mpmgrid.splines[1])
tend = run(model, particles, mpmgrid, spline_storage);

initial_position = particles.position - particles.displacement
xinit = reshape(initial_position[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
yinit = reshape(initial_position[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
displ = reshape(particles.displacement[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
stress = reshape(particles.σ[:, 1], (nix - 1) * nppc, (niy-1)*nppc)

# yinit = initial_position[:, 2]

x_true = 0:0.001:L
u_true = u_analytic.(x_true, tend, L, α)
σ_true = σ_analytic.(x_true, tend, L, τ)
figu = plot(x_true, u_true, title="2D run dyn traction", xlabel="x [m]", ylabel="u_x [m]", legend=:topleft, label="Analytic")
figs = plot(x_true, σ_true, title="2D run dyn traction", xlabel="x [m]", ylabel="σ_{xx} [Pa]", legend=:topleft, label="Analytic")

for i = 1:size(xinit, 2)
    plot!(figu, xinit[:, i], displ[:, i], label="BSMPM at y=$(yinit[1,i])")
    plot!(figs, xinit[:, i], stress[:, i], label="BSMPM at y=$(yinit[1,i])")
end

display(figu)
display(figs)