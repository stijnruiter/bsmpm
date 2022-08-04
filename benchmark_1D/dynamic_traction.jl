include("../src/simulation.jl")

using Plots

L = 1
ni = 51
nppc = 4
degree = 2
E = 100
ρ = E
Δt = 1e-4
tN = 5000
tend = Δt * tN
τ = 1
α = (L*τ)/(ρ * π)


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

function rhs_traction(t::Real, model::Model{1}, particles::Particles{1, np}, mpmgrid::MPMGrid{1}, splines::AbstractBasisSplineStorage) where np
    global L, τ, α, traction_storage
    current_rhs_boundary = L + u_analytic(L, t, L, α)
    current_rhs_traction = σ_analytic(L, t, L, τ)
    compute_bspline_values!(traction_storage, [current_rhs_boundary], mpmgrid.splines[1])
    if model.runparams.spline == :webspline
        # j = 52
        # e_ij = [0.5; -1.5; 2.0]
        # stable_supp = [49; 50; 51]
        # traction_storage.B[:, stable_supp] += traction_storage.B[:, j] * e_ij'
        web_splines!(traction_storage, mpmgrid.splines[1], particles.position[particles.bel] )
    end
    return traction_storage.B'*current_rhs_traction
end

mpmgrid = MPMGrid((0, L), ni, degree)
particles = initialize_uniform_particles(mpmgrid, ρ, (ni - 1) * nppc)
particles.bel .= 0
particles.bel[begin] = 1
particles.bel[end] = 1
mpmgrid = MPMGrid((0, L+u_analytic(L, 1, L, α)), ni, degree)
dirichlet = get_boundary_indices(mpmgrid; fix_left = true)#, fix_top=true, fix_bottom=true)
model = initialize_model_1D(ρ, E, Δt, tN; 
            dirichlet = dirichlet, 
            traction_force = rhs_traction,
            constitutive_model = hooke_linear_elastic_deform,
            runparams = RunParameters(true, false, false, :deformation, :webspline))
spline_storage = initialize_spline_storage(particles, mpmgrid)
traction_storage = initialize_spline_storage(1, mpmgrid.splines[1])
tend = run(model, particles, mpmgrid, spline_storage);


xinit = particles.position - particles.displacement
u_true = u_analytic.(xinit, tend, L, α)
σ_true = σ_analytic.(xinit, tend, L, τ)
figu = plot(xinit, particles.displacement, legend=:topleft)
plot!(figu, xinit, u_true)

figs = plot(xinit, particles.σ, legend=:topleft)
plot!(figs, xinit, σ_true)

display(figu)
display(figs)
