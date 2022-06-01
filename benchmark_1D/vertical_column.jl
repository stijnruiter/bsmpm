include("../src/simulation.jl")

using Plots

L = 25
ni = 31
nppc = 2
degree = 2
E = 5e4;
ρ = 1;
g = 9.81
Δt = 1e-5
tN = 5000
tend = Δt * tN

function u_analytic(y::Real, t::Real, ρ::Real, g::Real, E::Real, H::Real; taylor_sum_n::Int = 500)
    u_true = (0.5*ρ*g* y.^2/E - ρ*g*H*y/E);
    for n = 1:taylor_sum_n
        un = 16*ρ*g*H^2/((4*n^2 - 4*n + 1)*(2*n - 1)*π^3*E)
        u_true += un*cos((√(E/ρ))*(2*n-1)*π*t/(2*H))*sin((2*n-1)*π*y/(2*H))
    end
    return u_true
end

function v_analytic(y::Real, t::Real, ρ::Real, g::Real, E::Real, H::Real; taylor_sum_n::Int = 500)
    v_true = (0.5*ρ*g* y^2/E - ρ*g*H*y/E);
    for n = 1:taylor_sum_n
        un = 16*ρ*g*H^2/((4*n^2 - 4*n + 1)*(2*n - 1)*π^3*E)
        v_true += (-(√(E/ρ))*(2*n-1)*π/(2*H))*un*sin((√(E/ρ))*(2*n-1)*π*t/(2*H))*sin.((2*n-1)*π*y/(2*H))
    end
    return v_true
end

function σ_analytic(y::Real, t::Real, ρ::Real, g::Real, E::Real, H::Real; taylor_sum_n::Int = 500)
    σ_true = (ρ*g* y - ρ*g*H);
    for n = 1:taylor_sum_n
        un = 16*ρ*g*H^2/((4*n^2 - 4*n + 1)*(2*n - 1)*π^3*E)
        σ_true += E*(2*n-1)*π/(2*H) * un*cos((√(E/ρ))*(2*n-1)*π*t/(2*H))*cos((2*n-1)*π*y/(2*H))
    end
    return σ_true
end

function graviation(t::Real, model::Model{1}, particles::Particles{1, np}, mpmgrid::MPMGrid{1}, splines::BasisSplineStorage) where np
    global g
    return splines.B'*(particles.mass .* -g)
end

mpmgrid = MPMGrid((0, L), ni, degree)
particles = initialize_uniform_particles(mpmgrid, ρ, (ni - 1) * nppc)
dirichlet = get_boundary_indices(mpmgrid; fix_left = true)#, fix_top=true, fix_bottom=true)
model = initialize_model_1D(ρ, E, Δt, tN; 
                    dirichlet = dirichlet, 
                    body_force = graviation,
                    constitutive_model = hooke_linear_elastic,
                    runparams = RunParameters(true, false, false, :strain))
spline_storage = initialize_spline_storage(particles, mpmgrid)
tend = run(model, particles, mpmgrid, spline_storage);


xinit = particles.position - particles.displacement
u_true = u_analytic.(xinit, tend, ρ, g, E, L)
σ_true = σ_analytic.(xinit, tend, ρ, g, E, L)
figu = plot(xinit, particles.displacement)
plot!(figu, xinit, u_true)

figs = plot(xinit, particles.σ)
plot!(figs, xinit, σ_true)

display(figu)
display(figs)