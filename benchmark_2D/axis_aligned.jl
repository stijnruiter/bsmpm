using Plots
include("../src/simulation.jl")


L = 1
H = 1
nix = 11
niy = 11
nppc = 4
degree = 2
E = 100
ρ = E
Δt = 1e-4
tN = 100
tend = Δt * tN
τ = 1
α = (L*τ)/(ρ * π)
ν = 0.3
u_0 = 0.005
lame = compute_lame_parameters(model)

mpmgrid = MPMGrid((0, L), nix, degree, (0, H), niy, degree)

ux_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real) = u_0 * sin.(2*π*x) * sin(sqrt(E/ρ)*π*t)
uy_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real) = u_0 * sin.(2*π*y) * sin(sqrt(E/ρ)*π*t + π)

vx_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real) = u_0 * sqrt(E/ρ)*π * sin.(2*π*x) * cos(sqrt(E/ρ)*π*t)
vy_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real) = u_0 * sqrt(E/ρ)*π * sin.(2*π*y) * cos(sqrt(E/ρ)*π*t + π)

deform_xx_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real) = 1 .+ 2 * u_0 * π * cos.(2*pi*x) * sin(sqrt(E/ρ)*π*t)
deform_yy_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real) = 1 .+ 2 * u_0 * π * cos.(2*pi*y) * sin(sqrt(E/ρ)*π*t + π)

function σ_xx_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real, lame::LameParameters)
    Dxx = deform_xx_analytic(x, y, t, u_0, E, ρ)
    Dyy = deform_yy_analytic(x, y, t, u_0, E, ρ)
    return (lame.λ * log.(Dxx.*Dyy) + lame.μ * (Dxx.^2 .- 1)) ./ (Dxx .* Dyy)
end

function σ_yy_analytic(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, t::Real, u_0::Real, E::Real, ρ::Real, lame::LameParameters)
    Dxx = deform_xx_analytic(x, y, t, u_0, E, ρ)
    Dyy = deform_yy_analytic(x, y, t, u_0, E, ρ)
    return  (lame.λ * log.(Dxx.*Dyy) + lame.μ * (Dyy.^2 .- 1)) ./ (Dxx .* Dyy)
end

function deform_body_force(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::AbstractBasisSplineStorage) where np
    global u_0, lame

    initial_position = particles.position - particles.displacement
    
    Dxx = deform_xx_analytic(initial_position[:, 1], initial_position[:, 2], t, u_0, model.E, model.ρ)
    Dyy = deform_yy_analytic(initial_position[:, 1], initial_position[:, 2], t, u_0, model.E, model.ρ)

    ux = ux_analytic(initial_position[:, 1], initial_position[:, 2], t, u_0, model.E, model.ρ)
    uy = uy_analytic(initial_position[:, 1], initial_position[:, 2], t, u_0, model.E, model.ρ)


    gx = π^2 * ux .* (4*lame.μ/ρ.-E/ρ.-4*(lame.λ*(log.(Dxx.*Dyy).-1).-lame.μ)./(ρ*(Dxx.^2)))
    gy = π^2 * uy.* (4*lame.μ/ρ.-E/ρ.-4*(lame.λ*(log.(Dxx.*Dyy).-1).-lame.μ)./(ρ*(Dyy.^2)))

    return splines.B' * (particles.mass .* [gx gy])#[splines.B'*(particles.mass .* gx) splines.B'*(particles.mass .* gy)]
end

particles = initialize_uniform_particles(mpmgrid, ρ, (nix - 1) * nppc, (niy - 1) * nppc)
t = 0
particles.velocity = [vx_analytic(particles.position[:, 1], particles.position[:, 2], t, u_0, E, ρ) vy_analytic(particles.position[:, 1], particles.position[:, 2], t, u_0, E, ρ)]


dirichlet = get_boundary_indices(mpmgrid; fix_left=true, fix_right=true, fix_top=true, fix_bottom=true)
model = initialize_model_2D(ρ, E, ν, Δt, tN; 
        dirichlet = dirichlet, 
        body_force = deform_body_force,
        constitutive_model = hooke_linear_elastic_deform,
        runparams = RunParameters(false, true, false, :deformation))#, traction_force = axis_aligned_traction)
spline_storage = initialize_spline_storage(particles, mpmgrid)
tend = run(model, particles, mpmgrid, spline_storage);

initial_position = particles.position - particles.displacement
xinit = reshape(initial_position[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
yinit = reshape(initial_position[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
displx = reshape(particles.displacement[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
disply = reshape(particles.displacement[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
stress = reshape(particles.σ[:, 1], (nix - 1) * nppc, (niy-1)*nppc)

# yinit = initial_position[:, 2]

x_true = 0:0.001:L
y_true = 0:0.001:H
ux_true = ux_analytic(x_true, y_true, tend, u_0, E, ρ)
uy_true = uy_analytic(x_true, y_true, tend, u_0, E, ρ)

# ux_true = σ_xx_analytic(x_true, y_true, tend, u_0, E, ρ, lame)
# uy_true = σ_yy_analytic(x_true, y_true, tend, u_0, E, ρ, lame)

# σ_true = σ_analytic.(x_true, tend, L, τ)
figux = plot(x_true, ux_true, title="2D run axis aligned X", xlabel="x [m]", ylabel="u_x [m]", legend=false, label="Analytic")
figuy = plot(y_true, uy_true, title="2D run axis aligned X", xlabel="y [m]", ylabel="u_y [m]", legend=false, label="Analytic")

for i = 1:((niy - 1) * nppc)
    plot!(figux, xinit[:, i], displx[:, i], label="BSMPM at y=$(yinit[1,i])")
end
for i = 1:((nix - 1)*nppc) 
    plot!(figuy, yinit[i, :], disply[i, :], label="BSMPM at y=$(xinit[i,1])")
end

display(figux)
display(figuy)