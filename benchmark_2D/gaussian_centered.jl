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
tN = 500
tend = Δt * tN
τ = 1
α = (L*τ)/(ρ * π)
ν = 0.3
u_0 = 0.005
lame = compute_lame_parameters(model)


function bivariate_guassian(position::AbstractMatrix, μ::Tuple{<:Real, <:Real}, σ::Tuple{<:Real, <:Real})
    bivariate_guassian(position[:, 1], position[:, 2], μ, σ)
end

function bivariate_guassian(x::AbstractVector, y::AbstractVector, μ::Tuple{<:Real, <:Real}, σ::Tuple{<:Real, <:Real})
    gaussian(x, μ[1], σ[1]) .* gaussian(y, μ[2], σ[2])
end

function gaussian(x::AbstractVector, μ::Real, σ::Real)
    1/σ*sqrt(2π) * exp.(-0.5*(x.-μ).^2/σ^2)
end

mpmgrid = MPMGrid((0, L), nix, degree, (0, H), niy, degree)

particles = initialize_uniform_particles(mpmgrid, ρ, (nix - 1) * nppc, (niy - 1) * nppc)
t = 0
# particles.velocity = [vx_analytic(particles.position[:, 1], particles.position[:, 2], t, u_0, E, ρ) vy_analytic(particles.position[:, 1], particles.position[:, 2], t, u_0, E, ρ)]
particles.σ[:, 1] = bivariate_guassian(particles.position, (L/2, H/2), (0.1, 0.1))
particles.σ[:, 4] = -bivariate_guassian(particles.position, (L/2, H/2), (0.1, 0.1))


dirichlet = get_boundary_indices(mpmgrid)#; fix_left=true, fix_right=true, fix_top=true, fix_bottom=true)
model = initialize_model_2D(ρ, E, ν, Δt, tN; dirichlet = dirichlet)#, body_force = deform_body_force)#, traction_force = axis_aligned_traction)
spline_storage = initialize_spline_storage(particles, mpmgrid)
tend = run(model, particles, mpmgrid, spline_storage);

initial_position = particles.position - particles.displacement
xinit = reshape(initial_position[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
yinit = reshape(initial_position[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
displx = reshape(particles.displacement[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
disply = reshape(particles.displacement[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
stressxx = reshape(particles.σ[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
stressyy = reshape(particles.σ[:, 4], (nix - 1) * nppc, (niy-1)*nppc)

# yinit = initial_position[:, 2]

x_true = 0:0.001:L
y_true = 0:0.001:H
ux_true = ux_analytic(x_true, y_true, tend, u_0, E, ρ)
uy_true = uy_analytic(x_true, y_true, tend, u_0, E, ρ)

# ux_true = σ_xx_analytic(x_true, y_true, tend, u_0, E, ρ, lame)
# uy_true = σ_yy_analytic(x_true, y_true, tend, u_0, E, ρ, lame)

# σ_true = σ_analytic.(x_true, tend, L, τ)
figux = plot(title="2D run gaussian X", xlabel="x [m]", ylabel="u_x [m]", legend=false, label="Analytic")
figuy = plot(title="2D run gaussian Y", xlabel="y [m]", ylabel="u_y [m]", legend=false, label="Analytic")
figsxx = plot(title="2D run gaussian σ_xx", xlabel="x [m]", ylabel="σ [Pa]", legend=false, label="Analytic")
figsyy = plot(title="2D run gaussian σ_yy", xlabel="y [m]", ylabel="σ [Pa]", legend=false, label="Analytic")
for i = 1:((niy - 1) * nppc)
    plot!(figux, xinit[:, i], displx[:, i], label="BSMPM at y=$(yinit[1,i])")
    plot!(figsxx, xinit[:, i], stressxx[:, i], label="BSMPM at y=$(yinit[1,i])")
end
for i = 1:((nix - 1)*nppc) 
    plot!(figuy, yinit[i, :], disply[i, :], label="BSMPM at y=$(xinit[i,1])")
    plot!(figsyy, xinit[:, i], stressyy[:, i], label="BSMPM at y=$(yinit[1,i])")
end

display(figux)
display(figuy)
display(figsxx)
display(figsyy)