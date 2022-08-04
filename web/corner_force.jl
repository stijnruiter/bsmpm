using Plots
include("../src/simulation.jl")


L = 1
H = L
nix = 11
niy = 11
nppc = 4
degree = 2
E = 1e2
ρ = 1e3
Δt = 1e-3
tN = 2500
tend = Δt * tN
τ = 1
α = (L*τ)/(ρ * π)
ν = 0.3
corners = [ (0.0, 0.0),
            (L, 0.0),
            (L, H),
            (0.0, H)]

g = 40
disp_logger = zeros(tN+1, 3)
disp_logger_index = 1
function corner_force(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::AbstractBasisSplineStorage) where np
    global L, τ, α, traction_storage, oppY1, g, corner_particles, disp_logger, disp_logger_index

    disp_logger[disp_logger_index, 1] = t
    disp_logger[disp_logger_index, 2] = particles.displacement[end, 1]
    disp_logger[disp_logger_index, 3] = particles.displacement[end, 2]
    disp_logger_index = disp_logger_index + 1
    # current_topright_corner_particle = (particles.position[end, :])'#.+sqrt(particles.volume[end])/2)'
    # F = 0.1 * particles.ρ[end]
    # compute_bspline_values!(traction_storage, current_topright_corner_particle, mpmgrid.splines)
    # if model.runparams.spline == :webspline
    #     # Alter
    #     web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    # end
    # splines

    ue = 0.1;
    c = sqrt(model.E/model.ρ)
    T = exp(-c*t/L)
    lame = compute_lame_parameters(model)
    xinit = particles.position[:, 1] - particles.displacement[:, 1]
    yinit = particles.position[:, 2] - particles.displacement[:, 2]
    # bx = (-(2*(1-exp(-c*t/L))*ue*xinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*xinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit .*yinit.^2/L^4
    # by = (-(2*(1-exp(-c*t/L))*ue*yinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*yinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit.^2 .*yinit/L^4

    bx = (-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * xinit./particles.ρ - c^2*(T)*ue*xinit.* yinit.^2/L^4
    by = (-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * yinit./particles.ρ - c^2*(T)*ue*xinit.^2 .*yinit/L^4
    return -splines.B' * (particles.mass .* [bx by])
end
function corner_force_corner(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::AbstractBasisSplineStorage) where np
    global L, τ, α, traction_storage, oppY1, g, corner_particles, disp_logger, disp_logger_index

    disp_logger[disp_logger_index, 1] = t
    disp_logger[disp_logger_index, 2] = particles.displacement[end, 1]
    disp_logger[disp_logger_index, 3] = particles.displacement[end, 2]
    disp_logger_index = disp_logger_index + 1
    current_topright_corner_particle = (particles.position[end, :])'#.+sqrt(particles.volume[end])/2)'
    F = 1
    compute_bspline_values!(traction_storage, current_topright_corner_particle, mpmgrid.splines)
    if model.runparams.spline == :webspline
        # Alter
        web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    end
    # splines

    # ue = 0.01;
    # c = sqrt(model.E/model.ρ)
    # T = (1-exp(-c*t/L))/L^2
    # lame = compute_lame_parameters(model)
    # xinit = particles.position[:, 1] - particles.displacement[:, 1]
    # yinit = particles.position[:, 2] - particles.displacement[:, 2]
    # # bx = (-(2*(1-exp(-c*t/L))*ue*xinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*xinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit .*yinit.^2/L^4
    # # by = (-(2*(1-exp(-c*t/L))*ue*yinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*yinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit.^2 .*yinit/L^4

    # dux_dt = -c^2/L^4*exp(-c*t/L)*ue*xinit.*yinit.^2
    # duy_dt = -c^2/L^4*exp(-c*t/L)*ue*yinit.*xinit.^2

    # dσ_x = 8*T^2*ue^2*xinit.^2 .*yinit*lame.μ + ((2*T*ue*xinit .* (T*ue*yinit.^2 .+1) - 8 * T^2*ue^2*xinit .* yinit.^2)*lame.λ) ./ ((T*ue*xinit.^2 .+1).*(T*ue*yinit.^2 .+ 1) - 4*T^2*ue^2 * xinit.^2 .* yinit.^2)
    # dσ_y = 8*T^2*ue^2*yinit.^2 .*xinit*lame.μ + ((2*T*ue*yinit .* (T*ue*xinit.^2 .+1) - 8 * T^2*ue^2*yinit .* xinit.^2)*lame.λ) ./ ((T*ue*xinit.^2 .+1).*(T*ue*yinit.^2 .+ 1) - 4*T^2*ue^2 * xinit.^2 .* yinit.^2)


    # bx = dux_dt .- dσ_x ./ particles.ρ # (-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * xinit./particles.ρ - c^2*T*ue*xinit.* yinit.^2/L^4
    # by = duy_dt .- dσ_y ./ particles.ρ #(-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * yinit./particles.ρ - c^2*T*ue*xinit.^2 .*yinit/L^4
    return traction_storage.B' * (particles.mass[end] .* [F F])
end

function corner_force_deform(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::AbstractBasisSplineStorage) where np
    global L, τ, α, traction_storage, oppY1, g, corner_particles, disp_logger, disp_logger_index

    disp_logger[disp_logger_index, 1] = t
    disp_logger[disp_logger_index, 2] = particles.displacement[end, 1]
    disp_logger[disp_logger_index, 3] = particles.displacement[end, 2]
    disp_logger_index = disp_logger_index + 1
    # current_topright_corner_particle = (particles.position[end, :])'#.+sqrt(particles.volume[end])/2)'
    # F = 0.1 * particles.ρ[end]
    # compute_bspline_values!(traction_storage, current_topright_corner_particle, mpmgrid.splines)
    # if model.runparams.spline == :webspline
    #     # Alter
    #     web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    # end
    # splines

    ue = 0.01;
    c = sqrt(model.E/model.ρ)
    T = (1-exp(-c*t/L))/L^2
    lame = compute_lame_parameters(model)
    xinit = particles.position[:, 1] - particles.displacement[:, 1]
    yinit = particles.position[:, 2] - particles.displacement[:, 2]
    # bx = (-(2*(1-exp(-c*t/L))*ue*xinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*xinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit .*yinit.^2/L^4
    # by = (-(2*(1-exp(-c*t/L))*ue*yinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*yinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit.^2 .*yinit/L^4

    dux_dt = -c^2/L^4*exp(-c*t/L)*ue*xinit.*yinit.^2
    duy_dt = -c^2/L^4*exp(-c*t/L)*ue*yinit.*xinit.^2

    dσ_x = 8*T^2*ue^2*xinit.^2 .*yinit*lame.μ + ((2*T*ue*xinit .* (T*ue*yinit.^2 .+1) - 8 * T^2*ue^2*xinit .* yinit.^2)*lame.λ) ./ ((T*ue*xinit.^2 .+1).*(T*ue*yinit.^2 .+ 1) - 4*T^2*ue^2 * xinit.^2 .* yinit.^2)
    dσ_y = 8*T^2*ue^2*yinit.^2 .*xinit*lame.μ + ((2*T*ue*yinit .* (T*ue*xinit.^2 .+1) - 8 * T^2*ue^2*yinit .* xinit.^2)*lame.λ) ./ ((T*ue*xinit.^2 .+1).*(T*ue*yinit.^2 .+ 1) - 4*T^2*ue^2 * xinit.^2 .* yinit.^2)


    bx = dux_dt .- dσ_x ./ particles.ρ # (-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * xinit./particles.ρ - c^2*T*ue*xinit.* yinit.^2/L^4
    by = duy_dt .- dσ_y ./ particles.ρ #(-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * yinit./particles.ρ - c^2*T*ue*xinit.^2 .*yinit/L^4
    return -splines.B' * (particles.mass .* [bx by])
end



function corner_force_sin(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::AbstractBasisSplineStorage) where np
    global L, τ, α, traction_storage, oppY1, g, corner_particles, disp_logger, disp_logger_index

    disp_logger[disp_logger_index, 1] = t
    disp_logger[disp_logger_index, 2] = particles.displacement[end, 1]
    disp_logger[disp_logger_index, 3] = particles.displacement[end, 2]
    disp_logger_index = disp_logger_index + 1
    # current_topright_corner_particle = (particles.position[end, :])'#.+sqrt(particles.volume[end])/2)'
    # F = 0.1 * particles.ρ[end]
    # compute_bspline_values!(traction_storage, current_topright_corner_particle, mpmgrid.splines)
    # if model.runparams.spline == :webspline
    #     # Alter
    #     web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    # end
    # splines

    ue = 0.1 * L;
    c = sqrt(model.E/model.ρ)
    T = (1-exp(-c*t/L))/L^2
    lame = compute_lame_parameters(model)
    xinit = particles.position[:, 1] - particles.displacement[:, 1]
    yinit = particles.position[:, 2] - particles.displacement[:, 2]

    sinux = ue*sin.(pi*xinit / (2*L)) .* sin.(pi*yinit / (2*L)) / L^2
    # sinuy = ue*sin(pi*xinit / (2*L)) .* sin(pi*yinit / (2*L)) / L^2
    
    cosux = ue*cos.(pi*xinit / (2*L)) .* cos.(pi*yinit / (2*L)) / L^2
    # cosuy = ue*cos(pi*xinit / (2*L)) .* cos(pi*yinit / (2*L)) / L^2

    
    internal_du = pi^2/(4*L^2) * (T*cosux - T*sinux)

    dσ_dx = internal_du*(lame.λ + lame.μ) - lame.μ*(pi^2*T*sinux/(2*L^2))
    du_dt = -c^2/L^2*exp(-c*t/L)*sinux

    bx = by = (du_dt - dσ_dx ./ particles.ρ)

    # bx = (-(2*(1-exp(-c*t/L))*ue*xinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*xinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit .*yinit.^2/L^4
    # by = (-(2*(1-exp(-c*t/L))*ue*yinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*yinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit.^2 .*yinit/L^4

    # dux_dt = -c^2/L^4*exp(-c*t/L)*ue*xinit.*yinit.^2
    # duy_dt = -c^2/L^4*exp(-c*t/L)*ue*yinit.*xinit.^2

    # dσ_x = 8*T^2*ue^2*xinit.^2 .*yinit*lame.μ + ((2*T*ue*xinit .* (T*ue*yinit.^2 .+1) - 8 * T^2*ue^2*xinit .* yinit.^2)*lame.λ) ./ ((T*ue*xinit.^2 .+1).*(T*ue*yinit.^2 .+ 1) - 4*T^2*ue^2 * xinit.^2 .* yinit.^2)
    # dσ_y = 8*T^2*ue^2*yinit.^2 .*xinit*lame.μ + ((2*T*ue*yinit .* (T*ue*xinit.^2 .+1) - 8 * T^2*ue^2*yinit .* xinit.^2)*lame.λ) ./ ((T*ue*xinit.^2 .+1).*(T*ue*yinit.^2 .+ 1) - 4*T^2*ue^2 * xinit.^2 .* yinit.^2)

    # dux_dt = lame.λ / (4*L^2) * () + lame.μ/(4*L^2) * 


    # bx = dux_dt .- dσ_x ./ particles.ρ # (-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * xinit./particles.ρ - c^2*T*ue*xinit.* yinit.^2/L^4
    # by = duy_dt .- dσ_y ./ particles.ρ #(-2*lame.λ-4*lame.μ) *(1-T)*ue/L^2 * yinit./particles.ρ - c^2*T*ue*xinit.^2 .*yinit/L^4
    return -splines.B' * ( [bx by])
end

mpmgrid = MPMGrid((0, L*1.5), nix, degree, (0, H*1.5), niy, degree)
# particles = initialize_uniform_particles(ρ, corners, (nix - 1) * nppc, (niy - 1) * nppc) 
particles = initialize_uniform_particles(ρ, corners, 100, 100) 
# corner_particles = (particles.position[:,1] .> 0.9) .& (particles.position[:, 2] .> 0.9)
dirichlet = get_boundary_indices(mpmgrid; fix_left = true, fix_bottom = true)
model = initialize_model_2D(ρ, E, ν, Δt, tN; 
        dirichlet = dirichlet, 
        body_force = corner_force_corner,
        constitutive_model = hooke_linear_elastic,
        runparams = RunParameters(true, false, false, :strain, :webspline))

fig=scatter(particles.position[:, 1], particles.position[:, 2])
scatter!(fig,particles.position[particles.bel, 1], particles.position[particles.bel, 2])
display(fig)

spline_storage = splines = initialize_spline_storage(particles, mpmgrid)
traction_storage = initialize_spline_storage(1, mpmgrid)
tend = run(model, particles, mpmgrid, spline_storage);

# initial_position = particles.position - particles.displacement
# xinit = reshape(initial_position[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
# yinit = reshape(initial_position[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
# displ = reshape(particles.displacement[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
# vmstress= 0.5 * ((particles.σ[:, 1] - particles.σ[:, 4]).^2 + 6 * particles.σ[:, 2].^2)
# stress = reshape(particles.σ[:, 1], (nix - 1) * nppc, (niy-1)*nppc)



initial_position = particles.position - particles.displacement
xinit = reshape(initial_position[:, 1], 100, 100)
yinit = reshape(initial_position[:, 2], 100, 100)
displ = reshape(particles.displacement[:, 1], 100, 100)
vmstress= 0.5 * ((particles.σ[:, 1] - particles.σ[:, 4]).^2 + 6 * particles.σ[:, 2].^2)
stress = reshape(particles.σ[:, 1], 100, 100)

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
fig = plot(disp_logger[1:(end-1), 1], sqrt.(sum(disp_logger[1:(end-1), 2:3].^2, dims=2)), title="$(model.runparams.spline)")
t = 0:0.1:tend
xpos = particles.position[end, 1] - particles.displacement[end, 1]
ypos = particles.position[end, 2] - particles.displacement[end, 2]
ue = 0.001
a = ue*L
b = ue*H
ux = a*(1 .-exp.(-sqrt(E/ρ)*t/L))*sin(pi*xpos/(2*L)) .* sin(pi*ypos/(2*H))
uy = a*(1 .-exp.(-sqrt(E/ρ)*t/L))*sin(pi*xpos/(2*L)) .* sin(pi*ypos/(2*H))
mag = sqrt.(ux.^2+uy.^2)
plot!(fig, t, mag)
display(fig)

fig = plot(disp_logger[1:(end-1), 1], disp_logger[1:(end-1), 2:3], title="$(model.runparams.spline)")
# plot!(fig, t, ux)
display(fig)
