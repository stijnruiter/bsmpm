using Plots
include("../src/simulation.jl")
include("../src/thbspline_2d.jl")


L = 1
H = L
nix = 10
niy = 10
nppc = 30
degree = 2
E = 1e3
ρ = 1e3
Δt = 1e-2
tN = 100
tend = Δt * tN
τ = 1
α = (L*τ)/(ρ * π)
ν = 0.3
corners = [ (0.0, 0.0),
            (L, 0.0),
            (L, H),
            (0.0, H)]

ue=0.01
disp_logger = zeros(tN, 3)
error_logger = zeros(tN, 8)
error_logger_index = 1
disp_logger_index = 1



function analytic_displacement(model::Model{dim}, particles::Particles{dim, np}, t::Real, ue::Real, L::Real, H::Real) where {dim, np}
    return analytic_displacement(model, particles.position - particles.displacement, t, ue, L, H)
end
function analytic_displacement(model::Model{dim}, position::AbstractMatrix, t::Real, ue::Real, L::Real, H::Real) where {dim, np}
    c = sqrt(model.E / model.ρ)
    wx = c / L
    wy = c / H
    ux = ue * (1-exp(-wx*t)) * position[:, 1] .* (position[:, 2] / H).^2
    uy = ue * (1-exp(-wy*t)) * position[:, 2] .* (position[:, 1] / L).^2
    return [ux uy]
end

function analytic_velocity(model::Model{dim}, particles::Particles{dim, np}, t::Real, ue::Real, L::Real, H::Real) where {dim, np}
    return analytic_velocity(model, particles.position - particles.displacement, t, ue, L, H)
end
function analytic_velocity(model::Model{dim}, position::AbstractMatrix, t::Real, ue::Real, L::Real, H::Real) where {dim, np}
    c = sqrt(model.E / model.ρ)
    wx = c / L
    wy = c / H
    vx = ue * wx * exp(-wx*t) * position[:, 1] .* (position[:, 2] / H).^2
    vy = ue * wy * exp(-wy*t) * position[:, 2] .* (position[:, 1] / L).^2
    return [vx vy]
end


function analytic_stress(model::Model{dim}, particles::Particles{dim, np}, t::Real, ue::Real, L::Real, H::Real) where {dim, np}
    return analytic_stress(model, particles.position - particles.displacement, t, ue, L, H)
end
function analytic_stress(model::Model{dim}, position::AbstractMatrix, t::Real, ue::Real, L::Real, H::Real) where {dim, np}
    c = sqrt(model.E / model.ρ)
    wx = c / L
    wy = c / H

    lame = compute_lame_parameters(model)

    x = position[:, 1]
    y = position[:, 2]

    # s_xx = ((ue*(1-exp(-t*wx))*y.^2)/L^2+(ue*(1-exp(-t*wy))*x.^2)/L^2)*lame.λ+(2*lame.μ*ue*(1-exp(-t*wx))*y.^2)/L^2
    # s_xy = lame.μ*((2*ue*(1-exp(-t*wy)).*x.*y)/L^2+(2*ue*(1-exp(-t*wx)).*x.*y)/L^2)
    # s_yy = ((ue*(1-exp(-t*wx))*y.^2)/L^2+(ue*(1-exp(-t*wy))*x.^2)/L^2)*lame.λ+(2*lame.μ*ue*(1-exp(-t*wy))*x.^2)/L^2
    # return [s_xx s_xy s_xy s_yy]
    T = ue/L*(1-exp(-wx * t))
    P_xx = T*(lame.λ*x.^2 + (lame.λ+2*lame.μ)*y.^2)
    P_xy = 4*T*lame.μ*x.*y
    P_yy = T*((lame.λ+2*lame.μ)*x.^2+lame.λ*y.^2)

    P =  [P_xx P_xy P_xy P_yy]

    # return P

    F_xx = 1 .+T*y.^2
    F_xy = 2*T*x.*y
    F_yy = 1 .+T*x.^2

    J = F_xx.*F_yy-F_xy.^2

    return compute_matrix_product([F_xx F_xy F_xy F_yy], P) ./ J

    # return 
end

function corner_force(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::AbstractMPMGrid, splines::AbstractBasisSplineStorage) where np
    global L, H, ue, disp_logger, disp_logger_index

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

    c = sqrt(model.E / model.ρ)
    wx = c / L
    wy = c / H
    lame = compute_lame_parameters(model)
    x = particles.position[:, 1] - particles.displacement[:, 1]
    y = particles.position[:, 2] - particles.displacement[:, 2]
    lambda = lame.λ
    mu = lame.μ
    # bx = (-(2*(1-exp(-c*t/L))*ue*xinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*xinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit .*yinit.^2/L^4
    # by = (-(2*(1-exp(-c*t/L))*ue*yinit*lame.λ/L^2+4*lame.μ*(1-exp(-c*t/L))*ue*yinit/L^2)./particles.ρ) - c^2*exp(-c*t/L)*ue*xinit.^2 .*yinit/L^4
    rho = model.ρ

    # bx = -(2*ue*(1-exp(-t*wy))*x*lambda)/L^2-mu*((2*ue*(1-exp(-t*wy))*x)/L^2+(2*ue*(1-exp(-t*wx))*x)/L^2)-(rho*ue*wx^2*exp(-t*wx).*x .* y.^2)/L^2
    # by = -(2*ue*(1-exp(-t*wx))*y*lambda)/L^2-mu*((2*ue*(1-exp(-t*wy))*y)/L^2+(2*ue*(1-exp(-t*wx))*y)/L^2)-(rho*ue*wy^2*exp(-t*wy).*x.^2 .* y)/L^2

    b = -ue * wx^2/L*exp(-wx*t) * [(x.*y.^2) (x.^2 .* y)]-2*ue/L*(1-exp(-wx*t))*(lame.λ + 2*lame.μ)/model.ρ*[x y]
    return splines.B' * (particles.mass .* b)
end

function corner_traction_force(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::AbstractMPMGrid, splines::AbstractBasisSplineStorage) where np
    global traction_storage, L, H, ue

    n = nparticles(traction_storage)

    dp = 1/n
    p = range(dp/2, stop=(1-(dp/2)), length=n)

    # xt = range(0, stop=L, length=(n+1))
    # yt = range(0, stop=H, length=(n+1))
    # x = (xt[2:end] + xt[1:(end-1)]) / 2
    # y = (yt[2:end] + yt[1:(end-1)]) / 2
    x = L * p
    y = H * p
    dx = dp * L
    dy = dp * H

    bottom = [x zeros(n)]
    top = [x (H*ones(n))]
    left = [zeros(n) y]
    right = [(L*ones(n)) y]


    # bottom: n = (0, -1)
    stress = analytic_stress(model, bottom, t, ue, L, H)
    compute_thbspline_values!(traction_storage, bottom, mpmgrid)
    if model.runparams.spline == :webspline
        web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    end
    Ft_bottom = traction_storage.B' * (-[stress[:, 2] stress[:, 4]] .* dx)

    # left: n = (-1, 0)
    stress = analytic_stress(model, left, t, ue, L, H)
    compute_thbspline_values!(traction_storage, left, mpmgrid)
    if model.runparams.spline == :webspline
        web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    end
    Ft_left = traction_storage.B' * (-[stress[:, 1] stress[:, 3]] .* dy)

    #compute normal vector
    c = sqrt(model.E / model.ρ)
    wx = c / L
    T = ue * (1-exp(-wx*t))

    # top
    # n_length = sqrt.(1 .+ 2*T .+ (4*p.^2 .+ 1)*T^2)
    n1 = (-2 * T * p) #./ n_length
    n2 = (1 + T) #./ n_length

    # len = sqrt.(n1.^2 .+ n2.^2)
    # n1 = n1 ./ len
    # n2 = n2 ./ len

    # n1 = 0
    # n2 = 1

    # n_length .= 1
    stress = analytic_stress(model, top, t, ue, L, H)
    
    compute_thbspline_values!(traction_storage, top+analytic_displacement(model, top, t, ue, L, H), mpmgrid)
    if model.runparams.spline == :webspline
        web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    end
    Ft_top = traction_storage.B' * ([(n1.*stress[:, 1] + n2.*stress[:, 3]) (n1.*stress[:, 2] + n2.*stress[:, 4])] .* dx)

    # right
    stress = analytic_stress(model, right, t, ue, L, H)
    compute_thbspline_values!(traction_storage, right+analytic_displacement(model, right, t, ue, L, H), mpmgrid)
    if model.runparams.spline == :webspline
        web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    end
    Ft_right = traction_storage.B' * ([(n2.*stress[:, 1] + n1.*stress[:, 3]) (n2.*stress[:, 2] + n1.*stress[:, 4])] .* dy)

    return Ft_bottom + Ft_right + Ft_top + Ft_left+ penalty_dirichlet_force(t, model, particles, mpmgrid, splines)

end

function penalty_dirichlet_force(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::AbstractMPMGrid, splines::AbstractBasisSplineStorage) where np
    global L, H, ue, traction_storage
    
    β = 1e5

    nb = nparticles(traction_storage)
    ds = 1/nb
    s = range(ds/2, stop=(1-(ds/2)), length=nb)
    ui = zeros(ndof(splines), 2)
    M = splines.B' * (splines.B .* particles.mass)
    ui[splines.active, :] = M[splines.active, splines.active] \ (splines.B' * (particles.mass .* particles.displacement))[splines.active, :]

    dx = ds * L

    xb = s * L
    yb = zeros(nb)
    compute_thbspline_values!(traction_storage, [xb yb], mpmgrid)
    if model.runparams.spline == :webspline
        web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    end
    # F_bottom = β * traction_storage.B' * (traction_storage.B * (ones(ndof(traction_storage))))
    # F_bottom = β * traction_storage.B' * sum(traction_storage.B, dims=2)
    F_bottom = -β * traction_storage.B' * (traction_storage.B * (ui .* dx))
    # F_bottom[:, 1] .= 0
    
    xb = zeros(nb)
    yb = s * H
    compute_thbspline_values!(traction_storage, [xb yb], mpmgrid)
    if model.runparams.spline == :webspline
        web_splines!(traction_storage, mpmgrid, particles.position[particles.bel, :])
    end
    # F_left = β * traction_storage.B' * sum(traction_storage.B, dims=2)
    F_left = -β * traction_storage.B' * (traction_storage.B * (ui .* dx))
    # F_left[:, 2] .= 0
    # l
    
    return F_left + F_bottom
end



function error_analysis(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::AbstractMPMGrid, splines::AbstractBasisSplineStorage) where np
    global ue, L, H, error_values, error_logger, error_logger_index
    er_u = sum((particles.displacement - analytic_displacement(model, particles, t, ue, L, H)).^2, dims=1)
    er_v = sum((particles.velocity - analytic_velocity(model, particles, t, ue, L, H)).^2, dims=1)
    er_σ = sum((particles.σ - analytic_stress(model, particles, t, ue, L, H)).^2, dims=1)

    error_logger[error_logger_index, :] = [er_u er_v er_σ]
    error_logger_index += 1
end

npx = nix*nppc
npy = niy*nppc

Ω0 = Rect(0, 0, L+1*ue, H+1*ue)
Ω1 = [Rect(.5, .5, Ω0.ux, Ω0.uy)]
Ω2 = [Rect(.8, .8, Ω0.ux, Ω0.uy)]

mpmgrid = MPMGrid((Ω0.lx, Ω0.ux), nix, degree, (Ω0.ly, Ω0.uy), niy, degree)

domains = HierarchicalDomain2D(Ω0, Ω1, Ω2)
thb_grid = HierarchicalMPMGrid2D(mpmgrid, domains)

# particles = initialize_uniform_particles(ρ, corners, (nix - 1) * nppc, (niy - 1) * nppc) 
particles = initialize_uniform_particles(ρ, corners, npx, npy) 
# particles.bel .= 0
# particles.bel[(end-npx+1):end] .= 1
# particles.bel[npx:npx:end] .= 1
# corner_particles = (particles.position[:,1] .> 0.9) .& (particles.position[:, 2] .> 0.9)
dirichlet = get_boundary_indices(mpmgrid)#; fix_left = true, fix_bottom = true)
model = initialize_model_2D(ρ, E, ν, Δt, tN; 
        dirichlet = dirichlet, 
        body_force = corner_force,
        traction_force = corner_traction_force,
        constitutive_model = hooke_linear_elastic_deform,
        error = error_analysis,
        runparams = RunParameters(false, false, false, :deformation, :webspline))
particles.displacement = analytic_displacement(model, particles, 0, ue, L, H)
particles.velocity = analytic_velocity(model, particles, 0, ue, L, H)
particles.σ = analytic_stress(model, particles, 0, ue, L, H)

# fig=scatter(particles.position[:, 1], particles.position[:, 2])
# scatter!(fig,particles.position[particles.bel, 1], particles.position[particles.bel, 2])
# display(fig)

spline_storage = splines = initialize_spline_storage(nparticles(particles), thb_grid)
traction_storage = initialize_spline_storage(1000, thb_grid)
tend = run(model, particles, thb_grid, spline_storage);


# initial_position = particles.position - particles.displacement
# xinit = reshape(initial_position[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
# yinit = reshape(initial_position[:, 2], (nix - 1) * nppc, (niy-1)*nppc)
# displ = reshape(particles.displacement[:, 1], (nix - 1) * nppc, (niy-1)*nppc)
# vmstress= 0.5 * ((particles.σ[:, 1] - particles.σ[:, 4]).^2 + 6 * particles.σ[:, 2].^2)
# stress = reshape(particles.σ[:, 1], (nix - 1) * nppc, (niy-1)*nppc)



initial_position = particles.position - particles.displacement
xinit = reshape(initial_position[:, 1], npx, npy)
yinit = reshape(initial_position[:, 2], npx, npy)
displ = reshape(particles.displacement[:, 1], npx, npy)
vmstress= 0.5 * ((particles.σ[:, 1] - particles.σ[:, 4]).^2 + particles.σ[:, 1].^2 + particles.σ[:, 4].^2 + 6 * particles.σ[:, 2].^2)
# vmstress= 0.5 * ((particles.σ[:, 1]).^2)
stress = reshape(particles.σ[:, 1], npx, npy)

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
# fig = plot(disp_logger[1:(end-1), 1], sqrt.(sum(disp_logger[1:(end-1), 2:3].^2, dims=2)), title="$(model.runparams.spline)")
# t = 0:0.1:tend
# xpos = particles.position[end, 1] - particles.displacement[end, 1]
# ypos = particles.position[end, 2] - particles.displacement[end, 2]
# ue = 0.001
# a = ue*L
# b = ue*H
# ux = a*(1 .-exp.(-sqrt(E/ρ)*t/L))*sin(pi*xpos/(2*L)) .* sin(pi*ypos/(2*H))
# uy = a*(1 .-exp.(-sqrt(E/ρ)*t/L))*sin(pi*xpos/(2*L)) .* sin(pi*ypos/(2*H))
# mag = sqrt.(ux.^2+uy.^2)
# plot!(fig, t, mag)
# display(fig)

# fig = plot(disp_logger[1:(end-1), 1], disp_logger[1:(end-1), 2:3], title="$(model.runparams.spline)")
# # plot!(fig, t, ux)
# display(fig)

x_init = particles.position[end, 1] - particles.displacement[end, 1]
y_init = particles.position[end, 2] - particles.displacement[end, 2]
c = sqrt(model.E / model.ρ)
wx = c / L
wy = c / H
ux = ue * (1 .-exp.(-wx*disp_logger[:, 1])) * x_init * (y_init / H)^2
uy = ue * (1 .-exp.(-wy*disp_logger[:, 1])) * y_init * (x_init / L)^2
title_fig = "Δt=$(Δt), nix=$(nix), niy=$(niy), type=$(model.runparams.spline), ν=$(ν)"
fig = plot(disp_logger[:, 1], disp_logger[:, 2:3], label = ["MPM ux" "MPM uy"], legend=:bottomright, ylabel="Displacement [m]", xlabel="time [s]", title=title_fig)
plot!(fig, disp_logger[:, 1], ux, label="Analytic ux")
plot!(fig, disp_logger[:, 1], uy, label="Analytic uy")
display(fig)

fig = plot(disp_logger[:, 1], sqrt.((disp_logger[:, 2] - ux).^2), label="Error ux", xlabel="t [s]", ylabel="||u-u_e||_2", legend=:bottomright)
plot!(fig, disp_logger[:, 1], sqrt.((disp_logger[:, 2] - uy).^2), label="Error uy")
display(fig)