using Plots

include("../src/webspline.jl")



t = (0.01:0.01:2).*π

radius = .75
# boundaryParticlesX = radius .* cos.(t)  .+ 1
# boundaryParticlesY = radius .* sin.(t) .+ 1

# boundaryParticles = [boundaryParticlesX boundaryParticlesY]


nix = 20
niy = 20
degree = 2
L = 2
H = 2

mpmgrid = MPMGrid((0, L), nix, degree, (0, H), niy, degree)


particles = initialize_uniform_particles_circle(mpmgrid, 1, radius, 100, 200)
particles.position .+= 1
# corners = [ (0.5, 0.5),
#             (1.5, 0.5),
#             (1.5, 1.5),
#             (0.5,  1.5)]
#             npx = 100
#             npy = 100
# particles = initialize_uniform_particles(1, corners, npx, npy)
# bel = BitVector(undef, npx*npy)
# bel .= 0
# bel[1:npy] .= 1
# bel[1:npy:end] .= 1
# bel[npy:npy:end] .= 1
# bel[(end-npy-1):end] .= 1
# boundaryParticles = particles.position[bel, :]

spline_storage = initialize_spline_storage(particles, mpmgrid)

compute_bspline_values!(spline_storage, particles.position, mpmgrid)

grid_cells = get_grid_cells(mpmgrid)
splines, grid_splines = identify_splines(mpmgrid, grid_cells)
interior, boundary, exterior = identify_grid_cells(grid_cells, particles.position[particles.bel, :])
stable_splines, unstable, exterior_splines = identify_spline_stability(mpmgrid, grid_cells, particles.position[particles.bel, :])


# display(scatter(particles.position[:, 1], particles.position[:, 2], spline_storage.B[:, 69]))
# web_splines!(spline_storage, mpmgrid, bel, particles)
# j = findall(unstable)[1]

# suppBj = support(mpmgrid, j)
# closest_stable_grid_cell = find_closest_stable_basis_mid(grid_cells, interior, suppBj)
# stable_supp = grid_splines[closest_stable_grid_cell]

# Bi = grid_cells[closest_stable_grid_cell]


# vj_1 = (j - 1) % ndof(mpmgrid.splines[1]) + 1
# vj_2 = floor(Int, (j - 1) / ndof(mpmgrid.splines[1]) + 1)

# # vi_1 = (stable_splines .- 1) .% ndof(mpmgrid.splines[1]) .+ 1
# # vi_2 = floor.(Int, (stable_splines .- 1) ./ ndof(mpmgrid.splines[1]) .+ 1)
# rtaylor_1 = (Bi.lx + Bi.ux) / 2
# rtaylor_2 = (Bi.ly + Bi.uy) / 2

# e_ij_1 = compute_eij_1d(mpmgrid.splines[1], rtaylor_1, vj_1)
# e_ij_2 = compute_eij_1d(mpmgrid.splines[2], rtaylor_2, vj_2)

# e_ij = (kron(ones(ndof(mpmgrid.splines[2])), e_ij_1) .* kron(e_ij_2, ones(ndof(mpmgrid.splines[1]))))[stable_supp]
# # display(e_ij)
# spline_storage.B[:, stable_supp] += spline_storage.B[:, j] * e_ij'
# spline_storage.dB1[:, stable_supp] += spline_storage.dB1[:, j] * e_ij'
# spline_storage.dB2[:, stable_supp] += spline_storage.dB2[:, j] * e_ij'
# # end
display(scatter(particles.position[:, 1], particles.position[:, 2], spline_storage.B[:, stable_splines][:, 3]))
web_splines!(spline_storage, mpmgrid, particles)
if !all(sum(spline_storage.B[:, stable_splines], dims=2) .≈ 1)
    throw(ErrorException("No paritition of unity in WEB splines"))
end
display(scatter(particles.position[:, 1], particles.position[:, 2], spline_storage.B[:, stable_splines][:, 3]))

xstart = kron(ones(ndof(mpmgrid.splines[2])), mpmgrid.splines[1].knot_vector[1:ndof(mpmgrid.splines[1])])
ystart = kron(mpmgrid.splines[2].knot_vector[1:ndof(mpmgrid.splines[2])], ones(ndof(mpmgrid.splines[1])))

fig = scatter(legend=false, aspect_ratio=:equal)
# fig = plot(legend=false)
plot_rect!(fig, grid_cells)
plot_rect!(fig, grid_cells[boundary]; fillcolor = :blue)
plot_rect!(fig, grid_cells[exterior]; fillcolor = :red)
plot_rect!(fig, grid_cells[interior]; fillcolor = :green)
scatter!(fig, xstart[unstable], ystart[unstable], color=:white)
scatter!(fig, xstart[stable_splines], ystart[stable_splines], color=:black)
scatter!(fig, xstart[exterior_splines], ystart[exterior_splines], color=:green)


# scatter!(fig, particles.position[:, 1][notunity], particles.position[:, 2][notunity])
# scatter!(fig, boundaryParticles[:, 1], boundaryParticles[:, 2])

# for el ∈ findall(boundary)
#     clostest_hausdorff = find_closest_stable_basis_mid(grid_cells, interior, el)
#     clostest_mid = find_closest_stable_basis_mid(grid_cells, interior, el)
#     if clostest_hausdorff !== clostest_mid
#         println("Not equals $(el)")
#     end
# end

j = findlast(unstable)
suppBj = support(mpmgrid, j)

closest_interior = find_closest_stable_basis_mid(grid_cells, interior, suppBj)
relevant_splines = grid_splines[closest_interior]

xi_relevant = (relevant_splines .- 1) .% ndof(mpmgrid.splines[1]) .+ 1
yi_relevant = floor.(Int, (relevant_splines .- 1) ./ ndof(mpmgrid.splines[1])) .+ 1 
plot_rect!(fig, [suppBj]; fillcolor = :gray)
plot_rect!(fig, [grid_cells[closest_interior]]; fillcolor = :black)
scatter!(fig, mpmgrid.splines[1].knot_vector[xi_relevant], mpmgrid.splines[2].knot_vector[yi_relevant], color=:yellow)

display(fig)

# scatter!(xstart[grid_splines[closest_interior]], ystart[grid_splines[closest_interior]])
# display(fig)

# fig = plot(legend=false)
# for i = 1:npx
#     y = ((i-1)*npy+1):(i*npy)
#     x = 1:npy:(npx*npy)
#     plot!(fig, particles.position[x, 1], spline_storage.B[x, stable_splines][:, 3])
# end
# display(fig)

# fig = plot(legend=false)
# # for i = 1:npx
# #     y = ((i-1)*npy+1):(i*npy)
# #     x = 1:npy:(npx*npy)
# #     plot!(fig, particles.position[x, 1], spline_storage.B[x, stable_splines][:, 3])
# # end
# display(fig)


# fig = plot()
# x = 0.5:0.01:1.5
# plot!(fig, x, reduce(hcat, dBspline.(x, Ref(mpmgrid.splines[1]), mpmgrid.splines[1].degree, 1))[9, :])
# fig = scatter(particles.position[:, 1], particles.position[:,2], color=:black, aspect_ratio=:equal)
# scatter!(fig, particles.position[bel, 1], particles.position[bel, 2], color=:red)
# sp = reduce(hcat, dBspline.(1.0, Ref(mpmgrid.splines[1]), mpmgrid.splines[1].degree, 0:mpmgrid.splines[1].degree))