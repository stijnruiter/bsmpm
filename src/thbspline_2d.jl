include("bspline.jl")
include("data_structures.jl")
include("set_operation.jl")

using Plots



struct HierarchicalDomain2D
    max_level::Int64
    domains::Vector{Vector{Rect}}
    function HierarchicalDomain2D(ground_level::Rect, domainLevels::Vector{Rect}...)
        max_level = length(domainLevels) + 1
        # TODO, check if domains are nested
        if length(domainLevels) >= 1
            if !contains(ground_level, domainLevels[1])
                throw(ErrorException("Ω1⊆Ω0 does not hold"))
            end
            for i = 2:length(domainLevels)
                if !contains(domainLevels[i-1], domainLevels[i])
                    throw(ErrorException("Ω$(i)⊆Ω$(i-1) does not hold"))
                end
            end
        end
        new(max_level, union([[ground_level]], [i for i ∈ domainLevels]))
    end
end

function refineSpline(spline::BasisSpline)
    knot = copy(spline.knot_vector)
    S = refineKnot!(knot, spline.degree)
    return S, BasisSpline(knot, spline.degree)
end

function refineKnot!(knot::AbstractVector{Float64}, p::Int)
    nsplines = length(knot) - p - 1
    # subdivision matrix of all knot insertions
    S = Matrix{Float64}(I, nsplines, nsplines)
    inserts = Vector{Float64}(undef, 0)
    for i = 1:(length(knot) - 1)
        if knot[i] == knot[i + 1]
            continue;
        else
            s = (knot[i] + knot[i + 1]) / 2;
            # insert!(inserts, length(inserts) + 1, (knot[i] + s)/2);
            insert!(inserts, length(inserts) + 1, s);
            # insert!(inserts, length(inserts) + 1, (s + knot[i + 1]) / 2);
        end
    end
    for new_knot in inserts
        s = findlast(knot .<= new_knot)
        # println(s)
        A = zeros(nsplines + 1, nsplines)
        for k = 1:(nsplines + 1)
            α = 0;
            # println((new_knot - knot[s]) / (knot[s + p] - knot[s]))
            if k <= s - p
                α = 1
            elseif k >= s + 1
                α = 0;
            else
                α = (new_knot - knot[k]) / (knot[k + p] - knot[k])  
            end
            # println(string(k) * ": " * string(α))
            if k > 1
                A[k, k - 1] = 1 - α
            end
            if k <= nsplines
                A[k, k] = α
            end
        end
        insert!(knot, s + 1, new_knot)
        S = A * S
        nsplines = nsplines + 1
    end
    return S
end

function refineGrid(mpmgrid::MPMGrid{2}, hierDom::HierarchicalDomain2D)
    refined_grids = Vector{MPMGrid{2}}(undef, hierDom.max_level)
    refined_grids[1] = mpmgrid
    knot_1 = copy(mpmgrid.splines[1].knot_vector)
    knot_2 = copy(mpmgrid.splines[2].knot_vector)
    subdivisions = Vector{Matrix{Float64}}(undef, (hierDom.max_level-1) * 2)
    for i = 2:hierDom.max_level
        knot_1 = copy(knot_1)
        knot_2 = copy(knot_2)
        subdivisions[(i-1)*2 - 1] = refineKnot!(knot_1, mpmgrid.splines[1].degree)
        subdivisions[(i-1)*2] = refineKnot!(knot_2, mpmgrid.splines[2].degree)
        refined_grids[i] = MPMGrid(knot_1, knot_2, mpmgrid.splines[1].degree, mpmgrid.splines[2].degree)
    end
    return refined_grids, subdivisions
end

function get_spline_indices(i::Int64, mpmgrid::MPMGrid{2})
    if i < 1 || i > ndof(mpmgrid)
        throw(BoundsError(mpmgrid, i))
    end
    i_1 = (i - 1) % ndof(mpmgrid.splines[1]) + 1
    i_2 = floor(Int64, (i - 1) / ndof(mpmgrid.splines[2]) + 1)
    return [i_1; i_2]
end

get_spline_indices(i::AbstractVector{Int64}, mpmgrid::MPMGrid{2}) = mapreduce(permutedims, vcat, get_spline_indices.(i, Ref(mpmgrid)))
get_spline_indices(i::BitVector, mpmgrid::MPMGrid{2}) = get_spline_indices(findall(i), mpmgrid)


function get_spline_support(i::Int64, mpmgrid::MPMGrid{2})::Rect
    (i1, i2) = get_spline_indices(i, mpmgrid)
    return Rect(mpmgrid.splines[1].knot_vector[i1],
                mpmgrid.splines[2].knot_vector[i2], 
                mpmgrid.splines[1].knot_vector[i1 + mpmgrid.splines[1].degree + 1],
                mpmgrid.splines[2].knot_vector[i2 + mpmgrid.splines[2].degree + 1])
end

function compute_active_splines(mpmgrid::MPMGrid{2}, hierDom::HierarchicalDomain2D)

    refinedGrids, subdivisions = refineGrid(mpmgrid, hierDom)
    active_spline_indices = Vector{BitVector}(undef, length(refinedGrids))
    active_spline_indices[1] = BitVector(undef, ndof(refinedGrids[1])) .= 1
    for i = 2:length(active_spline_indices)
        active_spline_indices[i] = BitVector(undef, ndof(refinedGrids[i])) .= 0
    end

    for l = 2:hierDom.max_level
        # for k = 1:(l - 1)
        # H_A
        for i ∈ findall(active_spline_indices[l-1])
            if contains(hierDom.domains[l], get_spline_support(i, refinedGrids[l-1]))
                active_spline_indices[l-1][i] = 0
            end
        end
        # end
        # H_B
        for i = 1:ndof(refinedGrids[l])
            if contains(hierDom.domains[l], get_spline_support(i, refinedGrids[l]))
                active_spline_indices[l][i] = 1
            end
        end
    end
    return active_spline_indices, refinedGrids, subdivisions
end

struct HierarchicalMPMGrid2D{ni, level} <: AbstractMPMGrid{2}
    hierDom::HierarchicalDomain2D
    active_spline_indices_2D::Vector{Vector{Int64}}
    active_spline_indices_1D::Vector{Tuple{Vector{Int64}, Vector{Int64}}}
    grids::Vector{MPMGrid{2}}
    subdivisions::Vector{Matrix{Float64}}
    function HierarchicalMPMGrid2D(mpmgrid::MPMGrid{2}, hierDom::HierarchicalDomain2D)
        active_spline_indices, grids, subdivisions = compute_active_splines(mpmgrid, hierDom)
        indices_1d = Vector{Tuple{Vector{Int64}, Vector{Int64}}}(undef, hierDom.max_level)
        for i = 1:hierDom.max_level
            tmp = get_spline_indices(active_spline_indices[i], grids[i])
            indices_1d[i] = (sort(unique(tmp[:, 1])), sort(unique(tmp[:, 2])))
        end

        active_spline_indices_2D = findall.(active_spline_indices)

        subdivisions2D = Vector{Matrix{Float64}}(undef, hierDom.max_level - 1)
        for i = 1:length(subdivisions2D)
            subdivisions2D[i] = (kron(ones(ndof(grids[i+1].splines[2])), kron(ones(1, ndof(grids[i].splines[2])), subdivisions[i*2-1])) .*
                                kron(kron(subdivisions[i*2], ones(1, ndof(grids[i].splines[1]))), ones(ndof(grids[i+1].splines[1]))))[active_spline_indices_2D[i+1], active_spline_indices_2D[i]]
        end 
        ni = sum(sum.(active_spline_indices))
        new{ni, hierDom.max_level}(hierDom, active_spline_indices_2D, indices_1d, grids, subdivisions2D)
    end
end

ndof(thb_grid::HierarchicalMPMGrid2D{ni, level}) where {ni, level} = ni
maxlevel(thb_grid::HierarchicalMPMGrid2D{ni, level}) where {ni, level} = level

mutable struct THBasisSplineDenseStorage2D{np, ni} <: AbstractBasisSplineStorage2D{np, ni}
    B::Matrix{Float64}
    dB1::Matrix{Float64}
    dB2::Matrix{Float64}
    splines_1d::Vector{BasisSplineDenseStorage1D{np, ni1} where ni1}
    active::BitVector
end

function initialize_spline_storage(nparticles::Int64, thb_grid::HierarchicalMPMGrid2D)
    storages_1D = Vector{BasisSplineDenseStorage1D}(undef, thb_grid.hierDom.max_level * 2)
    for i = 1:thb_grid.hierDom.max_level
        storages_1D[i * 2 - 1] = initialize_spline_storage(nparticles, thb_grid.grids[i].splines[1])
        storages_1D[i * 2]     = initialize_spline_storage(nparticles, thb_grid.grids[i].splines[2])
    end
    return THBasisSplineDenseStorage2D{nparticles, ndof(thb_grid)}(zeros(nparticles, ndof(thb_grid)), 
            zeros(nparticles, ndof(thb_grid)), 
            zeros(nparticles, ndof(thb_grid)), 
            storages_1D, 
            BitVector(ones(ndof(thb_grid))))
end

function get_active_1d_spline_indices(active_spline_indices::AbstractVector{Int64}, grids::AbstractVector{MPMGrid{2}}, hierDom::HierarchicalDomain2D)
    indices = Vector{Tuple{Vector{Int64}, Vector{Int64}}}(undef, hierDom.max_level)
    for i = 1:hierDom.max_level
        tmp = get_spline_indices(active_spline_indices[i], grids[i])
        indices[i] = (sort(unique(tmp[:, 1])), sort(unique(tmp[:, 2])))
    end
    return indices
end

# function compute_bspline_values!(storage::THBasisSplineDenseStorage2D, coord::AbstractMatrix{<:Real}, thb_grid::HierarchicalMPMGrid2D)
#     for i = 1:maxlevel(thb_grid)
#         compute_bspline_values!(storage.splines_1d[i * 2 - 1], coord[:, 1], thb_grid.grids[i].splines[1])
#         compute_bspline_values!(storage.splines_1d[i * 2], coord[:, 2], thb_grid.grids[i].splines[2])
#     end
#     current_index = 1
#     for i = 1:maxlevel(thb_grid)
#         for j ∈ thb_grid.active_spline_indices_2D[i]
#             i1, i2 = get_spline_indices(j, thb_grid.grids[i])
#             storage.B[:, current_index] = storage.splines_1d[i * 2 - 1].B[:, i1] .* storage.splines_1d[i * 2].B[:, i2]
#             storage.dB1[:, current_index] = storage.splines_1d[i * 2 - 1].dB[:, i1] .* storage.splines_1d[i * 2].B[:, i2]
#             storage.dB2[:, current_index] = storage.splines_1d[i * 2 - 1].B[:, i1] .* storage.splines_1d[i * 2].dB[:, i2]
#             current_index += 1
#         end
#     end
#     truncate_bsplines!(storage, thb_grid)
# end

function compute_hbspline_values!(storage::THBasisSplineDenseStorage2D, particles::Particles, thb_grid::HierarchicalMPMGrid2D)
    compute_hbspline_values!(storage, particles.position, thb_grid)
end

function compute_hbspline_values!(storage::THBasisSplineDenseStorage2D, coord::AbstractMatrix{<:Real}, thb_grid::HierarchicalMPMGrid2D)
    for i = 1:maxlevel(thb_grid)
        compute_bspline_values!(storage.splines_1d[i * 2 - 1], coord[:, 1], thb_grid.grids[i].splines[1])
        compute_bspline_values!(storage.splines_1d[i * 2], coord[:, 2], thb_grid.grids[i].splines[2])
    end
    current_index = 1
    for i = 1:maxlevel(thb_grid)
        for j ∈ thb_grid.active_spline_indices_2D[i]
            i1, i2 = get_spline_indices(j, thb_grid.grids[i])
            storage.B[:, current_index] = storage.splines_1d[i * 2 - 1].B[:, i1] .* storage.splines_1d[i * 2].B[:, i2]
            storage.dB1[:, current_index] = storage.splines_1d[i * 2 - 1].dB[:, i1] .* storage.splines_1d[i * 2].B[:, i2]
            storage.dB2[:, current_index] = storage.splines_1d[i * 2 - 1].B[:, i1] .* storage.splines_1d[i * 2].dB[:, i2]
            current_index += 1
        end
    end    
end
function compute_thbspline_values!(storage::THBasisSplineDenseStorage2D, particles::Particles, thb_grid::HierarchicalMPMGrid2D)
    compute_thbspline_values!(storage, particles.position, thb_grid)
end
function compute_thbspline_values!(storage::THBasisSplineDenseStorage2D, coord::AbstractMatrix{<:Real}, thb_grid::HierarchicalMPMGrid2D)
    compute_hbspline_values!(storage, coord, thb_grid)
    truncate_bsplines!(storage, thb_grid)
end

function compute_ethbspline_values!(storage::THBasisSplineDenseStorage2D, particles::Particles, thb_grid::HierarchicalMPMGrid2D)
    compute_hbspline_values!(storage, particles.position, thb_grid)
    truncate_bsplines!(storage, thb_grid)
    web_splines!(storage, thb_grid, particles.position[particles.bel, :])
end

function truncate_bsplines!(storage::THBasisSplineDenseStorage2D, thb_grid::HierarchicalMPMGrid2D)
    offsets = [0; cumsum(length.(thb_grid.active_spline_indices_2D))]
    for i = 1:(maxlevel(thb_grid)-1)
        S = thb_grid.subdivisions[i]
        for j = (i+1):maxlevel(thb_grid)
            storage.B[:, (offsets[i] + 1):offsets[i+1]] -= storage.B[:, (offsets[j] + 1):offsets[j+1]] * S
            storage.dB1[:, (offsets[i] + 1):offsets[i+1]] -= storage.dB1[:, (offsets[j] + 1):offsets[j+1]] * S
            storage.dB2[:, (offsets[i] + 1):offsets[i+1]] -= storage.dB2[:, (offsets[j] + 1):offsets[j+1]] * S
            if j < maxlevel(thb_grid)
                S =  thb_grid.subdivisions[j] * S
            end
        end
    end
end

function web_splines!(storage::THBasisSplineDenseStorage2D, thb_grid::HierarchicalMPMGrid2D, boundaryParticles::AbstractMatrix)
    offsets = [0; cumsum(length.(thb_grid.active_spline_indices_2D))]
    for l = 1:maxlevel(thb_grid)
        grid_cells = get_grid_cells(thb_grid.grids[l])
        splines, grid_splines = identify_splines(thb_grid.grids[l], grid_cells)
        interior, boundary, exterior = identify_grid_cells(thb_grid.grids[l], grid_cells, boundaryParticles)
        stable_splines, unstable, exterior_splines = identify_spline_stability(thb_grid.grids[l], grid_cells, grid_splines, boundaryParticles)
        storage.active[(offsets[l]+1):offsets[l+1]] = stable_splines[thb_grid.active_spline_indices_2D[l]]
        for j ∈ findall(unstable)
            if j in thb_grid.active_spline_indices_2D[l]
                suppBj = support(thb_grid.grids[l], j)
                closest_stable_grid_cell = find_closest_stable_basis_mid(grid_cells, interior, suppBj)
                stable_supp = grid_splines[closest_stable_grid_cell]
        
                Bi = grid_cells[closest_stable_grid_cell]
                vj_1 = (j - 1) % ndof(thb_grid.grids[l].splines[1]) + 1
                vj_2 = floor(Int, (j - 1) / ndof(thb_grid.grids[l].splines[1]) + 1)
                
                rtaylor_1 = (Bi.lx + Bi.ux) / 2
                rtaylor_2 = (Bi.ly + Bi.uy) / 2

                e_ij_1 = compute_eij_1d(thb_grid.grids[l].splines[1], rtaylor_1, vj_1)
                e_ij_2 = compute_eij_1d(thb_grid.grids[l].splines[2], rtaylor_2, vj_2)

                e_ij = (kron(ones(ndof(thb_grid.grids[l].splines[2])), e_ij_1) .* kron(e_ij_2, ones(ndof(thb_grid.grids[l].splines[1]))))[stable_supp]
                
                indices = find_index_in_supmat(stable_supp, thb_grid, l)
                in_truncation = indices .!= nothing
                indices = indices[in_truncation] .+ offsets[l]
                j_offset = findfirst(thb_grid.active_spline_indices_2D[l] .== j) + offsets[l]
                storage.B[:, indices] += storage.B[:, j_offset] * e_ij[in_truncation]'
                storage.dB1[:, indices] += storage.dB1[:, j_offset] * e_ij[in_truncation]'
                storage.dB2[:, indices] += storage.dB2[:, j_offset] * e_ij[in_truncation]'
            end
        end
    end
end

function find_index_in_supmat(splines, thb_grid, level)
    return [findfirst(thb_grid.active_spline_indices_2D[level] .== splines[i]) for i in eachindex(splines)]
end

function get_boundary_indices(thb_grid::HierarchicalMPMGrid2D; fix_left::Bool = false, 
    fix_right::Bool = false, fix_top::Bool = false, fix_bottom::Bool = false) 
    indices_1 = BitVector(undef, 0)
    indices_2 = BitVector(undef, 0)
    total = ndof(thb_grid)

    for i = 1:maxlevel(thb_grid)
        bc = get_boundary_indices(thb_grid.grids[i]; fix_left=fix_left, fix_right=fix_right, fix_top=fix_top, fix_bottom=fix_bottom)
        id1 = bc.vector_indices[bc.vector_indices .<= ndof(thb_grid.grids[i])]
        id2 = bc.vector_indices[bc.vector_indices .> ndof(thb_grid.grids[i])] .- ndof(thb_grid.grids[i])
        indices_1 = [indices_1; in.(thb_grid.active_spline_indices_2D[i], Ref(id1))]
        indices_2 = [indices_2; in.(thb_grid.active_spline_indices_2D[i], Ref(id2))]
    end
    indices_1 = findall(indices_1)
    indices_2 = findall(indices_2)
    vector_indices = sort(unique([indices_1; indices_2 .+ total]))
    scalar_indices = sort(unique([indices_1; indices_2]))
    return DirichletBoundaryConditions{2}(scalar_indices, vector_indices)
end