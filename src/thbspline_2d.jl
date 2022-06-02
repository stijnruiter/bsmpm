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
    for i = 2:hierDom.max_level
        knot_1 = copy(knot_1)
        knot_2 = copy(knot_2)
        S1 = refineKnot!(knot_1, mpmgrid.splines[1].degree)
        S2 = refineKnot!(knot_2, mpmgrid.splines[2].degree)
        refined_grids[i] = MPMGrid(knot_1, knot_2, mpmgrid.splines[1].degree, mpmgrid.splines[2].degree)
    end
    return refined_grids
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

    refinedGrids = refineGrid(mpmgrid, hierDom)
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
    return active_spline_indices, refinedGrids
end

struct HierarchicalBasisSpline2D{ni, level}
    hierDom::HierarchicalDomain2D
    active_spline_indices_2D::Vector{Vector{Int64}}
    active_spline_indices_1D::Vector{Tuple{Vector{Int64}, Vector{Int64}}}
    grids::Vector{MPMGrid{2}}
    function HierarchicalBasisSpline2D(mpmgrid::MPMGrid{2}, hierDom::HierarchicalDomain2D)
        active_spline_indices, grids = compute_active_splines(mpmgrid, hierDom)
        indices_1d = Vector{Tuple{Vector{Int64}, Vector{Int64}}}(undef, hierDom.max_level)
        for i = 1:hierDom.max_level
            tmp = get_spline_indices(active_spline_indices[i], grids[i])
            indices_1d[i] = (sort(unique(tmp[:, 1])), sort(unique(tmp[:, 2])))
        end
        ni = sum(sum.(active_spline_indices))
        new{ni, hierDom.max_level}(hierDom, findall.(active_spline_indices), indices_1d, grids)
    end
end

ndof(thb_splines::HierarchicalBasisSpline2D{ni, level}) where {ni, level} = ni
maxlevel(thb_splines::HierarchicalBasisSpline2D{ni, level}) where {ni, level} = level

mutable struct THBasisSplineDenseStorage2D{np, ni}
    B::Matrix{Float64}
    dB1::Matrix{Float64}
    dB2::Matrix{Float64}
    temp_store::Matrix{Float64}
    splines_1d::Vector{BasisSplineDenseStorage1D{np, ni1} where ni1}
end

function initialize_spline_storage(nparticles::Int64, thbspline::HierarchicalBasisSpline2D)
    storages_1D = Vector{BasisSplineDenseStorage1D}(undef, thbspline.hierDom.max_level * 2)
    for i = 1:thbspline.hierDom.max_level
        storages_1D[i * 2 - 1] = initialize_spline_storage(nparticles, thbspline.grids[i].splines[1])
        storages_1D[i * 2]     = initialize_spline_storage(nparticles, thbspline.grids[i].splines[2])
    end
    return THBasisSplineDenseStorage2D{nparticles, ndof(thbspline)}(zeros(nparticles, ndof(thbspline)), 
            zeros(nparticles, ndof(thbspline)), 
            zeros(nparticles, ndof(thbspline)), 
            zeros(nparticles, ndof(thbspline)), 
            storages_1D)
end

function get_active_1d_spline_indices(active_spline_indices::AbstractVector{Int64}, grids::AbstractVector{MPMGrid{2}}, hierDom::HierarchicalDomain2D)
    indices = Vector{Tuple{Vector{Int64}, Vector{Int64}}}(undef, hierDom.max_level)
    for i = 1:hierDom.max_level
        tmp = get_spline_indices(active_spline_indices[i], grids[i])
        indices[i] = (sort(unique(tmp[:, 1])), sort(unique(tmp[:, 2])))
    end
    return indices
end

function compute_bspline_values!(storage::THBasisSplineDenseStorage2D, coord::AbstractMatrix{<:Real}, thbsplines::HierarchicalBasisSpline2D)
    for i = 1:maxlevel(thbsplines)
        compute_bspline_values!(storage.splines_1d[i * 2 - 1], coord[:, 1], thbsplines.grids[i].splines[1])
        compute_bspline_values!(storage.splines_1d[i * 2], coord[:, 2], thbsplines.grids[i].splines[2])
    end
    current_index = 1
    for i = 1:maxlevel(thbsplines)
        for j ∈ thbsplines.active_spline_indices_2D[i]
            i1, i2 = get_spline_indices(j, thb_splines.grids[i])
            storage.B[:, current_index] = storage.splines_1d[i * 2 - 1].B[:, i1] .* storage.splines_1d[i * 2].B[:, i2]
            storage.dB1[:, current_index] = storage.splines_1d[i * 2 - 1].dB[:, i1] .* storage.splines_1d[i * 2].B[:, i2]
            storage.dB2[:, current_index] = storage.splines_1d[i * 2 - 1].B[:, i1] .* storage.splines_1d[i * 2].dB[:, i2]
            current_index += 1
        end
    end
end
