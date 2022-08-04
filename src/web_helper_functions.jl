function get_grid_cells(mpmgrid::MPMGrid{2})
    cx = unique(mpmgrid.splines[1].knot_vector)
    cy = unique(mpmgrid.splines[2].knot_vector)

    lx = kron(ones(length(cy) - 1), cx[1:(end-1)])
    ux = kron(ones(length(cy) - 1), cx[2:end])
    ly = kron(cy[1:(end-1)], ones(length(cx)-1))
    uy = kron(cy[2:end], ones(length(cx)-1))
    
    return Rect.(lx, ly, ux, uy)
end

function get_grid_cells(spline::BasisSpline)
    cx = unique(spline.knot_vector)
    return [(cx[i], cx[i+1]) for i=1:(length(cx)-1)]

end

function get_grid_cells(mpmgrid::MPMGrid{1})
    return get_grid_cells(mpmgrid.splines[1])
end


function identify_grid_cells(spline::BasisSpline, grid_cells::AbstractVector{T}, boundaryParticles::AbstractVecOrMat) where {T<:Tuple{<:Real, <:Real}}
    nx = length(unique(spline.knot_vector)) - 1

    bound_ind = BitVector(undef, length(grid_cells))
    bound_ind .= 0
    for p = 1:size(boundaryParticles, 1)
        bound_ind .|= contains.(grid_cells, boundaryParticles[p, 1])
    end

    # display(bound_ind)

    relevant_ind = BitVector(undef, length(grid_cells))
    relevant_ind .= 0
    index = 1
    lx = findfirst(bound_ind .== 1)
    ux = findlast(bound_ind .== 1)
    relevant_ind[lx:ux] .= 1
    exterior = (relevant_ind .== 0)
    interior = (bound_ind .== 0) .& (exterior .== 0) 

    return interior, bound_ind, exterior
end



function identify_grid_cells(mpmgrid::MPMGrid{2}, grid_cells::AbstractVector{Rect}, boundaryParticles::AbstractMatrix)
    nx = length(unique(mpmgrid.splines[1].knot_vector)) - 1
    ny = length(unique(mpmgrid.splines[2].knot_vector)) - 1

    bound_ind = BitVector(undef, length(grid_cells))
    bound_ind .= 0
    for p = 1:size(boundaryParticles, 1)
        bound_ind .|= contains.(grid_cells, boundaryParticles[p, 1], boundaryParticles[p, 2])
    end

    relevant_ind = BitVector(undef, length(grid_cells))
    relevant_ind .= 0
    index = 1
    for row = 1:ny
        bounds = bound_ind[index:(index+nx-1)]
        lx = findfirst(bounds .== 1)
        ux = findlast(bounds .== 1)
        if lx == ux
            lx = 1
        end
        if lx !== nothing && ux !== nothing
            relevant_ind[(index + lx - 1):(index+ux-1)] .= 1
        end
        index += nx
    end
    exterior = (relevant_ind .== 0)
    interior = (bound_ind .== 0) .& (exterior .== 0) 

    return interior, bound_ind, exterior
end

get_plot_shape(r::Rect) = Shape([r.lx,r.ux,r.ux,r.lx], [r.ly,r.ly,r.uy,r.uy])

function plot_rect!(fig, r::AbstractVector{Rect}; fillcolor=false)
    for ri ∈ r
        plot!(fig, get_plot_shape(ri), fillcolor=fillcolor)
    end
end

contains(a::Rect, px::Real, py::Real) = (a.lx <= px <= a.ux) && (a.ly <= py <= a.uy)
contains(a::Tuple{<:Real, <:Real}, b::Tuple{<:Real, <:Real}) = (a[1] <= b[1] <= b[2] <= a[2])
contains(a::Tuple{<:Real, <:Real}, b::Real) = (a[1] <= b <= a[2])

function get_boundary_cells(grid_cells::AbstractVector{Rect}, boundary_particles::AbstractMatrix)
    indices = BitVector(undef, length(grid_cells))
    indices .= 0
    for p = 1:size(boundary_particles, 1)
        indices .|= contains.(grid_cells, boundary_particles[p, 1], boundaryParticles[p, 2])
    end


    return indices
end


# grid_cells, interior, boundary, exterior = get_grid_cells(mpmgrid, boundaryParticles)
# boundary_cell_indices = get_boundary_cells(grid_cells, boundaryParticles)

function support(mpmgrid::MPMGrid{2}, i::Int, j::Int)
    return Rect(mpmgrid.splines[1].knot_vector[i], 
                mpmgrid.splines[2].knot_vector[j], 
                mpmgrid.splines[1].knot_vector[i+mpmgrid.splines[1].degree+1],
                mpmgrid.splines[2].knot_vector[j+mpmgrid.splines[2].degree+1])
end

function support(mpmgrid::MPMGrid{1}, i::Int)
    return support(mpmgrid.splines[1], i)
end

function support(spline::BasisSpline, i::Int)
    return (spline.knot_vector[i], spline.knot_vector[i+spline.degree+1])

end

function support(mpmgrid::MPMGrid{2}, i::Int)
    ii = (i-1) % ndof(mpmgrid.splines[1]) + 1
    ij = floor(Int, (i-1) / ndof(mpmgrid.splines[1]) + 1)
    return support(mpmgrid, ii, ij)
end


function identify_splines(mpmgrid, grid_cells)
    splines = Vector{Vector{Int64}}(undef, ndof(mpmgrid))
    grid = [Vector{Int64}(undef, 0) for i=1:length(grid_cells)]
    ind = 1
    # for j = 1:ndof(mpmgrid.splines[2])
    #     for i = 1:ndof(mpmgrid.splines[1])
    for ind = 1:ndof(mpmgrid)
        splines[ind] = Vector{Int64}(undef, 0)
        suppB = support(mpmgrid, ind)
        for k = 1:length(grid_cells)
            if contains(suppB, grid_cells[k])
                append!(splines[ind], k)
                append!(grid[k], ind)
            end
        end
        # ind += 1
    end
    #     end
    # end

    return splines, grid
end

function identify_spline_stability(mpmgrid, grid_cells, grid_splines, boundaryParticles)
    interior, boundary, exterior = identify_grid_cells(mpmgrid, grid_cells, boundaryParticles)

    relevant_splines = BitVector(undef, ndof(mpmgrid))
    stable_splines = BitVector(undef, ndof(mpmgrid))
    
    relevant_splines .= 0
    stable_splines .= 0
    
    for i = 1:length(grid_cells)
        for s ∈ grid_splines[i]
            if interior[i] == 1
                stable_splines[s] = 1
                relevant_splines[s] = 1
            end
            if boundary[i] == 1
                relevant_splines[s] = 1
            end
        end
    end
    
    unstable = relevant_splines .& (stable_splines .== 0)
    exterior_splines = (relevant_splines .== 0)

    return stable_splines, unstable, exterior_splines
end

function hausdorff_distance(r1::Rect, r2::Rect)
    h = 0
    for xr1 ∈ [r1.lx, r1.ux]
        for yr1 ∈ [r1.ly, r1.uy]
            shortest = Inf
            for xr2 ∈ [r2.lx, r2.ux]
                for yr2 ∈ [r2.ly, r2.uy]
                    dij = sqrt((xr1 - xr2)^2 + (yr1 - yr2)^2)
                    if dij < shortest
                        shortest = dij
                    end
                end
            end
            if shortest > h
                h = shortest
            end
        end
    end
    return h
end

function midpoint_distance(r1::Rect, r2::Rect)
    return sqrt((r1.ux + r1.lx - r2.ux - r2.lx)^2 + (r1.uy + r1.ly - r2.uy - r2.ly)^2) / 2
end

function midpoint_distance(r1::Tuple{<:Real, <:Real}, r2::Tuple{<:Real, <:Real})
    return sqrt((sum(r1)/2 - sum(r2) / 2)^2)
end

function find_closest_stable_basis(grid_cells::AbstractVector{Rect}, interior::BitVector, boundary_index::Int)
    closest = findmin(hausdorff_distance.(grid_cells[interior], Ref(grid_cells[boundary_index])))
    return findall(interior)[closest[2]]
end

function find_closest_stable_basis_mid(grid_cells::AbstractVector{Rect}, interior::BitVector, boundary_index::Int)
    closest = findmin(midpoint_distance.(grid_cells[interior], Ref(grid_cells[boundary_index])))
    return findall(interior)[closest[2]]
end

function find_closest_stable_basis_mid(grid_cells::AbstractVector{Rect}, interior::BitVector, suppB::Rect)
    closest = findmin(midpoint_distance.(grid_cells[interior], Ref(suppB)))
    return findall(interior)[closest[2]]
end

function find_closest_stable_basis_mid(grid_cells::AbstractVector{T}, interior::BitVector, suppB::T) where {T<:Tuple{<:Real, <:Real}}
    closest = findmin(midpoint_distance.(grid_cells[interior], Ref(suppB)))
    return findall(interior)[closest[2]]
end