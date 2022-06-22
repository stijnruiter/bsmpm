include("bspline.jl")

abstract type AbstractMPMGrid{dim} end

struct MPMGrid{dim} <: AbstractMPMGrid{dim}
    splines::Vector{BasisSpline}
    bounds::Vector{Tuple{Float64, Float64}}

    # Generates a 1D MPM grid
    function MPMGrid(knot_vector::AbstractVector{<:Real}, degree::Int64)
        spline = BasisSpline(knot_vector, degree)
        bounds = (knot_vector[begin], knot_vector[end])
        return new{1}([spline], [bounds])
    end
    # Generates a 1D MPM grid
    function MPMGrid(bounds::Tuple{<:Real, <:Real}, length::Int, degree::Int)
        spline = BasisSpline(bounds[1], bounds[2], length, degree)
        return MPMGrid(spline)
    end
    
    function MPMGrid(knot_vector1::AbstractVector{<:Real}, knot_vector2::AbstractVector{<:Real}, degree1::Int64, degree2::Int64)
        spline1 = BasisSpline(knot_vector1, degree1)
        spline2 = BasisSpline(knot_vector2, degree2)
        bounds1 = (knot_vector1[begin], knot_vector1[end])
        bounds2 = (knot_vector2[begin], knot_vector2[end])
        return new{2}([spline1; spline2], [bounds1, bounds2])
    end

    function MPMGrid(bound1::Tuple{<:Real, <:Real}, n1::Int, degree1::Int, bound2::Tuple{<:Real, <:Real}, n2::Int, degree2::Int)
        spline1 = BasisSpline(bound1[1], bound1[2], n1, degree1)
        spline2 = BasisSpline(bound2[1], bound2[2], n2, degree2)
        return MPMGrid(spline1, spline2)
    end

    function MPMGrid(splines::BasisSpline...)
        dim = length(splines)
        s = [splines[i] for i = 1:dim]
        b = [(splines[i].knot_vector[begin], splines[i].knot_vector[end]) for i = 1:dim]
        return new{dim}(s, b)
    end
end

function compute_bspline_values!(storage::AbstractBasisSplineStorage1D, coord::AbstractVecOrMat{<:Real}, mpmgrid::MPMGrid{1})
    compute_bspline_values!(storage, coord, mpmgrid.splines)
end
function compute_bspline_values!(storage::AbstractBasisSplineStorage2D, coord::AbstractMatrix{<:Real}, mpmgrid::MPMGrid{2})
    compute_bspline_values!(storage, coord, mpmgrid.splines)
end

mutable struct Particles{dim, np}
    position::Matrix{Float64}
    velocity::Matrix{Float64}
    displacement::Matrix{Float64}

    # ϵ::Vector{Matrix{Float64}}
    # σ::Vector{Matrix{Float64}}
    # deformation::Vector{Matrix{Float64}}

    ϵ::Matrix{Float64}
    σ::Matrix{Float64}
    deformation::Matrix{Float64}

    ρ::Vector{Float64}
    volume::Vector{Float64}
    volume0::Vector{Float64}
    mass::Vector{Float64}

    function _compute_particle_volumes!(volumes::AbstractVector{<:Real}, position::AbstractVecOrMat{<:Real})
        volumes .= 1

        dim = size(position, 2)
        np = size(position, 1)
        for i = 1:dim
            diff_i = sum(diff(position[:, i])) / np
            volumes .*= diff_i
        end
        return volumes
    end

    function _compute_particle_volumes(position::AbstractVecOrMat{<:Real})
        volume = Vector{Float64}(undef, size(position, 1))
        _compute_particle_volumes!(volume, position)
        return volume
    end
    
    function Particles(position::AbstractVecOrMat{<:Real}, ρ::Real)
        dim = size(position, 2)
        if dim === 0
            throw(ArgumentError("Spatial dimension cannot be 0"))
        end
        np = size(position, 1)

        particle_volumes = _compute_particle_volumes(position)
        velocity = zeros(np, dim)
        displacement = zeros(np, dim)
        strain = zeros(np, dim*dim)
        stress = zeros(np, dim*dim)
        deformation = get_identity_matrix(np, dim)
        
        # strain = [zeros(dim, dim) for i = 1:np]
        # stress = [zeros(dim, dim) for i = 1:np]
        # deformation = [Matrix{Float64}(I, dim, dim) for i = 1:np]
        # deformation[:, 2:3] .= 0

        return new{dim, np}(position, 
                            velocity,
                            displacement,
                            strain,
                            stress,
                            deformation,
                            ones(np) * ρ,
                            particle_volumes,
                            copy(particle_volumes),
                            ρ .* particle_volumes)
    end
end

function initialize_uniform_particles(grid::MPMGrid{dim}, ρ::Real, np::Int64...; onedge::Bool = false) where {dim}
    position = Vector{Vector{Float64}}(undef, dim)
    dx = Vector{Float64}(undef, dim)
    if length(np) !== dim
        throw(ArgumentError("MPM grid dimensions should be equal to length(np)"))
    end
    for i = 1:dim
        if onedge
            dx[i] = (grid.splines[i].knot_vector[end] - grid.splines[i].knot_vector[begin]) / (np[i] - 1)
            position[i] = range(grid.splines[i].knot_vector[begin], stop=grid.splines[i].knot_vector[end], length=np[i])
        else
            dx[i] = (grid.splines[i].knot_vector[end] - grid.splines[i].knot_vector[begin]) / np[i]
            position[i] = range(grid.splines[i].knot_vector[begin] + dx[i] / 2, stop=(grid.splines[i].knot_vector[end] - dx[i] / 2), length=np[i])
        end
    end
    tensor_prod_coordinates!(position)
    positions = zeros(length(position[1]), dim)
    for i = 1:dim
        positions[:, i] .= position[i]
    end
    particles = Particles(positions, ρ)
    particles.ρ .= ρ
    particles.volume .= prod(dx)
    particles.mass .= particles.volume .* particles.ρ
    return particles
end

function initialize_uniform_particles(ρ::Real, lbound::Real, ubound::Real, np::Int)
    corners = [(lbound, ubound)]
    initialize_uniform_particles(ρ, corners, np)
end

function initialize_uniform_particles(ρ::Real, corners::AbstractVector{T}, np::Int64...) where T<:Tuple{<:Real, <:Real}
    if length(corners) <= 2
        throw(ErrorException("Unable to populate 2 dimensional space with only 2 corners"))
    elseif length(corners) > 4
        throw(NotImplementedException("More then 4 corners is not supported"))
    end
    positions = populate_square_or_triangle(corners, np[1], np[2])
    total_volume = opp_square_or_triangle(corners)
    particles = Particles(positions, ρ)
    particles.ρ .= ρ
    particles.volume .= total_volume ./ nparticles(particles)
    particles.mass .= particles.volume .* particles.ρ
    return particles
end


ndof(mpmgrid::MPMGrid) = ndof(mpmgrid.splines)

function generate_particles_uniform(grid::MPMGrid{2}, np1::Int, np2::Int, ρ::Real)
    bounds = grid.bounds[1]
    dp = (bounds[2] - bounds[1]) / np / 2
    x = collect(range(bounds[1]+dp, stop=bounds[2]-dp, length=np))
    particles = Particles(grid, x, ρ)
    Vp = diff([bounds[1]; (x[2:end] + x[1:(end-1)]) / 2; bounds[2]])
    particles.volume = Vp
    particles.mass = Vp * ρ
    return particles
end

function ndim(grid::MPMGrid{dim}) where dim
    return dim
end

function ndim(particles::Particles{dim, np}) where {dim, np}
    return dim
end

function nparticles(particles::Particles{dim, np}) where {dim, np}
    return np
end

function initialize_spline_storage(particles::Particles{dim, np}, mpmgrid::MPMGrid{dim}; sparse=false) where {dim, np}
    initialize_spline_storage(np, mpmgrid)
end

function initialize_spline_storage(nparticles::Int, mpmgrid::MPMGrid{dim}; sparse=false) where {dim}
    if dim == 1
        initialize_spline_storage(nparticles, mpmgrid.splines[1]; sparse=sparse)
    elseif dim == 2
        initialize_spline_storage(nparticles, mpmgrid.splines[1], mpmgrid.splines[2]; sparse=sparse)
    else
        throw(DimNotImplementedException())
    end
end


struct DirichletBoundaryConditions{dim}
    scalar_indices::Vector{Int64}
    vector_indices::Vector{Int64}
end

function get_boundary_indices_left(mpmgrid::MPMGrid{dim}) where dim
    if dim == 1
        return [1]
    elseif dim == 2
        return 1:mpmgrid.splines[1].ndof:ndof(mpmgrid)
    else 
        throw(DimNotImplementedException())
    end
end

function get_boundary_indices_right(mpmgrid::MPMGrid{dim}) where dim
    if dim == 1
        return [mpmgrid.splines[1].ndof]
    elseif dim == 2
        return mpmgrid.splines[1].ndof:mpmgrid.splines[1].ndof:ndof(mpmgrid)
    else 
        throw(DimNotImplementedException())
    end
end

function get_boundary_indices_top(mpmgrid::MPMGrid{2})
    ndof_1 = ndof(mpmgrid.splines[1])
    total = ndof(mpmgrid)
    return (total - ndof_1 + 1):total
end

function get_boundary_indices_bottom(mpmgrid::MPMGrid{2})
    # ndof_1 = ndof(mpmgrid.splines[1])
    return 1:mpmgrid.splines[1].ndof
end
    
function get_boundary_indices(mpmgrid::MPMGrid{1}; fix_left::Bool = false, fix_right::Bool = false) 
    indices_1 = Vector{Int64}(undef, 0)
    ndof_1 = ndof(mpmgrid.splines[1])
    if fix_left
        indices_1 = get_boundary_indices_left(mpmgrid)
    end
    if fix_right
        indices_1 = union(indices_1, get_boundary_indices_right(mpmgrid))
    end
    return DirichletBoundaryConditions{1}(indices_1, indices_1)
end

function get_boundary_indices(mpmgrid::MPMGrid{2}; fix_left::Bool = false, 
    fix_right::Bool = false, fix_top::Bool = false, fix_bottom::Bool = false) 
    indices_1 = Vector{Int64}(undef, 0)
    indices_2 = Vector{Int64}(undef, 0)
    total = ndof(mpmgrid)
    if fix_left
        indices_1 = get_boundary_indices_left(mpmgrid)
    end
    if fix_right
        indices_1 = union(indices_1, get_boundary_indices_right(mpmgrid))
    end
    if fix_top 
        indices_2 = get_boundary_indices_top(mpmgrid)
    end
    if fix_bottom
        indices_2 = union(indices_2, get_boundary_indices_bottom(mpmgrid))
    end
    vector_indices = sort(unique([indices_1; indices_2 .+ total]))
    scalar_indices = sort(unique([indices_1; indices_2]))
    return DirichletBoundaryConditions{2}(scalar_indices, vector_indices)
end

# function get_boundary_indices(mpmgrid::MPMGrid{2}; fix_left::Bool = false, 
#     fix_right::Bool = false, fix_top::Bool = false, fix_bottom::Bool = false) 
#     indices_1 = Vector{Int64}(undef, 0)
#     indices_2 = Vector{Int64}(undef, 0)
#     ndof_1 = ndof(mpmgrid.splines[1])
#     ndof_2 = ndof(mpmgrid.splines[2])
#     total = ndof(mpmgrid)
#     if fix_left
#         indices_1 = [indices_1; 1:ndof_1:total]
#     end
#     if fix_right
#         indices_1 = [indices_1; ndof_1:ndof_1:total]
#     end
#     if fix_top 
#         indices_2 = [indices_2; (total - ndof_1 + 1):total]
#     end
#     if fix_bottom
#         indices_2 = [indices_2; 1:ndof_1]
#     end
#     vector_indices = sort(unique([indices_1; indices_2 .+ total]))
#     scalar_indices = sort(unique([indices_1; indices_2]))
#     return DirichletBoundaryConditions{2}(scalar_indices, vector_indices)
# end
