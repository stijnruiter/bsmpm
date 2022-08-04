using LinearAlgebra, SparseArrays
using BSplineKit

include("exception.jl")
include("algebra.jl")

# AbstractBasisSplineStorage1D{np, ni} = Union{BasisSplineDenseStorage1D{np, ni}, BasisSplineSparseStorage1D{np, ni}}
# AbstractBasisSplineStorage2D{np, ni} = Union{BasisSplineDenseStorage2D{np, ni}, BasisSplineSparseStorage2D{np, ni}}
# BasisSplineStorage{np, ni} = Union{AbstractBasisSplineStorage1D{np, ni}, AbstractBasisSplineStorage2D{np, ni}}

abstract type AbstractBasisSplineStorage{np, ni} end
abstract type AbstractBasisSplineStorage1D{np, ni} <: AbstractBasisSplineStorage{np, ni} end
abstract type AbstractBasisSplineStorage2D{np, ni} <: AbstractBasisSplineStorage{np, ni} end
 
struct BasisSpline
    ndof::Int64
    lbound::Float64
    ubound::Float64
    knot_vector::Vector{Float64}
    degree::Int64

    function check_knot_vector(knot_vector::AbstractVector{<:Real}, degree::Int64)
        ndof = length(knot_vector) - degree - 1
        if ndof < 1
            throw(InvalidDefinitionException("Degrees of freedom should be positive"))
        elseif degree < 0
            throw(InvalidDefinitionException("Polynomial degree cannot be negative"))
        elseif !ismonotonic(knot_vector)
            throw(InvalidDefinitionException("The knot vector should be monotonically increasing"))
        elseif !(length(unique(knot_vector[begin:(degree+1)])) == length(unique(knot_vector[(end - degree):end])) == 1)
            throw(InvalidDefinitionException("Knot vector has not enough repeating begin and start values"))
        end
    end

    function BasisSpline(lbound::Real, ubound::Real, nnodes::Int, degree::Int)
        knot_vector = range(lbound, stop=ubound, length=nnodes)
        knot_vector = [lbound * ones(degree); knot_vector; ubound * ones(degree)]
        # ndof = length(knot_vector) - degree - 1
        ndof = nnodes + degree - 1
        new(ndof, lbound, ubound, knot_vector, degree)
    end
    function BasisSpline(knot_vector::AbstractVector{<:Real}, degree::Int64)
        check_knot_vector(knot_vector, degree)
        ndof = length(knot_vector) - degree - 1
        new(ndof, knot_vector[begin], knot_vector[end], knot_vector, degree)
    end
end

mutable struct BasisSplineDenseStorage1D{np, ni} <: AbstractBasisSplineStorage1D{np, ni}
    B::Matrix{Float64}
    dB::Matrix{Float64}
    active::BitVector
end

mutable struct BasisSplineDenseStorage2D{np, ni} <: AbstractBasisSplineStorage2D{np, ni}
    B::Matrix{Float64}
    dB1::Matrix{Float64}
    dB2::Matrix{Float64}
    temp_store::Matrix{Float64}
    splines1::BasisSplineDenseStorage1D{np, ni1} where ni1
    splines2::BasisSplineDenseStorage1D{np, ni2} where ni2
    active::BitVector
end

mutable struct BasisSplineSparseStorage1D{np, ni} <: AbstractBasisSplineStorage1D{np, ni}
    B::SparseMatrixCSC{Float64, Int64}
    dB::SparseMatrixCSC{Float64, Int64}
    active::BitVector
end

mutable struct BasisSplineSparseStorage2D{np, ni} <: AbstractBasisSplineStorage2D{np, ni}
    B::SparseMatrixCSC{Float64, Int64}
    dB1::SparseMatrixCSC{Float64, Int64}
    dB2::SparseMatrixCSC{Float64, Int64}
    temp_store::SparseMatrixCSC{Float64, Int64}
    splines1::BasisSplineSparseStorage1D{np, ni1} where ni1
    splines2::BasisSplineSparseStorage1D{np, ni2} where ni2
    active::BitVector
end



function nparticles(splineStorage::AbstractBasisSplineStorage{np, ni}) where {np, ni}
    return np
end

function ndof(splineStorage::AbstractBasisSplineStorage{np, ni}) where {np, ni}
    return ni
end

ndof(spline::BasisSpline) = spline.ndof
ndof(splines::AbstractVector{BasisSpline}) = prod(ndof.(splines))

function ndim(splineStorage::AbstractBasisSplineStorage1D)
    return 1
end
function ndim(splineStorage::BasisSplineSparseStorage1D)
    return 1
end

function ndim(splineStorage::AbstractBasisSplineStorage2D)
    return 2
end
function ndim(splineStorage::BasisSplineSparseStorage2D)
    return 2
end

function Base.isapprox(spline1::AbstractBasisSplineStorage, spline2::AbstractBasisSplineStorage)
    if typeof(spline1) !== typeof(spline2)
        return false
    elseif typeof(spline1) <: AbstractBasisSplineStorage1D || typeof(spline1) <: BasisSplineSparseStorage1D
        return  isapprox(spline1.B, spline2.B) &&
                isapprox(spline1.dB, spline2.dB)
    elseif typeof(spline1) <: AbstractBasisSplineStorage2D|| typeof(spline1) <: BasisSplineSparseStorage2D
        return  isapprox(spline1.B, spline2.B) &&
                isapprox(spline1.dB1, spline2.dB1) &&
                isapprox(spline1.dB2, spline2.dB2)
    else
        return false
    end
end

function check_implemented_dims(dim::Any)
    if dim !== 1 && dim !== 2
        throw(DimNotImplementedException())
    end
end

function initialize_spline_storage(nparticles::Int64, spline::BasisSpline; sparse=false)
    if sparse
        BasisSplineSparseStorage1D{nparticles, spline.ndof}(spzeros(nparticles, spline.ndof), spzeros(nparticles, spline.ndof), BitVector(ones(spline.ndof)))
    else
        BasisSplineDenseStorage1D{nparticles, spline.ndof}(zeros(nparticles, spline.ndof), zeros(nparticles, spline.ndof), BitVector(ones(spline.ndof)))
    end
end

function initialize_spline_storage(coordinates::AbstractVector{<:Real}, knot_vector::AbstractVector{<:Real}, bspline_degree::Int; sparse=false)
    initialize_spline_storage(coordinates, BasisSpline(knot_vector, bspline_degree); sparse=sparse)
end

function initialize_spline_storage(coordinates::AbstractVector{<:Real}, spline::BasisSpline; sparse=false)
    initialize_spline_storage(length(coordinates), spline; sparse=sparse)
end

function initialize_spline_storage(nparticles::Int64, spline1::BasisSpline, spline2::BasisSpline; sparse=false)
    storage1 = initialize_spline_storage(nparticles, spline1; sparse=sparse)
    storage2 = initialize_spline_storage(nparticles, spline2; sparse=sparse)
    ndof_t = spline1.ndof * spline2.ndof
    if sparse
        BasisSplineSparseStorage2D{nparticles, ndof_t}(spzeros(nparticles, ndof_t), spzeros(nparticles, ndof_t), spzeros(nparticles, ndof_t), spzeros(nparticles, ndof_t), storage1, storage2, BitVector(ones(ndof_t)))
    else
        BasisSplineDenseStorage2D{nparticles, ndof_t}(zeros(nparticles, ndof_t), zeros(nparticles, ndof_t), zeros(nparticles, ndof_t), zeros(nparticles, ndof_t), storage1, storage2, BitVector(ones(ndof_t)))
    end
end

function initialize_spline_storage(coordinates::AbstractVector{<:Real}, spline1::BasisSpline, spline2::BasisSpline; sparse=false)
    initialize_spline_storage(length(coordinates), spline1, spline2; sparse=sparse)
end

function _compute_coxdeboor_derivative_coefficients(knot::AbstractVector{<:Real}, knot_index::Int64, degree::Int64)
    C = degree / (knot[knot_index + degree] - knot[knot_index]);
    D = degree / (knot[knot_index + degree + 1] - knot[knot_index + 1]);
    if isnan(C) || isinf(C)
        C = 0;
    end
    if isnan(D) || isinf(D)
        D = 0;
    end
    return C, D
end

function _compute_coxdeboor_coefficients(position::Real, knot::AbstractVector{<:Real}, knot_index::Int64, degree::Int64)
    A = (position - knot[knot_index]) / (knot[knot_index + degree] - knot[knot_index]);
    B = (knot[knot_index + degree + 1] - position) / (knot[knot_index + degree + 1] - knot[knot_index + 1]);
    if isnan(A) || isinf(A)
        A = 0;
    end
    if isnan(B) || isinf(B)
        B = 0;
    end
    return A, B
end

function compute_bspline_values!(storage::AbstractBasisSplineStorage1D, coord::AbstractVecOrMat{<:Real}, spline::AbstractVector{BasisSpline}) 
    compute_bspline_values!(storage, coord, spline[begin])
end

function compute_bspline_values!(storage::AbstractBasisSplineStorage1D, coord::AbstractVecOrMat{<:Real}, spline::BasisSpline)
    storage.B .= 0
    storage.dB .= 0
    for i = 1:length(coord)
        _recursive_bspline_particle!(storage, coord, spline, spline.degree, i)
    end
    storage.active = (vec(all(storage.B .== 0, dims=1)).==0)
end

function _recursive_bspline_particle!(storage::AbstractBasisSplineStorage1D, coord::AbstractVecOrMat{<:Real}, spline::BasisSpline, deg::Int, p::Int)
    if deg == 0
        index = findlast(x->x<=coord[p], spline.knot_vector)
        if index <= spline.ndof
            storage.B[p, index] = 1
        else
            storage.B[p, end] = 1
            index = spline.ndof
        end
        return index
    else    
        index = _recursive_bspline_particle!(storage, coord, spline, deg-1, p)
        for j = (index-deg):index
            if deg == spline.degree
                C, D = _compute_coxdeboor_derivative_coefficients(spline.knot_vector, j, deg)
                if j < spline.ndof
                    storage.dB[p, j] = C*storage.B[p, j]-D*storage.B[p, j+1];
                else#if j == spline.ndof
                    storage.dB[p, j] = C*storage.B[p, j];
                end
            end
            A, B = _compute_coxdeboor_coefficients(coord[p], spline.knot_vector, j, deg)
            if j < spline.ndof
                storage.B[p, j] = A*storage.B[p, j]+B*storage.B[p, j+1];
            else#if j==spline.ndof 
                storage.B[p, j] = A*storage.B[p, j];
            end
        end
    end
    return index
end

function compute_bspline_values!(storage::BasisSplineSparseStorage2D, coord1::AbstractVector{<:Real}, coord2::AbstractVector{<:Real}, spline1::BasisSpline, spline2::BasisSpline)
    compute_bspline_values!(storage.splines1, coord1, spline1)
    compute_bspline_values!(storage.splines2, coord2, spline2)
    storage.B .= kron(ones(1, spline2.ndof),storage.splines1.B).*kron(storage.splines2.B, ones(1, spline1.ndof))
    storage.dB1 .= kron(ones(1, spline2.ndof),storage.splines1.dB).*kron(storage.splines2.B, ones(1, spline1.ndof))
    storage.dB2 .= kron(ones(1, spline2.ndof),storage.splines1.B).*kron(storage.splines2.dB, ones(1, spline1.ndof))
    storage.active = (vec(all(spline_storage.B .== 0, dims=1)).==0)
    return storage
end


function compute_bspline_values!(storage::BasisSplineDenseStorage2D, coord1::AbstractVector{<:Real}, coord2::AbstractVector{<:Real}, spline1::BasisSpline, spline2::BasisSpline)
    compute_bspline_values!(storage.splines1, coord1, spline1)
    compute_bspline_values!(storage.splines2, coord2, spline2)

    kron!(storage.B, ones(1, spline2.ndof), storage.splines1.B)
    kron!(storage.dB1, ones(1, spline2.ndof),storage.splines1.dB)
    kron!(storage.temp_store, storage.splines2.B, ones(1, spline1.ndof))
    kron!(storage.dB2, storage.splines2.dB, ones(1, spline1.ndof))

    storage.dB1 .*= storage.temp_store
    storage.dB2 .*= storage.B
    storage.B   .*= storage.temp_store

    storage.active = (vec(all(storage.B .== 0, dims=1)).==0)

    return storage
end

function compute_bspline_values!(storage::AbstractBasisSplineStorage2D, coord::AbstractMatrix{<:Real}, splines::AbstractVector{BasisSpline})
    compute_bspline_values!(storage, coord[:, 1], coord[:, 2], splines[1], splines[2])
end

generate_random_particles(spline::BasisSpline, np::Int) = spline.lbound .+ rand(np) * (spline.ubound - spline.lbound)

greville_abscissae(spline::BasisSpline) = [sum(spline.knot_vector[(i+1):(i+spline.degree)]) / spline.degree for i = 1:spline.ndof]
nnodes(spline::BasisSpline) = length(spline.knot_vector) - 2*spline.degree

function oppBspline(spline::BasisSpline)
    oppBspline(spline.lbound, spline.ubound, nnodes(spline), spline.degree)
end

function oppBspline(x1, x2, n, deg)
    splines = BSplineBasis(BSplineOrder(deg+1),range(x1, stop=x2, length=n))
    ndof = n + deg - 1
    opps = zeros(ndof)
    for i = 1:ndof
        coeff = zeros(ndof)
        coeff[i] = 1
        s = Spline(splines, coeff)
        integrate = integral(s)
        opps[i] = integrate(x2) - integrate(x1)
    end
    return opps
end