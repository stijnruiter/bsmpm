using LinearAlgebra, SparseArrays

function ismonotonic(A::AbstractVector, cmp = <)
    current = A[begin] # begin instead of 1
    for i in axes(A, 1)[2:end] # skip the first element
        newval = A[i] # don't use copy here
        cmp(newval, current) && return false
        current = newval
    end
    return true
end

function get_identity_matrix(np::Int, dim::Int)
    if dim == 1
        return ones(np,1)
    elseif dim == 2
        return [ones(np) zeros(np, 2) ones(np)]
    else
        throw(DimNotImplementedException())
    end
end

function compute_symmetric!(matrix::AbstractVecOrMat{Float64})
    if size(matrix, 2) == 1
        return matrix
    elseif size(matrix, 2) == 4
        matrix[:, 2] = 0.5*(matrix[:, 2] + matrix[:, 3])
        matrix[:, 3] .= matrix[:, 2]
        return matrix
    else
        throw(DimNotImplementedException())
    end
end 


function compute_symmetric(matrix::AbstractVecOrMat{Float64})
    if size(matrix, 2) == 1
        return deepcopy(matrix)
    elseif size(matrix, 2) == 4
        mat = deepcopy(matrix)
        mat[:, 2] = 0.5*(matrix[:, 2] + matrix[:, 3])
        mat[:, 3] .= mat[:, 2]
        return mat
    else
        throw(DimNotImplementedException())
    end
end 

function compute_determinant(matrix::AbstractVecOrMat{Float64})
    if size(matrix, 2) == 1
        return vec(matrix)
    elseif size(matrix, 2) == 4
        return matrix[:, 1] .* matrix[:, 4] - matrix[:, 2] .* matrix[:, 3]
    else
        throw(DimNotImplementedException())
    end
end 

# function transpose_mat(A::AbstractVecOrMat)
#     if size(A, 2) == 1
#         return A
#     elseif size(A, 2) == 4
#         return view
# end

function compute_matrix_product(A::AbstractMatrix, B::AbstractMatrix)
    (np, dim) = size(A)
    if size(B) !== (np, dim)
        throw(DimensionMismatch("A*B expected B to be of size $((np, dim)), got $(size(B)) instead"))
    end
    if dim == 1
        return A .* B
    elseif dim == 4
        val = Matrix{Base.return_types(*, (eltype(A), eltype(B)))[1]}(undef, np, dim)
        val[:, 1:2] = A[:, 1] .* B[:, 1:2] + A[:, 2] .* B[:, 3:4]
        # val[:, 2] = A[:, 1] .* B[:, 2] + A[:, 2] .* B[:, 4]
        val[:, 3:4] = A[:, 3] .* B[:, 1:2] + A[:, 4] .* B[:, 3:4]
        # val[:, 4] = A[:, 2] .* B[:, 2] + A[:, 4] .* B[:, 4]
        return val
    else
        throw(DimNotImplementedException())
    end
end


function compute_matrix_product_transposed(A::AbstractMatrix, B::AbstractMatrix)
    (np, dim) = size(A)
    if size(B) !== (np, dim)
        throw(DimensionMismatch("A*B expected B to be of size $((np, dim)), got $(size(B)) instead"))
    end
    if dim == 1
        return A .* B
    elseif dim == 4
        val = Matrix{Base.return_types(*, (eltype(A), eltype(B)))[1]}(undef, np, dim)
        val[:, 1:2] = A[:, 1] .* B[:, [1, 3]] + A[:, 2] .* B[:, [2, 4]]
        # val[:, 2] = A[:, 1] .* B[:, 2] + A[:, 2] .* B[:, 4]
        val[:, 3:4] = A[:, 3] .* B[:, [1, 3]] + A[:, 4] .* B[:, [2, 4]]
        # val[:, 4] = A[:, 2] .* B[:, 2] + A[:, 4] .* B[:, 4]
        return val
    else
        throw(DimNotImplementedException())
    end
end

function compute_mat_trace(A::AbstractMatrix)
    dim = size(A, 2)
    if dim == 1
        return A
    elseif dim == 4
        return A[:, 1] .+ A[:, 4]
    else
        throw(DimNotImplementedException())
    end
end

function test_mat_mult(n::Int=10)
    for i = 1:n
        A = rand(2, 2)
        B = rand(2, 2)
        if view(compute_matrix_product(view(A, [1 3 2 4]), view(B, [1 3 2 4])), [1 2; 3 4]) != A*B
            throw(ErrorException("compute_matrix_product did not behave as expected for dim=2"))
        end
        A = rand(1, 1)
        B = rand(1, 1)
        if compute_matrix_product(A, B) != A * B
            throw(ErrorException("compute_matrix_product did not behave as expected for dim=1"))
        end
    end
    return true
end

function populate_square_or_triangle(corners::AbstractVector{T}, np1::Int, np2::Int;onedge::Bool = false) where T <: Tuple{<:Real, <:Real}
    s1 = 0
    s2 = 1
    t1 = 0
    t2 = 1
    if !onedge
        ds = 1/np1
        dt = 1/np2
        s1 = ds/2
        s2 = 1 - ds/2
        t1 = dt/2
        t2 = 1 - dt/2
    end
    
    s = range(s1, stop=s2, length=np1)
    t = range(t1, stop=t2, length=np2)
    n  = 1
    pos = zeros(length(s) * length(t), 2)
    bel = BitVector(undef, np1*np2)
    bel .= 0
    if length(corners) < 3
        throw(ErrorException("Not enough corners"))
    elseif length(corners) == 3
        for i ∈ s
            for j ∈ t
                pos[n,begin], pos[n, end] = (1-j)*i .*corners[1] .+ j .*i .* corners[2] .+ (1-i).* corners[3]
                if i == s1 || j == t1 || i == s2 || j == t2
                    bel[n] = 1
                end
                n += 1
            end
        end
    else
        for i ∈ s
            for j ∈ t
                pos[n,begin], pos[n, end] = (1-j) .*((1-i).*corners[1] .+ i .*corners[2]) .+ j.*((1-i).*corners[4] .+ i .*corners[3])
                if i == s1 || j == t1 || i == s2 || j == t2
                    bel[n] = 1
                end
                n += 1
            end
        end
    end
    return pos, bel
end
function opp_square_or_triangle(corners::AbstractVector{T}) where T<:Tuple{<:Real, <:Real}
    if length(corners) < 3
        throw(ErrorException("Not enough corners"))
    end
    A, B = corners[1] .- corners[2]
    C = - A*corners[1][1] - B*corners[1][2]

    h = abs(A*corners[3][1] + B*corners[3][2] + C) / sqrt(A^2 + B^2)
    if length(corners) == 3
        x1 = sqrt(sum((corners[1] .- corners[2]).^2))
        return 0.5 * x1 * h
    elseif length(corners) == 4
        x1 = sqrt(sum((corners[1] .- corners[2]).^2)) 
        x2 = sqrt(sum((corners[3] .- corners[4]).^2)) 
        return 0.5 * (x1 + x2) * h
    else 
        throw(ErrorException("Unhandled corners"))
    end
end


function tensor_prod_coordinates!(position::AbstractVector{<:AbstractVector{<:Real}})
    dim = length(position)
    n = Vector{Int64}(undef, dim)
    for i = 1:dim
        n[i] = length(position[i])
    end
    for i = 1:dim
        tmp = ones(n[i])
        for j = 1:(i-1)
            position[j] = kron(tmp, position[j])
        end
        for j = (i+1):dim
            position[j] = kron(position[j], tmp)
        end
    end
    return position
end

function emptyRow(mat,cutoff = 1e-10, d = 2)
    return dropdims(sum(mat.<=cutoff,dims=d).==size(mat)[d],dims=d)
end