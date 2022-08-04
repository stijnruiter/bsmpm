
include("bspline.jl")
include("data_structures.jl")
include("set_operation.jl")
include("web_helper_functions.jl")

function Bspline(coord::Real, spline1d::BasisSpline, deg::Int)
    if deg == 0
        values = zeros(ndof(spline1d))
        for j=1:ndof(spline1d)
            values[j]=B0(coord, spline1d.knot_vector, j)
        end
        return values, zeros(ndof(spline1d))
    else    
        prev, dprev = Bspline(coord, spline1d, deg-1)
        for j=1:ndof(spline1d)
            C, D = _compute_coxdeboor_derivative_coefficients(spline1d.knot_vector, j, deg)
            if j < ndof(spline1d)
                dprev[j] = C*prev[j]-D*prev[j+1];
            else
                dprev[j] = C*prev[j];
            end

            A, B = _compute_coxdeboor_coefficients(coord, spline1d.knot_vector, j, deg)
            if j < ndof(spline1d)
                prev[j] = A*prev[j]+B*prev[j+1];
            else
                prev[j] = A*prev[j];
            end
        end
        return prev, dprev
        # N, dN #Bspline1D{T, np, ni}(dropzeros(sparse(N)), dropzeros(sparse(dN)))
    end
end
function B0(x::Real, knot::AbstractVector{<:Real}, i::Int64)
    if knot[i] <= x < knot[i+1]
        return 1
    elseif knot[i+1] == knot[end] && knot[i] <= x <= knot[i+1]
        return 1
    else 
        return 0
    end
end

function dBspline(coord::Real, spline1d::BasisSpline, deg::Int, nth::Int)
    if nth > deg
        return zeros(ndof(spline1d))
    elseif nth == 0
        return Bspline(coord, spline1d, deg)[1]
    elseif nth == 1
        return Bspline(coord, spline1d, deg)[2]
    else
        dB = dBspline(coord, spline1d, deg - 1, nth - 1);
        dN = zeros(ndof(spline1d))
        for j=1:ndof(spline1d)
            C, D = _compute_coxdeboor_derivative_coefficients(spline1d.knot_vector, j, deg)
            if j < ndof(spline1d)
                dN[j] = C*dB[j]-D*dB[j+1];
            else 
                dN[j] = C*dB[j]
            end
        end
        return dN;
    end
end

function web_splines!(spline_storage::AbstractBasisSplineStorage2D, mpmgrid::MPMGrid{2}, particles::Particles)
    web_splines!(spline_storage, mpmgrid, particles.position[particles.bel, :])
end

function web_splines!(spline_storage::AbstractBasisSplineStorage2D, mpmgrid::MPMGrid{2}, boundaryParticles::AbstractMatrix)
    grid_cells = get_grid_cells(mpmgrid)
    splines, grid_splines = identify_splines(mpmgrid, grid_cells)
    interior, boundary, exterior = identify_grid_cells(mpmgrid, grid_cells, boundaryParticles)
    stable_splines, unstable, exterior_splines = identify_spline_stability(mpmgrid, grid_cells, grid_splines, boundaryParticles)
    spline_storage.active .= copy(stable_splines)
    for j ∈ findall(unstable)
        suppBj = support(mpmgrid, j)
        closest_stable_grid_cell = find_closest_stable_basis_mid(grid_cells, interior, suppBj)
        stable_supp = grid_splines[closest_stable_grid_cell]

        Bi = grid_cells[closest_stable_grid_cell]


        vj_1 = (j - 1) % ndof(mpmgrid.splines[1]) + 1
        vj_2 = floor(Int, (j - 1) / ndof(mpmgrid.splines[1]) + 1)
        
        # vi_1 = (stable_supp .- 1) .% ndof(mpmgrid.splines[1]) .+ 1
        # vi_2 = floor.(Int, (stable_supp .- 1) ./ ndof(mpmgrid.splines[1]) .+ 1)
        rtaylor_1 = (Bi.lx + Bi.ux) / 2
        rtaylor_2 = (Bi.ly + Bi.uy) / 2

        e_ij_1 = compute_eij_1d(mpmgrid.splines[1], rtaylor_1, vj_1)
        # display(e_ij_1)
        e_ij_2 = compute_eij_1d(mpmgrid.splines[2], rtaylor_2, vj_2)

        e_ij = (kron(ones(ndof(mpmgrid.splines[2])), e_ij_1) .* kron(e_ij_2, ones(ndof(mpmgrid.splines[1]))))[stable_supp]

        # display(e_ij)
        spline_storage.B[:, stable_supp] += spline_storage.B[:, j] * e_ij'
        spline_storage.dB1[:, stable_supp] += spline_storage.dB1[:, j] * e_ij'
        spline_storage.dB2[:, stable_supp] += spline_storage.dB2[:, j] * e_ij'
    end
    # return stable_splines
end

function web_splines!(spline_storage::AbstractBasisSplineStorage1D, mpmgrid::MPMGrid{1}, particles::Particles)
    web_splines!(spline_storage, mpmgrid.splines[1], particles.position[particles.bel])
end

function web_splines!(spline_storage::AbstractBasisSplineStorage1D, spline::BasisSpline, bounds::AbstractVector)
    grid_cells = get_grid_cells(spline)
    splines, grid_splines = identify_splines(spline, grid_cells)
    interior, boundary, exterior = identify_grid_cells(spline, grid_cells, bounds)
    stable_splines, unstable, exterior_splines = identify_spline_stability(spline, grid_cells, grid_splines, bounds)
    spline_storage.active .= copy(stable_splines)
    for j ∈ findall(unstable)
    # j = findlast(unstable)
        suppBj = support(spline, j)
        closest_stable_grid_cell = find_closest_stable_basis_mid(grid_cells, interior, suppBj)
        stable_supp = grid_splines[closest_stable_grid_cell]
    
        Bi = grid_cells[closest_stable_grid_cell]
    
    
        vj_1 = (j - 1) % ndof(spline) + 1
        # vj_2 = floor(Int, (j - 1) / ndof(mpmgrid.splines[1]) + 1)
        
        # vi_1 = (stable_supp .- 1) .% ndof(mpmgrid.splines[1]) .+ 1
        # vi_2 = floor.(Int, (stable_supp .- 1) ./ ndof(mpmgrid.splines[1]) .+ 1)
        rtaylor_1 = sum(Bi) / 2
        # rtaylor_2 = (Bi.ly + Bi.uy) / 2
    
        e_ij = compute_eij_1d(spline, rtaylor_1, vj_1)
        # display(e_ij)
        # display(e_ij_1)
        # e_ij_2 = compute_eij_1d(mpmgrid.splines[2], rtaylor_2, vj_2)
    
        # e_ij = (kron(ones(ndof(mpmgrid.splines[2])), e_ij_1) .* kron(e_ij_2, ones(ndof(mpmgrid.splines[1]))))[stable_supp]
    
        # display(e_ij)

        # println("$(j) $(e_ij[stable_supp]) $(stable_supp)")
        spline_storage.B[:, stable_supp] += spline_storage.B[:, j] * e_ij[stable_supp]'
        spline_storage.dB[:, stable_supp] += spline_storage.dB[:, j] * e_ij[stable_supp]'
        # spline_storage.dB2 += spline_storage.dB2[:, j] * e_ij'
    end
end

function compute_eij_1d(spline::BasisSpline, rtaylor, j)
    alpha_k = compute_ak(spline, rtaylor)
    beta_j_k = β_k(spline, j)
    # display(alpha_k)
    # display(beta_j_k)
    e_ij = zeros(ndof(spline))
    for k = 0:spline.degree
        e_ij += (-1)^k * factorial(spline.degree - k) * beta_j_k[spline.degree - k + 1] * factorial(k) * alpha_k[:, k + 1];
    end
    e_ij /= factorial(spline.degree);
    return e_ij
end

function compute_ak(spline::BasisSpline, rtaylor::Real)
    deriv = dBspline.(rtaylor, Ref(spline), spline.degree, 0:spline.degree)
    a_m = reduce(hcat, deriv) ./ factorial.(0:spline.degree)'
    a_k = zeros(ndof(spline), spline.degree + 1)
    for k = 0:spline.degree
        for m = k:spline.degree
            a_k[:, k+1] += binomial(m, k) * a_m[:, m+1] * (-rtaylor) ^ (m-k)
        end
    end
    return a_k
end

function β_k(spline::BasisSpline, j::Int)#, knot::Array{Float64,1}, p::Int)
    β = zeros(spline.degree + 1);
    for k = 0:spline.degree
        for l = 1:binomial(spline.degree, k)
            product = 1
            for t = 0:(spline.degree - k - 1)
                product *= (spline.knot_vector[j + ((l + t - 1) % spline.degree) + 1]);
            end
            β[k + 1] += product;
        end
        β[k+1] *= (-1)^(spline.degree - k)
    end
    return β
end

function compute_webspline_values!(storage::AbstractBasisSplineStorage, mpmgrid::MPMGrid, particles::Particles)
    compute_bspline_values!(storage, particles.position, mpmgrid.splines)
    # display(storage.active)
    web_splines!(storage, mpmgrid, particles)
end