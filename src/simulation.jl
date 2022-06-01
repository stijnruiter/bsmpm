include("data_structures.jl")

struct RunParameters
    apply_bc_before_node_computation::Bool
    apply_bc_after_node_computation::Bool
    mass_lumping::Bool
    volume_update::Symbol
end

struct Model{dim}
    ρ::Real
    E::Real
    ν::Real
    dt::Real
    tN::Int
    tend::Real
    runparams::RunParameters
    dirichlet::DirichletBoundaryConditions{dim}
    body_force::Function
    constitutive_model::Function
    traction_force::Function
end

struct LameParameters
    λ::Real
    μ::Real
end



no_traction_force(t::Real, model::Model{dim}, particles::Particles{dim, np}, mpmgrid::MPMGrid{dim}, splines::BasisSplineStorage) where {np, dim} = zeros(ndof(splines), dim)
no_body_froce(t::Real, model::Model{dim}, particles::Particles{dim, np}, mpmgrid::MPMGrid{dim}, splines::BasisSplineStorage) where {np, dim} = zeros(ndof(splines), dim)

function compute_lame_parameters(model::Model)::LameParameters
    compute_lame_parameters(model.E, model.ν)
end
function compute_lame_parameters(E::Real, ν::Real)::LameParameters
    μ = E / (2 * (1 + ν))
    λ = ν * E / ((1 + ν) * (1 - 2 * ν))
    LameParameters(λ, μ)
end


function hooke_linear_elastic(t::Real, model::Model{1}, particles::Particles{1, np}, mpmgrid::MPMGrid{1}, splines::BasisSplineStorage, strainrate::AbstractVecOrMat) where np 
    # particles.σ + (model.E .+ particles.σ) .* strainrate
    # With ν = 0, λ = 0, μ = E / 2, thus
    # λ tr(ϵ)I + 2μϵ = Eϵ
    particles.σ + model.E * strainrate
end
function hooke_linear_elastic(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::BasisSplineStorage, strainrate::AbstractVecOrMat) where np 
    lame = compute_lame_parameters(model)
    # trace = lame.λ * compute_mat_trace(strainrate)
    # particles.σ[:, 1] += trace
    # particles.σ[:, 4] += trace
    return particles.σ + lame.λ * compute_mat_trace(strainrate) .* get_identity_matrix(np, 2) + 2 * lame.μ * strainrate 
end

function hooke_linear_elastic_deform(t::Real, model::Model{dim}, particles::Particles{dim, np}, mpmgrid::MPMGrid{dim}, splines::BasisSplineStorage, strainrate::AbstractVecOrMat) where {np, dim} 
    lame = compute_lame_parameters(model)
    J = compute_determinant(particles.deformation)
    symFminI = compute_symmetric(particles.deformation) - get_identity_matrix(np, dim)

    return lame.λ ./ J .* compute_mat_trace(symFminI) .* particles.deformation + 2 * lame.μ ./ J .* particles.deformation .* symFminI
end

function hooke_large_deformation(t::Real, model::Model{1}, particles::Particles{1, np}, mpmgrid::MPMGrid{1}, splines::BasisSplineStorage, strainrate::AbstractVecOrMat) where np 
    particles.σ + (model.E .+ particles.σ) .* strainrate
end

function hooke_large_deformation(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::BasisSplineStorage, strainrate::AbstractVecOrMat) where np 
    particles.σ + (model.E .+ particles.σ) .* strainrate
end

function neo_hookean(t::Real, model::Model{1}, particles::Particles{1, np}, mpmgrid::MPMGrid{1}, splines::BasisSplineStorage, strainrate::AbstractVecOrMat) where np 
    #
    # With ν = 0, λ = 0, μ = E / 2, thus
    # σ = λlog(J)/J I + μ/J(FF^T-1) = μ/J(FF^T-1) = μ/F(F^2-1)
    # lame = compute_lame_parameters(model)
    # J = compute_determinant(particles.deformation)
    # particles.σ + (model.E / 2) ./ J .* (particles.deformation.*particles.deformation .- 1)
    # particles.σ[:, 1] += (lame.λ * log(J) - lame.μ) ./ J
    # particles.σ[:, 4] += (lame.λ * log(J) - lame.μ) ./ J

    # particles.σ + (lame.λ * log(J) - lame.μ) ./ J * get_identity_matrix(np, 2) + compute_matrix_product_transposed(particles.deformation, particles.deformation) ./ J
    lame = compute_lame_parameters(model)
    J = compute_determinant(particles.deformation)
    # particles.σ[:, 1] += (lame.λ * log(J) - lame.μ) ./ J
    # particles.σ[:, 4] += (lame.λ * log(J) - lame.μ) ./ J

    particles.σ + (lame.λ * log.(J) .- lame.μ) ./ J .* get_identity_matrix(np, 1) + compute_matrix_product_transposed(particles.deformation, particles.deformation) ./ J
end

function neo_hookean(t::Real, model::Model{2}, particles::Particles{2, np}, mpmgrid::MPMGrid{2}, splines::BasisSplineStorage, strainrate::AbstractVecOrMat) where np 
    lame = compute_lame_parameters(model)
    J = compute_determinant(particles.deformation)
    # particles.σ[:, 1] += (lame.λ * log(J) - lame.μ) ./ J
    # particles.σ[:, 4] += (lame.λ * log(J) - lame.μ) ./ J

    particles.σ + (lame.λ * log.(J) .- lame.μ) ./ J .* get_identity_matrix(np, 2) + compute_matrix_product_transposed(particles.deformation, particles.deformation) ./ J
end


function initialize_model_1D(ρ::Real, E::Real, dt::Real, number_of_timesteps::Int;
    dirichlet::DirichletBoundaryConditions{1} = DirichletBoundaryConditions{1}([], []),
    body_force::Function = no_body_froce, 
    constitutive_model::Function = hooke_linear_elastic,
    traction_force::Function = no_traction_force,
    runparams::RunParameters = RunParameters(true, false, false, :strain))
    ν = 0
    return Model{1}(ρ, E, ν, dt, number_of_timesteps, dt * number_of_timesteps, runparams, dirichlet, body_force, constitutive_model, traction_force)
end

function initialize_model_2D(ρ::Real, E::Real, ν::Real, dt::Real, number_of_timesteps::Int;
    dirichlet::DirichletBoundaryConditions{2} = DirichletBoundaryConditions{2}([], []),
    body_force::Function = no_body_froce, 
    constitutive_model::Function = hooke_linear_elastic,
    traction_force::Function = no_traction_force,
    runparams::RunParameters = RunParameters(true, false, false, :strain))
    ν = 0
    return Model{2}(ρ, E, ν, dt, number_of_timesteps, dt * number_of_timesteps, runparams, dirichlet, body_force, constitutive_model, traction_force)
end

function compute_internal_force(splines::BasisSplineStorage1D, particles::Particles)
    splines.dB' * (particles.σ .* particles.volume)
end
function compute_internal_force(splines::BasisSplineStorage2D, particles::Particles)
    [(splines.dB1' * (particles.σ[:, 1] .* particles.volume) + splines.dB2' * (particles.σ[:, 2] .* particles.volume)) (splines.dB1' *  (particles.σ[:, 3] .* particles.volume) + splines.dB2' *  (particles.σ[:, 4] .* particles.volume))]
end

function compute_velocity_gradient(splines::BasisSplineStorage1D, nodal_velocity::AbstractVecOrMat{Float64})
    splines.dB * nodal_velocity
end
function compute_velocity_gradient(splines::BasisSplineStorage2D, nodal_velocity::AbstractMatrix{Float64})
    view([splines.dB1 * nodal_velocity splines.dB2 * nodal_velocity], :, [1, 3, 2, 4])
end

function get_active_inactive_ndofs(A::AbstractMatrix)
    inactive = emptyRow(A)
    active = inactive .== 0
    return active, inactive
end

function nextstep!(t::Real, model::Model{dim}, particles::Particles{dim, np}, mpmgrid::MPMGrid{dim}, splines::BasisSplineStorage) where {dim, np}
    compute_bspline_values!(splines, particles.position, mpmgrid.splines)

    # Calculate the mass matrix
    M = (splines.B .* particles.mass)' * splines.B

    if model.runparams.mass_lumping
        M = diagm(vec(sum(M, dims=2)))
    end

    active, inactive = get_active_inactive_ndofs(M)


    # Calculate the force at the degrees of freedom
    F_int = compute_internal_force(splines, particles)

    F_ext = model.body_force(t, model, particles, mpmgrid, splines) + model.traction_force(t, model, particles, mpmgrid, splines)
    F = F_ext - F_int;

    F[inactive,:] .= 0

    # Calculate linear momentum
    L = splines.B' * (particles.mass .* particles.velocity) + F * model.dt
    L[inactive,:] .= 0

    if model.runparams.apply_bc_before_node_computation
        mass_val = tr(M) / ndof(mpmgrid)
        M[dirichlet.scalar_indices, :] .= 0
        M[:, dirichlet.scalar_indices] .= 0
        for i ∈ dirichlet.scalar_indices
            M[i, i] = mass_val
        end
        L[dirichlet.vector_indices] .= 0
        F[dirichlet.vector_indices] .= 0
    end

    # Calculate the acceleration at the nodes
    A = zeros(ndof(mpmgrid), dim)
    v_I = zeros(ndof(mpmgrid), dim)
    A[active,:] = M[active, active] \ F[active,:]
    v_I[active,:] = M[active, active] \ L[active,:]

    if model.runparams.apply_bc_after_node_computation
        v_I[dirichlet.vector_indices] .= 0
        A[dirichlet.vector_indices] .= 0
    end

    # update particles
    particles.velocity += splines.B * A * model.dt
    particles.displacement += splines.B * v_I * model.dt
    particles.position += splines.B * v_I * model.dt

    # Strain increment
    ∇VI = compute_velocity_gradient(splines, v_I)
    dϵp = compute_symmetric(∇VI) * model.dt

    particles.deformation += compute_matrix_product(∇VI*model.dt, particles.deformation)

    # Update particle strain and stress
    particles.ϵ += dϵp
    # particles.σ += (model.E .+ particles.σ)  .* dϵp

    particles.σ = model.constitutive_model(t, model, particles, mpmgrid, splines, dϵp)

    # Update volume and density using the strain increment
    if model.runparams.volume_update == :deformation
        particles.volume = compute_determinant(particles.deformation) .* particles.volume0
    elseif model.runparams.volume_update == :strain
        particles.volume = vec((1 .+ compute_determinant(dϵp)) .* particles.volume)
    else
        throw(NotImplementedException("Only :deformation and :strain are available as volume update"))
    end

    # particles.ρ = particles.mass ./ particles.volume
    # d["particles"].ρ = d["particles"].ρ ./ (1 .+ dϵp)
end

function run(model::Model, particles::Particles{dim, np}, mpmgrid::MPMGrid{dim}, splines::BasisSplineStorage) where {dim, np}
    t = 0
    println("Simulation starting")
    println("Δt=$(model.dt)s")
    println("t_end=$(model.dt*model.tN)s")
    print("Progress: 0/$(model.tN)")
    t_procent = max(floor(Int64, model.tN / 100), 1)
    for i = 1:model.tN
        nextstep!(t, model, particles, mpmgrid, splines)
        t = i * model.dt
        if i % t_procent == 0
            print("\e[2K") # rm line
            print("\e[1G") # curson start
            print("Progress: $(i)/$(model.tN)")
        end
    end
    print("\e[2K")
    print("\e[1G")
    println("Progress: Done")
    return t
end
