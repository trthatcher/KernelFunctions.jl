# Abstract Exponential Kernel ==============================================================

abstract type AbstractExponentialKernel{T<:Real} <: MercerKernel{T} end

@inline basefunction(::AbstractExponentialKernel) = SquaredEuclidean()


# Exponential Kernel =======================================================================
@doc raw"""
    ExponentialKernel([α=1])

The exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(α‖x-y‖)   α > 0
```

where `α` is a positive scaling parameter. See also [`SquaredExponentialKernel`](@ref) for
a related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentialKernel()
ExponentialKernel{Float64}(1.0)

julia> ExponentialKernel(2.0f0)
ExponentialKernel{Float32}(2.0)
```
"""
struct ExponentialKernel{T<:Real,A} <: AbstractExponentialKernel{T}
    α::A
    function ExponentialKernel{T}(α::Union{Real,AbstractVector{<:Real}}=T(1)) where {T<:Real}
        @check_args(ExponentialKernel, α, count(α .<= zero(T)) == 0, "α > 0")
        return new{T,typeof(α)}(α.^2)
    end
end

ExponentialKernel(α::Union{T,AbstractVector{T}}=1.0) where {T<:Real} = ExponentialKernel{promote_float(T)}(α)

@inline kappa(κ::ExponentialKernel{T}, d²::T) where {T} = exp(-√(d²))

function Base.convert(::Type{K}, κ::ExponentialKernel) where {K>:ExponentialKernel{T,A} where A} where T
    return ExponentialKernel{T}(T.(sqrt.(κ.α)))
end

"""
    LaplacianKernel([α=1])

Alias for [`ExponentialKernel`](@ref).
"""
const LaplacianKernel = ExponentialKernel


# Squared Exponential Kernel ===============================================================
@doc raw"""
    SquaredExponentialKernel([α=1])

The squared exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(α‖x-y‖²)   α > 0
```

where `α` is a positive scaling parameter. See also [`ExponentialKernel`](@ref) for a
related form of the kernel or [`GammaExponentialKernel`](@ref) for a generalization.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> SquaredExponentialKernel()
SquaredExponentialKernel{Float64}(1.0)

julia> SquaredExponentialKernel(2.0f0)
SquaredExponentialKernel{Float32}(2.0)
```
"""
struct SquaredExponentialKernel{T<:Real,A} <: AbstractExponentialKernel{T}
    α::A
    function SquaredExponentialKernel{T}(α::Union{Real,AbstractVector{<:Real}}=T(1)) where {T<:Real}
        @check_args(SquaredExponentialKernel, α, all(α .> zero(T)), "α > 0")
        return new{T,typeof(α)}(α)
    end
end

function SquaredExponentialKernel(α::Union{T,AbstractVector{T}}=1.0) where {T<:Real}
    # SquaredExponentialKernel{T}(α)
    SquaredExponentialKernel{promote_float(T)}(α)
end

@inline kappa(κ::SquaredExponentialKernel{T}, d²::T) where {T} = exp(-d²)

function convert(
        ::Type{K},
        κ::SquaredExponentialKernel
    ) where {K>:SquaredExponentialKernel{T,A} where {T,A}}
    return SquaredExponentialKernel{T}(T.(κ.α))
end

"""
    GaussianKernel([α=1])

Alias of [`SquaredExponentialKernel`](@ref).
"""
const GaussianKernel = SquaredExponentialKernel

"""
    RadialBasisKernel([α=1])

Alias of [`SquaredExponentialKernel`](@ref).
"""
const RadialBasisKernel = SquaredExponentialKernel


# Gamma Exponential Kernel =================================================================
@doc raw"""
    GammaExponentialKernel([α=1 [,γ=1]])

The ``\gamma``-exponential kernel is an isotropic Mercer kernel given by the formula:

```
    κ(x,y) = exp(α‖x-y‖²ᵞ)   α > 0, γ ∈ (0,1]
```
where `α` is a scaling parameter and `γ` is a shape parameter of the Euclidean distance.
When `γ = 1` use [`SquaredExponentialKernel`](@ref) and [`ExponentialKernel`](@ref)
when `γ = 0.5` since these are more efficient implementations.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> GammaExponentialKernel()
GammaExponentialKernel{Float64}(1.0,1.0)

julia> GammaExponentialKernel(2.0f0)
GammaExponentialKernel{Float32}(2.0,1.0)

julia> GammaExponentialKernel(2.0, 0.5)
GammaExponentialKernel{Float64}(2.0,0.5)
```
"""
struct GammaExponentialKernel{T<:Real,A} <: AbstractExponentialKernel{T}
    α::A
    γ::T
    function GammaExponentialKernel{T}(α::Union{Real,AbstractVector{<:Real}}=T(1), γ::Real=T(1)) where {T<:Real}
        @check_args(GammaExponentialKernel, α, count(α .<= zero(T))==0, "α > 0")
        @check_args(GammaExponentialKernel, γ, one(T) >= γ > zero(T), "γ ∈ (0,1]")
        return new{T,typeof(α)}(α.^(-γ), γ)
    end
end

function GammaExponentialKernel(α::Union{T₁,AbstractVector{T₁}}=1.0, γ::T₂=T₁(1)) where {T₁<:Real, T₂<:Real}
    return GammaExponentialKernel{promote_float(T₁,T₂)}(α, γ)
end

@inline kappa(κ::GammaExponentialKernel{T}, d²::T) where {T} = exp(-d²^κ.γ)

function convert(
        ::Type{K},
        κ::GammaExponentialKernel
    ) where {K>:GammaExponentialKernel{T,A} where A} where T
    return GammaExponentialKernel{T}(T.(κ.α.^(κ.γ)), T.(κ.γ))
end
