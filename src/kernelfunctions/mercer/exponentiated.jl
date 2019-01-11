@doc raw"""
    ExponentiatedKernel([α=1])

The exponentiated kernel is a Mercer kernel given by:

```
    κ(x,y) = exp(α⋅xᵀy)   α > 0
```
where `α` is a positive scaling parameter.

# Examples

```jldoctest; setup = :(using MLKernels)
julia> ExponentiatedKernel()
ExponentiatedKernel{Float64}(1.0)

julia> ExponentiatedKernel(2)
ExponentiatedKernel{Float64}(2.0)

julia> ExponentiatedKernel(2.0f0)
ExponentiatedKernel{Float32}(2.0)
```
"""
struct ExponentiatedKernel{T<:Real,A} <: MercerKernel{T}
    α::A
    function ExponentiatedKernel{T}(α::Union{Real,AbstractVector{Real}}=1.0) where {T<:Real}
        @check_args(ExponentiatedKernel, α, count(α .<= zero(T))==0, "α > 0")
        return new{T,typeof(α)}(α)
    end
end
ExponentiatedKernel(α::Union{T,AbstractVector{T}}=1.0) where {T<:Real} = ExponentiatedKernel{promote_float(T)}(α)

@inline basefunction(::ExponentiatedKernel) = ScalarProduct()

@inline kappa(κ::ExponentiatedKernel{T}, xᵀy::T) where {T} = exp(xᵀy)

function convert(
        ::Type{K},
        κ::ExponentiatedKernel
    ) where {K>:ExponentiatedKernel{T,A} where A} where T
    return ExponentiatedKernel{T}(T.(κ.α))
end
