@doc raw"""
    SigmoidKernel([a=1 [,c=1]])

The Sigmoid Kernel is given by:
```
    κ(x,y) = tanh(a⋅xᵀy + c)
```

# Examples

```jldoctest; setup = :(using MLKernels)
julia> SigmoidKernel()
SigmoidKernel{Float64}(1.0,1.0)

julia> SigmoidKernel(0.5f0)
SigmoidKernel{Float32}(0.5,1.0)

julia> SigmoidKernel(0.5f0, 0.5)
SigmoidKernel{Float64}(0.5,0.5)
```
"""
struct SigmoidKernel{T<:Real,A} <: Kernel{T}
    α::A
    c::T
    function SigmoidKernel{T}(a::Union{Real,AbstractVector{<:Real}}=T(1), c::Real=T(1)) where {T<:Real}
        @check_args(SigmoidKernel, a, count(a .<=  zero(T))==0, "a > 0")
        @check_args(SigmoidKernel, c, c >= zero(T), "c ≧ 0")
        return new{T,typeof(a)}(a, c)
    end
end
SigmoidKernel(a::Union{T₁,AbstractVector{T₁}}=1.0, c::T₂=T₁(1)) where {T₁<:Real,T₂<:Real} = SigmoidKernel{promote_float(T₁,T₂)}(a,c)

@inline basefunction(::SigmoidKernel) = ScalarProduct()

@inline kappa(κ::SigmoidKernel{T}, xᵀy::T) where {T} = tanh(xᵀy + κ.c)

function convert(::Type{K}, κ::SigmoidKernel) where {K>:SigmoidKernel{T,A} where A} where T
    return SigmoidKernel{T}(T.(κ.α), T(κ.c))
end
