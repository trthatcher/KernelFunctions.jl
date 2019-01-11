@doc raw"""
    PolynomialKernel([a=1 [,c=1 [,d=3]]])

The polynomial kernel is a Mercer kernel given by:

```
    κ(x,y) = (a⋅xᵀy + c)ᵈ   α > 0, c ≧ 0, d ∈ ℤ⁺
```

# Examples

```jldoctest; setup = :(using MLKernels)
julia> PolynomialKernel(2.0f0)
PolynomialKernel{Float32}(2.0,1.0,3)

julia> PolynomialKernel(2.0f0, 2.0)
PolynomialKernel{Float64}(2.0,2.0,3)

julia> PolynomialKernel(2.0f0, 2.0, 2)
PolynomialKernel{Float64}(2.0,2.0,2)
```
"""
struct PolynomialKernel{T<:Real,A} <: MercerKernel{T}
    α::A
    c::T
    d::T
    function PolynomialKernel{T}(
            a::Union{Real,AbstractVector{Real}}=T(1),
            c::Real=T(1),
            d::Real=T(3)
        ) where {T<:Real}
        @check_args(PolynomialKernel, a, count(a .<=  zero(T)) == 0, "a > 0")
        @check_args(PolynomialKernel, c, c >= zero(T), "c ≧ 0")
        @check_args(PolynomialKernel, d, d >= one(T) && d == trunc(d), "d ∈ ℤ₊")
        return new{T,typeof(a)}(a, c, d)
    end
end

function PolynomialKernel(
        a::Union{T₁,AbstractVector{T₁}}=1.0,
        c::T₂=T₁(1),
        d::T₃=convert(promote_float(T₁,T₂), 3)
    ) where {T₁<:Real,T₂<:Real,T₃<:Real}
    return PolynomialKernel{promote_float(T₁,T₂,T₃)}(a, c, d)
end

@inline basefunction(::PolynomialKernel) = ScalarProduct()

@inline function kappa(κ::PolynomialKernel{T}, xᵀy::T) where {T}
    return (xᵀy + κ.c)^(κ.d)
end

function convert(::Type{K}, κ::PolynomialKernel) where {K>:PolynomialKernel{T,A} where A} where T
    return PolynomialKernel{T}(T.(κ.α), T(κ.c), T(κ.d))
end
