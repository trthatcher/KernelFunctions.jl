# Base Function Test

struct Euclidean <: Metric end

MLK.base_aggregate(::Euclidean, s::T, scale::T, x::T, y::T) where {T} = s + scale*(x-y)^2
MLK.base_return(::Euclidean, s::T) where {T} = sqrt(s)


# Base Function References

const base_functions = (
    SquaredEuclidean,
    ScalarProduct,
    Euclidean
)

const base_functions_initiate = Dict(
    SquaredEuclidean => 0,
    ScalarProduct    => 0,
    Euclidean        => 0
)

const base_functions_aggregate = Dict(
    SquaredEuclidean => (s,scale,x,y) -> s + scale*(x-y)^2,
    ScalarProduct    => (s,scale,x,y) -> s + scale*x*y,
    Euclidean        => (s,scale,x,y) -> s + scale*(x-y)^2
)

const base_functions_return = Dict(
    SquaredEuclidean => s -> s,
    ScalarProduct    => s -> s,
    Euclidean        => s -> sqrt(s)
)

const base_functions_properties = Dict(
                       #|stnry |isotrop
    ScalarProduct    => (false, false),
    SquaredEuclidean => (true,  true),
    Euclidean        => (false, false) # just default values test
)


# Kernel Function References

const kernel_functions = (
    ExponentialKernel,
    SquaredExponentialKernel,
    GammaExponentialKernel,
    RationalQuadraticKernel,
    GammaRationalQuadraticKernel,
    MaternKernel,
    PolynomialKernel,
    ExponentiatedKernel,
    PowerKernel,
    LogKernel,
    SigmoidKernel
)

const kernel_functions_arguments = Dict(
    ExponentialKernel            => ((1.0,),        (2.0,)         ),
    SquaredExponentialKernel     => ((1.0,),        (2.0,)),
    GammaExponentialKernel       => ((1.0,1.0),     (2.0,0.5)),
    RationalQuadraticKernel      => ((1.0,1.0),     (2.0,2.0)),
    GammaRationalQuadraticKernel => ((1.0,1.0,1.0), (2.0,2.0,0.5)),
    MaternKernel                 => ((1.0,1.0),     (2.0,2.0)),
    PolynomialKernel             => ((1.0,1.0,3),   (2.0,2.0,2)),
    ExponentiatedKernel          => ((1.0,),        (2.0,)),
    PowerKernel                  => ((1.0,),        (0.5,)),
    LogKernel                    => ((1.0,1.0),     (2.0,0.5)),
    SigmoidKernel                => ((1.0,1.0),     (2.0,2.0))
)

const kernel_functions_kappa = Dict(
    ExponentialKernel            => (z,α)     -> exp(-sqrt(z)),
    SquaredExponentialKernel     => (z,α)     -> exp(-z),
    GammaExponentialKernel       => (z,α,γ)   -> exp(-z^γ),
    RationalQuadraticKernel      => (z,α,β)   -> (1 + z)^(-β),
    GammaRationalQuadraticKernel => (z,α,β,γ) -> (1 + z^γ)^(-β),
    MaternKernel                 => (z,ν,ρ)   -> begin
                                                   d = √(z)
                                                   T = typeof(z)
                                                   d = d < eps(T) ? eps(T) : d
                                                   tmp1 = √(2*ν)*d
                                                   tmp2 = 2^(1 - ν)
                                                   tmp2*(tmp1^ν)*besselk(ν, tmp1)/gamma(ν)
                                                end,
    PolynomialKernel             => (z,a,c,d) -> (z+c)^d,
    ExponentiatedKernel          => (z,a)     -> exp(z),
    PowerKernel                  => (z,γ)     -> z^γ,
    LogKernel                    => (z,α,γ)   -> log(z^γ+1),
    SigmoidKernel                => (z,a,c)   -> tanh(z+c)
)

const kernel_functions_base = Dict(
    PolynomialKernel         => ScalarProduct,
    ExponentiatedKernel      => ScalarProduct,
    SigmoidKernel            => ScalarProduct
)
