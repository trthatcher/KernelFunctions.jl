using Zygote
using ForwardDiff
using Flux.Tracker
using BenchmarkTools
using LinearAlgebra
using MLKernels

l = 0.5
X = rand(10,2)
Y = rand(100,2)
function approxK(l)
    k = SquaredExponentialKernel(l)
    Kx = kernelmatrix(k,X)
    Ky = kernelmatrix(k,Y)
    Kxy = kernelmatrix(k,X,Y)
    tr(Ky-Kxy'*inv(Kx)*Kxy)
end

approxK([l,l+0.5])
## Testing
# Zygote.gradient(trK,l)
# Zygote not working because of copytru

ForwardDiff.gradient(approxK,[l,l+0.5])
# Works nicely with everything
Tracker.data(Tracker.gradient(approxK,[l,l+0.5]))
# Only works for iso kernels

##Performance
# @btime Zygote.gradient(trK,l)
@btime ForwardDiff.gradient(approxK,[$l,$l+0.5]);
@btime Tracker.data(Tracker.gradient(approxK,[$l,$l+0.5]))[1];
