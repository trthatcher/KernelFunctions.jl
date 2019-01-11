using MLKernels
using ForwardDiff, LinearAlgebra
using Random: seed!
seed!(42)
cd(dirname(@__FILE__))

k = SquaredExponentialKernel([1.0,3.0])
X= rand(10,2)
y = rand(100,2)

KXy = kernelmatrix(k,X,y)

KXY = kernelmatrix(k,X)

A = rand(10,10)
B = rand(100,100)
function tracekernel(x)
    k = SquaredExponentialKernel(x)
    KXY = kernelmatrix(k,X,y)
    KX = kernelmatrix(k,X)
    return tr(inv(KX+1e-7*I)*A*KX + KXY*B*KXY')
end

tracekernel([1.0,3.0])
ForwardDiff.gradient(tracekernel,[1.0,3.0])

k.α = [2.0,3.0]
