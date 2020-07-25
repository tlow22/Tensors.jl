## Tensor constants
const ℤ32 = zero(SymmetricTensor{4,3,Float32,36})
const ℤ64 = zero(SymmetricTensor{4,3,Float64,36})
const ℤ32_full = zero(Tensor{4,3,Float32,81})
const ℤ64_full = zero(Tensor{4,3,Float64,81})
const 𝐈32 = one(SymmetricTensor{2,3,Float32})
const 𝐈64 = one(SymmetricTensor{2,3,Float64})
const 𝕀ᵗ32 = otimesl(𝐈32, 𝐈32)
const 𝕀ᵗ64 = otimesl(𝐈64, 𝐈64)

"""
    ℤ(T)

Returns the fourth order zero tensor with precision `T` that is stored as a
`SymmetricTensor` to save memory.
"""
@inline ℤ(::Type{Float32}) = ℤ32
@inline ℤ(::Type{Float64}) = ℤ64

"""
    ℤ_full(T)

Returns the fourth order zero tensor with precision `T` that is stored as a
`Tensor`.
"""
@inline ℤ_full(::Type{Float32}) = ℤ32_full
@inline ℤ_full(::Type{Float64}) = ℤ64_full

"""
    𝕀ᵗ(T)

Returns the minor transposed fourth order identity tensor with precision `T`
that is stored as a `SymmetricTensor`. Defined such that Iᵢⱼₖₗ = δᵢₗδⱼₖ
"""
@inline 𝕀ᵗ(::Type{Float32}) = 𝕀ᵗ32
@inline 𝕀ᵗ(::Type{Float64}) = 𝕀ᵗ64
