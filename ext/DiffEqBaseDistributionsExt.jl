module DiffEqBaseDistributionsExt

using Distributions, DiffEqBase

DiffEqBase.handle_distribution_u0(_u0::Distributions.Sampleable) = rand(_u0)
DiffEqBase.isdistribution(_u0::Distributions.Sampleable) = true

function DiffEqBase.handle_distribution_u0(_u0::AbstractArray)
    map(x->x isa Distributions.Sampleable ? rand(x) : x, _u0)
end

end
