using DiffEqBase, Test
using Distributions

@test DiffEqBase.promote_tspan((0.0, 1.0)) == (0.0, 1.0)
@test DiffEqBase.promote_tspan((0, 1.0)) == (0.0, 1.0)
@test DiffEqBase.promote_tspan(1.0) == (0.0, 1.0)
@test DiffEqBase.promote_tspan(nothing) == (nothing, nothing)
@test DiffEqBase.promote_tspan(Real[0, 1.0]) == (0.0, 1.0)

# https://github.com/SciML/OrdinaryDiffEq.jl/issues/1776
# promote_tspan(u0, p, tspan, prob, kwargs)
@test DiffEqBase.promote_tspan((0, 1)) == (0, 1)
@test DiffEqBase.promote_tspan(nothing, nothing, (0, 1), nothing, (dt = 1,)) == (0, 1)
@test DiffEqBase.promote_tspan(nothing, nothing, (0, 1), nothing, (dt = 1 / 2,)) ==
      (0.0, 1.0)

prob = ODEProblem((u, p, t) -> u, (p, t0) -> p[1], (p) -> (0.0, p[2]), (2.0, 1.0))
prob2 = DiffEqBase.get_concrete_problem(prob, true)

@test prob2.u0 == 2.0
@test prob2.tspan == (0.0, 1.0)

prob = ODEProblem((u, p, t) -> u, (p, t) -> Normal(p, 1), (0.0, 1.0), 1.0)
prob2 = DiffEqBase.get_concrete_problem(prob, true)
# @edit DiffEqBase.get_concrete_problem(prob, true)
# DiffEqBase.get_concrete_problem(prob, true)
# Debugger.@enter DiffEqBase.get_concrete_problem(prob, true)
# these fail now
@test typeof(prob2.u0) == Float64

kwargs(; kw...) = kw
prob = ODEProblem((u, p, t) -> u, 1.0, nothing)
prob2 = DiffEqBase.get_concrete_problem(prob, true, tspan = (1.2, 3.4))
@test prob2.tspan === (1.2, 3.4)

prob = ODEProblem((u, p, t) -> u, nothing, nothing)
prob2 = DiffEqBase.get_concrete_problem(prob, true, u0 = 1.01, tspan = (1.2, 3.4))
@test prob2.u0 === 1.01

prob = ODEProblem((u, p, t) -> u, 1.0, (0, 1))
prob2 = DiffEqBase.get_concrete_problem(prob, true)
@test prob2.tspan == (0.0, 1.0)

prob = DDEProblem((u, h, p, t) -> -h(p, t - p[1]), (p, t0) -> p[2], (p, t) -> 0,
                  (p) -> (0.0, p[3]), (1.0, 2.0, 3.0); constant_lags = (p) -> [p[1]])
prob2 = DiffEqBase.get_concrete_problem(prob, true)

@test prob2.u0 == 2.0
@test prob2.tspan == (0.0, 3.0)
@test prob2.constant_lags == [1.0]

# 
function f(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = -p[2] * u[2]
end

dist = Uniform(0, 1)
tspan = (0, 1)
prob = ODEProblem(f, [dist, 0.5], tspan, [dist, 3])
prob2 = DiffEqBase.get_concrete_problem(prob, true)
sol = solve(prob)

# honestly this is pretty confusing. 
# if we made it so that you provide distributions in defaults for sys and the ODEProblem(sys) construction concretizes, that could make a bit more sense
@test prob.p[1] isa Distributions.Sampleable
@test prob.u0[1] isa Distributions.Sampleable
@test sol.prob.p[1] isa Number
@test sol.prob.u0[1] isa Number

ds = [dist, Dirac(0.5)]
pd = product_distribution(ds)
rand(pd)

prob2 = ODEProblem(f, pd, tspan, pd)
sol2 = solve(prob2)
@test prob2.p isa Distributions.Sampleable
@test prob2.u0 isa Distributions.Sampleable
@test sol2.prob.p[1] isa Number
@test sol2.prob.u0[1] isa Number
