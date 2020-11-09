module ExplanationPOMDPs

using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions

mutable struct ExplainPOMDP <: POMDP{Int,Int64,Bool}
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
    p_listen_correctly::Float64
    discount_factor::Float64
end
ExplainPOMDP() = ExplainPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)
export ExplainPOMDP

POMDPs.states(::ExplainPOMDP) = (0, 1)
POMDPs.observations(::ExplainPOMDP) = (true, false)

POMDPs.stateindex(::ExplainPOMDP, s::Int) = Int64(s) + 1
POMDPs.actionindex(::ExplainPOMDP, a::Int) = a + 1
POMDPs.obsindex(::ExplainPOMDP, o::Int) = Int64(o) + 1

initial_belief(::ExplainPOMDP) = DiscreteBelief(2)
export initial_belief

const DRAW = 0
const TIGER_OPEN_LEFT = 1
const TIGER_OPEN_RIGHT = 2

const TIGER_LEFT = 0
const TIGER_RIGHT = 1

# Resets the problem after opening door; does nothing after listening
function POMDPs.transition(pomdp::ExplainPOMDP, s::Int64, a::Int64)
    p = 1.0
    if a == 1 || a == 2
        p = 0.5
    elseif s == 1
        p = 1.0
    else
        p = 0.0
    end
    return SparseCat([0, 1], [p, 1.0 - p])
end

function POMDPs.observation(pomdp::ExplainPOMDP, a::Int64, sp::Int64)

    pc = pomdp.p_listen_correctly
    p = 1.0
    if a == 0
        sp == 1 ? (p = pc) : (p = 1.0 - pc)
    else
        p = 0.5
    end
    return BoolDistribution(p)
end

function POMDPs.observation(pomdp::ExplainPOMDP, s::Int64, a::Int64, sp::Int64)
    return observation(pomdp, a, sp)
end

function POMDPs.observation(pomdp::ExplainPOMDP, sp::Int64)
    return observation(pomdp, a, sp)
end


function POMDPs.reward(pomdp::ExplainPOMDP, s::Int64, a::Int64)
    r = 0.0
    a == 0 ? (r += pomdp.r_listen) : (nothing)
    if a == 1
        s == 0 ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    end
    if a == 2
        s == 1 ? (r += pomdp.r_escapetiger) : (r += pomdp.r_findtiger)
    end
    return r
end

POMDPs.reward(pomdp::ExplainPOMDP, s::Int64, a::Int64, sp::Int64) = reward(pomdp, s, a)


POMDPs.initialstate(pomdp::ExplainPOMDP) = SparseCat([0, 1], [0.5, 0.5])

POMDPs.actions(::ExplainPOMDP) = [0,1,2]

function upperbound(pomdp::ExplainPOMDP, s::Int64)
    return pomdp.r_escapetiger
end
export upperbound

POMDPs.discount(pomdp::ExplainPOMDP) = pomdp.discount_factor

POMDPs.initialobs(p::ExplainPOMDP, s::Int64) = observation(p, 0, s) # listen 


function test_simulation()
    m = ExplainPOMDP()
    policy = FunctionPolicy(b -> DRAW)
    updater = DiscreteUpdater(m);
    s0 = rand(initialstate(m))
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, m, policy, updater)
    for step in eachstep(history)
        @printf("belief : %s\n", pdf(step.b, 1))
        @printf("action: %s, observation: %s\n", step.a, step.o)
    end
end

export test_simulation

end # module
