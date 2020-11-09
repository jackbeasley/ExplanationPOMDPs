module ExplanationPOMDPs

using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions

mutable struct ExplainPOMDP <: POMDP{Bool,Int64,Bool}
    r_listen::Float64
    r_findtiger::Float64
    r_escapetiger::Float64
    p_listen_correctly::Float64
    discount_factor::Float64
end
ExplainPOMDP() = ExplainPOMDP(-1.0, -100.0, 10.0, 0.85, 0.95)
export ExplainPOMDP

POMDPs.states(::ExplainPOMDP) = (true, false)
POMDPs.observations(::ExplainPOMDP) = (true, false)

POMDPs.stateindex(::ExplainPOMDP, s::Bool) = Int64(s) + 1
POMDPs.actionindex(::ExplainPOMDP, a::Int) = a + 1
POMDPs.obsindex(::ExplainPOMDP, o::Bool) = Int64(o) + 1

initial_belief(::ExplainPOMDP) = DiscreteBelief(2)
export initial_belief

const TIGER_LISTEN = 0
const TIGER_OPEN_LEFT = 1
const TIGER_OPEN_RIGHT = 2

const TIGER_LEFT = true
const TIGER_RIGHT = false


# Resets the problem after opening door; does nothing after listening
function POMDPs.transition(pomdp::ExplainPOMDP, s::Bool, a::Int64)
    p = 1.0
    if a == 1 || a == 2
        p = 0.5
    elseif s
        p = 1.0
    else
        p = 0.0
    end
    return BoolDistribution(p)
end

function POMDPs.observation(pomdp::ExplainPOMDP, a::Int64, sp::Bool)
    pc = pomdp.p_listen_correctly
    p = 1.0
    if a == 0
        sp ? (p = pc) : (p = 1.0 - pc)
    else
        p = 0.5
    end
    return BoolDistribution(p)
end

function POMDPs.observation(pomdp::ExplainPOMDP, s::Bool, a::Int64, sp::Bool)
    return observation(pomdp, a, sp)
end


function POMDPs.reward(pomdp::ExplainPOMDP, s::Bool, a::Int64)
    r = 0.0
    a == 0 ? (r += pomdp.r_listen) : (nothing)
    if a == 1
        s ? (r += pomdp.r_findtiger) : (r += pomdp.r_escapetiger)
    end
    if a == 2
        s ? (r += pomdp.r_escapetiger) : (r += pomdp.r_findtiger)
    end
    return r
end

POMDPs.reward(pomdp::ExplainPOMDP, s::Bool, a::Int64, sp::Bool) = reward(pomdp, s, a)


POMDPs.initialstate(pomdp::ExplainPOMDP) = BoolDistribution(0.5)

POMDPs.actions(::ExplainPOMDP) = [0,1,2]

function upperbound(pomdp::ExplainPOMDP, s::Bool)
    return pomdp.r_escapetiger
end
export upperbound

POMDPs.discount(pomdp::ExplainPOMDP) = pomdp.discount_factor

POMDPs.initialobs(p::ExplainPOMDP, s::Bool) = observation(p, 0, s) # listen 


function test_simulation()
    m = ExplainPOMDP(15, 20)
    policy = FunctionPolicy(b -> -1)
    updater = KMarkovUpdater(5);
    s0 = rand(initialstate(m))
    initial_observation = rand(initialobs(m, s0))
    initial_obs_vec = fill(initial_observation, 5)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, m, policy, updater, initial_obs_vec, s0)
    for step in eachstep(history)
        @printf("belief : %s\n", pdf(step.b, 15))
        @printf("action: %s, observation: %s\n", step.a, step.o)
    end
end

export test_simulation

end # module
