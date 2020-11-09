module ExplanationPOMDPs

using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions

mutable struct ExplainPOMDP <: POMDP{Int64,Int64,Bool}
    r_draw::Float64
    r_guess_right::Float64
    r_guess_wrong::Float64
    n_balls::Int64
    discount_factor::Float64
end
ExplainPOMDP(n::Int64) = ExplainPOMDP(-1.0, 5.0, -5.0, n, 0.95)

export ExplainPOMDP

POMDPs.states(pomdp::ExplainPOMDP) = collect(1:pomdp.n_balls)
POMDPs.observations(::ExplainPOMDP) = (true, false)

POMDPs.stateindex(::ExplainPOMDP, s::Int) = Int64(s)
POMDPs.actionindex(::ExplainPOMDP, a::Int) = a + 1
POMDPs.obsindex(::ExplainPOMDP, o::Bool) = Int64(o) + 1

initial_belief(pomdp::ExplainPOMDP) = DiscreteBelief(pomdp.n_balls)
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
        # Reset with uniform distribution
        return initialstate(pomdp)
    end
    probs = zeros(Float64, length(states(pomdp)))
    probs[s] = 1.0
    return SparseCat(states(pomdp), probs)
end

function POMDPs.observation(pomdp::ExplainPOMDP, a::Int64, sp::Int64)
    return BoolDistribution(sp / pomdp.n_balls)
end

function POMDPs.observation(pomdp::ExplainPOMDP, s::Int64, a::Int64, sp::Int64)
    return observation(pomdp, a, sp)
end

function POMDPs.observation(pomdp::ExplainPOMDP, sp::Int64)
    return observation(pomdp, a, sp)
end


function POMDPs.reward(pomdp::ExplainPOMDP, s::Int64, a::Int64)
    if a == 0
        return pomdp.r_draw
    elseif a == s
        return pomdp.r_guess_right
    else
        return pomdp.r_guess_wrong
    end
end

POMDPs.reward(pomdp::ExplainPOMDP, s::Int64, a::Int64, sp::Int64) = reward(pomdp, s, a)

POMDPs.initialstate(pomdp::ExplainPOMDP) = SparseCat(states(pomdp), fill(1.0 / length(states(pomdp)), length(states(pomdp))))

POMDPs.actions(pomdp::ExplainPOMDP) = collect(0:pomdp.n_balls)

function upperbound(pomdp::ExplainPOMDP, s::Int64)
    return pomdp.r_escapetiger
end
export upperbound

POMDPs.discount(pomdp::ExplainPOMDP) = pomdp.discount_factor

POMDPs.initialobs(p::ExplainPOMDP, s::Int64) = observation(p, 0, s) # listen 


function test_simulation()
    m = ExplainPOMDP(-1.0, -100.0, 10.0, 0.85, 10, 0.95)
    policy = FunctionPolicy(b -> DRAW)
    updater = DiscreteUpdater(m);
    hr = HistoryRecorder(max_steps=10)
    s = initialstate(m)
    println(s)
    println(pdf(s, 2))
    history = simulate(hr, m, policy, updater, initialstate(m))
    for step in eachstep(history)
        @printf("belief : %s\n", pdf(step.b, 1))
        @printf("action: %s, observation: %s, state: %s\n", step.a, step.o, step.s)
    end
end

export test_simulation

end # module
