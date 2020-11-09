module ExplanationPOMDPs

using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators

struct ExplainPOMDP <: POMDP{Int,Int,Bool}
    green_balls::Int 
    n_balls::Int 
end

export ExplainPOMDP

POMDPs.states(pomdp::ExplainPOMDP) = collect(0:pomdp.n_balls)
POMDPs.stateindex(::ExplainPOMDP, n::Int) = n + 1

POMDPs.actions(pomdp::ExplainPOMDP) = collect(-1:pomdp.n_balls)
POMDPs.actionindex(::ExplainPOMDP, a::Int) = a + 1

POMDPs.transition(::ExplainPOMDP, s::Int, a::Int) = Deterministic(s)

POMDPs.observations(::ExplainPOMDP) = [true, false]
POMDPs.obsindex(::ExplainPOMDP, o::Bool) = o + 1

function POMDPs.observation(pomdp::ExplainPOMDP,  a::Int, s::Int, )
    if a == -1
        # Return true with probability of green ball
        return BoolDistribution(float(pomdp.green_balls) / pomdp.n_balls)
    end
    return BoolDistribution(0.5)
end

POMDPs.observation(pomdp::ExplainPOMDP,  s, a, sp) = observation(pomdp, a, sp)

function POMDPs.reward(pomdp::ExplainPOMDP, s::Int, a::Int)::Float64
    if a == -1
        return 0.0
    end
    d = abs(s - d)

    return (pomdp.n_balls - d)^2
end

POMDPs.discount(::ExplainPOMDP) = 1.0

POMDPs.initialstate(pomdp::ExplainPOMDP) = DiscreteBelief(pomdp, fill(1.0 / length(states(pomdp)), length(states(pomdp))))

function test_simulation()
    m = ExplainPOMDP(15, 20)
    observe = FunctionPolicy(b -> -1)
    update_func = DiscreteUpdater(m);
    for (b, a, o, r) in stepthrough(m, observe, update_func, "b,a,o,r", max_steps=20)
        @printf("action: %s, observation: %s\n", a, o)
    end
end

export test_simulation

end # module
