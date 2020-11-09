module ExplanationPOMDPs

using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions


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

function POMDPs.observation(pomdp::ExplainPOMDP,  a::Int, s::Int)
    if a == -1
        # Return true with probability of green ball
        return BoolDistribution(float(pomdp.green_balls) / pomdp.n_balls)
    end
    return BoolDistribution(0.5)
end

function POMDPs.reward(pomdp::ExplainPOMDP, s::Int, a::Int)::Float64
    if a == -1
        return -1.0
    end
    if s == pomdp.green_balls
        return 10
    end
    return -10
end

POMDPs.discount(::ExplainPOMDP) = 1.0

POMDPs.initialstate(pomdp::ExplainPOMDP) = DiscreteUniform(0, pomdp.n_balls)
POMDPs.initialobs(p::ExplainPOMDP, s::Int) = observation(p, -1, s)

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
