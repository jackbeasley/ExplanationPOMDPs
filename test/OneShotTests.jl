using Test
using ExplanationPOMDPs.Urn
using ExplanationPOMDPs.Beliefs
using ExplanationPOMDPs.Policies
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators

pomdp = UrnPOMDP(4, 2)

@testset "skew beliefs" begin
    p = BeliefThresholdPolicy(pomdp, 0.4, actions(pomdp)[1])
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate_prior(pomdp, [0.0, 1.0, 0.0, 0.0, 0.0]))

    # Initial state and state after observation
    @test length(history) == 2
    bvec = beliefvec(pomdp, length(states(pomdp)), history[end].b)
    state = states(pomdp)[argmax(bvec)]
    @test hypothesis_num(state) == 1

    p = BeliefThresholdPolicy(pomdp, 0.4, actions(pomdp)[1])
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate_prior(pomdp, [0.0, 0.0, 0.0, 1.0, 0.0]))

    # Initial state and state after observation
    @test length(history) == 2
    bvec = beliefvec(pomdp, length(states(pomdp)), history[end].b)
    state = states(pomdp)[argmax(bvec)]
    @test hypothesis_num(state) == 3
end


@testset "Single Observation Popper Simulation" begin
    p = BeliefThresholdPolicy(pomdp, 0.4, actions(pomdp)[1])

    up = IBEUpdater(pomdp, PopperBonus(), 1.0)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end

@testset "Single Observation Schupbach Simulation" begin
    p = BeliefThresholdPolicy(pomdp, 0.4, actions(pomdp)[1])

    up = IBEUpdater(pomdp, GoodBonus(), 1.0)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end

@testset "Single Observation Good Simulation" begin
    p = BeliefThresholdPolicy(pomdp, 0.4, actions(pomdp)[1])

    up = IBEUpdater(pomdp, GoodBonus(), 1.0)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end



