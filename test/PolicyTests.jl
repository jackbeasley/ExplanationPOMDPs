using Test
using ExplanationPOMDPs.SingleObservationExplanation
using ExplanationPOMDPs.Policies
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators

pomdp = SingleObservationPOMDP(4, 10, 0.0, 1.0, -1.0, 1.0)

@testset "Simulation with threshold policy" begin

    p = BeliefThresholdPolicy(pomdp, 0.4, -1, collect(0:length(states(pomdp))))
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
    finalstep = eachstep(history)[end]
    @test finalstep[:s].hypothesis_num == finalstep[:a]

    p = BeliefThresholdPolicy(pomdp, 1.0, -1, collect(0:length(states(pomdp))))
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
    finalstep = eachstep(history)[2]
    @test -1 == finalstep[:a]
end

