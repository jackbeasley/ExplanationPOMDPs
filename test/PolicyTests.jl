using Test
using ExplanationPOMDPs.SingleObservationExplanation
using ExplanationPOMDPs.Policies
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators

pomdp = UrnPOMDP(4, 10)

@testset "Simulation with threshold policy" begin

    p = BeliefThresholdPolicy(pomdp, 0.0, actions(pomdp)[1])
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
    finalstep = eachstep(history)[end]
    @test hypothesis_num(finalstep[:s]) == hypothesis_num(finalstep[:a])

    p = BeliefThresholdPolicy(pomdp, 1.0, actions(pomdp)[1])
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
    finalstep = eachstep(history)[2]
    @test representation(actions(pomdp)[1]) == representation(finalstep[:a])
end

