using Test
using ExplanationPOMDPs.SingleObservationExplanation
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators

pomdp = SingleObservationPOMDP(4, 2, 0.0, 1.0, -1.0, 1.0)
@testset "Single Observation States" begin
    states = POMDPs.states(pomdp)
    # Ensure all stateindices map correctly
    @test states == [states[stateindex(pomdp, s)] for s in states]

    terminality = [isterminal(pomdp, s) for s in states]
    @test [false, true, false, false, false, false, false] == terminality

    @test uniform_state_probs(pomdp) == [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
end

@testset "Initial Beliefs" begin
    init = initial_belief(pomdp)
    # @test [pdf(init, s) for s in states(pomdp)] == [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]

    up = DiscreteUpdater(pomdp)
    belief = initialize_belief(up, init)
end

@testset "Single Observation Simulation" begin

    p = FunctionPolicy(s -> 2)

    up = DiscreteUpdater(pomdp)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end

