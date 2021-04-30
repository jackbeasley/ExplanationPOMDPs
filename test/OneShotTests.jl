using Test
using ExplanationPOMDPs.SingleObservationExplanation
using ExplanationPOMDPs.Beliefs
using ExplanationPOMDPs.Policies
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators

@testset "States" begin
    states = collect(explanation_states(2, 2))
    @test states == [
        OneShotState(true, -1, -1),
        OneShotState(false, 0, 1),
        OneShotState(false, 1, 1),
        OneShotState(false, 2, 1),
        OneShotState(false, 0, 2),
        OneShotState(false, 1, 2),
        OneShotState(false, 2, 2),
    ]
end

pomdp = SingleObservationPOMDP(4, 2, 0.0, 1.0, -1.0, 1.0)
@testset "Single Observation States" begin

    @test POMDPs.states(pomdp) == [
        OneShotState(true, -1, -1),
        OneShotState(false, 0, 1),
        OneShotState(false, 1, 1),
        OneShotState(false, 2, 1),
        OneShotState(false, 3, 1),
        OneShotState(false, 4, 1),
        OneShotState(false, 0, 2),
        OneShotState(false, 1, 2),
        OneShotState(false, 2, 2),
        OneShotState(false, 3, 2),
        OneShotState(false, 4, 2),
    ]
    # Ensure all stateindices map correctly
    states_vec = POMDPs.states(pomdp)
    indices = [stateindex(pomdp, s) for s in states_vec]
    @test collect(1:length(states_vec)) == indices
    @test states_vec == states_vec[indices]

    terminality = [isterminal(pomdp, s) for s in states_vec]
    @test [true, false, false, false, false, false, false, false, false, false, false] == terminality

    @test [pdf(initialstate(pomdp), s) for s in states_vec] == [0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
end

# @testset "Initial Beliefs" begin
#    init = initial_belief(pomdp)
#    # @test [pdf(init, s) for s in states(pomdp)] == [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]
#
#    up = DiscreteUpdater(pomdp)
#    belief = initialize_belief(up, init)
# end

@testset "Single Observation Simulation" begin
    p = FunctionPolicy(s -> 2)

    up = DiscreteUpdater(pomdp)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end

@testset "skew beliefs" begin
    p = BeliefThresholdPolicy(pomdp, 0.4, -1, collect(0:length(states(pomdp))))
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate_prior(pomdp, [0.0, 1.0, 0.0, 0.0, 0.0]))

    # Initial state and state after observation
    @test length(history) == 2
    bvec = beliefvec(pomdp, length(states(pomdp)), history[end].b)
    state = states(pomdp)[argmax(bvec)]
    @test state.hypothesis_num == 1

    p = BeliefThresholdPolicy(pomdp, 0.4, -1, collect(0:length(states(pomdp))))
    up = DiscreteUpdater(pomdp)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate_prior(pomdp, [0.0, 0.0, 0.0, 1.0, 0.0]))

    # Initial state and state after observation
    @test length(history) == 2
    bvec = beliefvec(pomdp, length(states(pomdp)), history[end].b)
    state = states(pomdp)[argmax(bvec)]
    @test state.hypothesis_num == 3
end



@testset "Single Observation Popper Simulation" begin
    p = FunctionPolicy(s -> 2)

    up = IBEUpdater(pomdp, PopperBonus(), 1.0)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end

@testset "Single Observation Schupbach Simulation" begin
    p = FunctionPolicy(s -> 2)

    up = IBEUpdater(pomdp, GoodBonus(), 1.0)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end

@testset "Single Observation Good Simulation" begin
    p = FunctionPolicy(s -> 2)

    up = IBEUpdater(pomdp, GoodBonus(), 1.0)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end



