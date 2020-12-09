using Test, ExplanationPOMDPs.SingleObservationExplanation, POMDPs

pomdp = SingleObservationPOMDP(4, 2, 0.0, 1.0, -1.0, 1.0)
@testset "Single Observation States" begin
    states = POMDPs.states(pomdp)
    ball_counts = [ball_count(pomdp, s) for s in states]
    @test ball_counts == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

    # Ensure all stateindices map correctly
    @test states == [states[stateindex(pomdp, s)] for s in states]

    terminality = [isterminal(pomdp, s) for s in states]
    @test [false, false, false, false, false, true, true, true, true, true] == terminality

    dist = initialstate(pomdp)
    prbs = [pdf(dist, s) for s in states]
    @test prbs == [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
end