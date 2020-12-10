using ExplanationPOMDPs, ExplanationPOMDPs.SingleObservationExplanation
using POMDPModelTools, POMDPPolicies, POMDPSimulators, BeliefUpdaters

m = SingleObservationPOMDP(10, 5, 0.0, 1.0, -1.0, 1.0)

p = FunctionPolicy(s -> 5)

up = DiscreteUpdater(m)

test_simulation(m, p, up)

