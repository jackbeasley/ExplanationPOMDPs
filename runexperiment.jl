using ExplanationPOMDPs.Urn
using POMDPs, POMDPModelTools, BeliefUpdaters, POMDPSimulators, POMDPPolicies, DataFrames
##
pomdp = UrnPOMDP(4, 2)

##
s = initialstate(pomdp)
##
p = FunctionPolicy(s -> actions(pomdp)[3])
up = DiscreteUpdater(pomdp)
hr = HistoryRecorder(max_steps=10)
##
history = DataFrame(simulate(hr, pomdp, p, up, initialstate(pomdp)))