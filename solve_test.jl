using ExplanationPOMDPs
using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions
using QMDP

m = ExplainPOMDP(15, 20)

solver = QMDPSolver()

policy = solve(solver, m)