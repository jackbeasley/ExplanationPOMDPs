using ExplanationPOMDPs
using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions
using QMDP

m = ExplainPOMDP()

solver = QMDPSolver()

policy = solve(solver, m)

sim = RolloutSimulator(max_steps=80)
reward = simulate(sim, m, policy)