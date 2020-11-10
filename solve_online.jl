using ExplanationPOMDPs
using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions
using BasicPOMCP

m = ExplainPOMDP(10)

solver = POMCPSolver()
planner = solve(solver, m)

sim = HistoryRecorder(max_steps=100)
hist = simulate(sim, m, planner)
total_reward = 0.0
belief_in_choice = Vector{Float64}()
for step in eachstep(hist)
    @printf("beliefs: %s\n", [pdf(step.b, i) for i in 1:length(states(m))])
    @printf("action: %s, reward: %s, observation: %s, state: %s\n", step.a, step.r, step.o, step.s)
    if step.a != 0
        append!(belief_in_choice, pdf(step.b, step.a))
    end
    total_reward += step.r
end
println(total_reward)
println(mean(belief_in_choice))