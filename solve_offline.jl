using ExplanationPOMDPs
using DataFrames
using POMDPs, POMDPModelTools, Printf, BeliefUpdaters, POMDPPolicies, POMDPSimulators, Distributions
using QMDP

m = ExplainPOMDP(10)

solver = QMDPSolver()
plan = solve(solver, m)

function run_experiment()
    sim = HistoryRecorder(max_steps=100)
    hist = simulate(sim, m, plan)
    total_reward = 0.0
    belief_in_choice = Vector{Float64}()
    correct_picks = 0
    for step in eachstep(hist)
        # @printf("beliefs: %s\n", [pdf(step.b, i) for i in 1:length(states(m))])
        # @printf("action: %s, reward: %s, observation: %s, state: %s\n", step.a, step.r, step.o, step.s)
        if step.a != 0
            append!(belief_in_choice, pdf(step.b, step.a))
        end
        total_reward += step.r
        if step.r == 5
            correct_picks += 1
        end
    end

    return total_reward, mean(belief_in_choice), length(belief_in_choice), correct_picks
end

rewards = Vector{Float64}()
mean_belief_threshold = Vector{Float64}()
number_selections = Vector{Float64}()
number_correct_selections = Vector{Float64}()
for i in 1:10
    (r, mean_belief, num_sel, correct) = run_experiment()
    append!(rewards, r)
    append!(mean_belief_threshold, mean_belief)
    append!(number_selections, num_sel)
    append!(number_correct_selections, correct)
end

df = DataFrame(Rewards=rewards, MeanBeliefThreshold=mean_belief_threshold, NumberSelections=number_selections, NumberCorrect=number_correct_selections)


