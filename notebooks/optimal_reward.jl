using ExplanationPOMDPs.SingleObservationExplanation
using ExplanationPOMDPs.Policies
using POMDPs, BeliefUpdaters, POMDPPolicies, POMDPSimulators
using DataFrames
##

pomdp = SingleObservationPOMDP(
    2,     # Four balls in each vase
    8,     # Two balls drawn for each 
    0.0,   # No reward or penalty for omitting an answer
    1.0,   # 1 point reward for a correct guess
    -1.0,  # 1 point penalty for an incorrect guess
    1.0    # No discount
)

function run_experiment(problem::POMDP, p::Policy, up::Updater, filter_init=true)::DataFrame
    hr = HistoryRecorder(max_steps=3)
    history = simulate(hr, problem, p, up, initialstate(problem))
    if filter_init
        return DataFrame(history[2:2])
    end
    return DataFrame(history)
end

function run_experiments(problem::POMDP, p::Policy, up::Updater, n::Int, filter_init=true)::DataFrame
    history = run_experiment(problem, p, up, filter_init)
    history[:run] = 1
    if n > 1
        for _ in 2:n
            run = run_experiment(problem, p, up, filter_init)
            run[:run] = n
            append!(history, run)
        end
    end
    return history
end
##

res = run_experiments(
    pomdp, 
    BeliefThresholdPolicy(pomdp, 0.1, -1, collect(0:length(states(pomdp)))), 
    DiscreteUpdater(pomdp),
    10
)


