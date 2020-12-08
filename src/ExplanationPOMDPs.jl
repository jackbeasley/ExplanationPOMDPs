module ExplanationPOMDPs

using POMDPs, Printf, POMDPSimulators

include("OneShot.jl")


function test_simulation(m::POMDP, policy::Policy, updater::Updater)
    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, m, policy, updater, initialstate(m))
    for step in eachstep(history)
        @printf("belief : %s\n", pdf(step.b, ))
        @printf("action: %s, observation: %s, state: %s\n", step.a, step.o, step.s)
    end 
end
export test_simulation


end # module
