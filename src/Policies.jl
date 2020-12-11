module Policies
using POMDPs, POMDPPolicies

struct BeliefThresholdPolicy{P <: Union{POMDP,MDP}} <: Policy 
    problem::P
    threshold::Float64
    under_action::Int
    policy_vec::Vector{Int}
end
export BeliefThresholdPolicy

function POMDPs.action(p::BeliefThresholdPolicy, b)
    bvec = beliefvec(p.problem, length(states(p.problem)), b)
    max_state_ind = argmax(bvec)

    println(b)
    if max_state_ind <= 2
        return -1
    end
    if bvec[max_state_ind] > p.threshold
        println(max_state_ind)
        return states(p.problem)[max_state_ind]
    end
    return p.under_action
end

end