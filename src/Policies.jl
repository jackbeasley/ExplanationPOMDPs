module Policies
using POMDPs, POMDPPolicies

struct BeliefThresholdPolicy{P <: Union{POMDP,MDP}} <: Policy 
    problem::P
    threshold::Float64
    fallback_action::Int
    policy_vec::Vector{Int}
end
BeliefThresholdPolicy(p, th, fallback) = BeliefThresholdPolicy(
    p, 
    th, 
    fallback, 
    collect(0:length(states(p)))
)
export BeliefThresholdPolicy

function POMDPs.action(p::BeliefThresholdPolicy, b)
    bvec = beliefvec(p.problem, length(states(p.problem)), b)
    max_state_ind = argmax(bvec)

    if max_state_ind <= 2
        return -1
    end
    if bvec[max_state_ind] > p.threshold
        return states(p.problem)[max_state_ind]
    end
    return p.fallback_action
end

end