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
    max_ind = argmax(bvec)
    if bvec[max_ind] > p.threshold
        return actions(p.problem)[max_ind]
    end
    return p.under_action
end

end