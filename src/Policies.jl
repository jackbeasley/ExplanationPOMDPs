module Policies
using POMDPs, POMDPPolicies

struct BeliefThresholdPolicy{S,A,O} <: Policy 
	problem::POMDP{S,A,O}
    threshold::Float64
    fallback_action::A
end
BeliefThresholdPolicy(p::P, th, fallback) where {P <: Union{POMDP,MDP}} = BeliefThresholdPolicy(
    p, 
    th, 
    fallback, 
)
export BeliefThresholdPolicy

function POMDPs.action(p::BeliefThresholdPolicy, b)
    bvec = beliefvec(p.problem, length(states(p.problem)), b)
    max_state_ind = argmax(bvec)

    if bvec[max_state_ind] > p.threshold
	    s = states(p.problem)[max_state_ind]
	    action_ind = argmax([reward(p.problem, s, a) for a in actions(p.problem)])
	    return actions(p.problem)[action_ind]
    end
    return p.fallback_action
end

end
