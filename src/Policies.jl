module Policies
using POMDPs, POMDPPolicies

struct BeliefThresholdPolicy{S,A,O} <: Policy 
	problem::POMDP{S,A,O}
    threshold::Float64
    fallback_action::A
end
export BeliefThresholdPolicy

function POMDPs.action(p::BeliefThresholdPolicy, b)
    bvec = beliefvec(p.problem, length(states(p.problem)), b)
    max_state_ind = argmax(bvec)
    if bvec[max_state_ind] > p.threshold
	    s = states(p.problem)[max_state_ind]
        rewards = [reward(p.problem, s, a) for a in actions(p.problem)]
	    action_ind = argmax(rewards)
	    return actions(p.problem)[action_ind]
    end
    return p.fallback_action
end

end
