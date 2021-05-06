module Urn

const TERMINAL_STATE = -1

struct UrnState{T <: Integer,H,S}
	state::T
end

all_urn_states(hyps, steps) = UrnState{Int8,hyps,steps}.(-1:(steps * hyps - 1))

# Static properties
n_states(::UrnState{<:Integer,H,S}) where {H,S} = H * S + 1
num_steps(::UrnState{<:Integer,H,S}) where {H,S} = S
n_hyps(::UrnState{<:Integer,H,S}) where {H,S} = H

representation(s::UrnState) = s.state

index(s::UrnState) = s.state + 2
end_state(s::UrnState) = s.state == TERMINAL_STATE
hypothesis_num(s::UrnState{<:Integer,H,S}) where {H,S} = s.state % H
step_num(s::UrnState{T,H,S}) where {T <: Integer,H,S} = T(floor(s.state / H))

function next(s::UrnState{T,H,S})::T where {T <: Integer,H,S}
	if num_steps(s) == step_num(s) + 1 || end_state(s)
		return -1
	end
	return s.state + H
end

next_state(s::UrnState{T,H,S}) where {T <: Integer,H,S} = UrnState{T,H,S}(next(s))

next_dist(s::UrnState{T,H,S}) where {T <: Integer,H,S} = Iterators.map(sn -> sn == next(s) ? 1.0 : 0.0, -1:S*H)

export UrnState, all_urn_states, n_states, num_steps, n_hyps, representation
export index, end_state, hypothesis_num, step_num, next, next_state, next_dist

const WITHHOLD_STATE = -1

struct UrnAction{T <: Integer,H}
	action::T
end

all_urn_actions(hyps) = UrnAction{Int8,hyps}.(-1:hyps)
n_actions(::UrnAction{<:Integer,H}) where {H} = H + 2
n_hyps(::UrnAction{<:Integer,H}) where {H} = H
representation(a::UrnAction) = a.action
index(a::UrnAction) = a.action + 2
hypothesis_num(a::UrnAction) = a.action

export UrnAction, all_urn_actions, n_actions, n_hyps, representation, index, hypothesis_num

struct UrnObservation{T <: Integer,C}
	observation::T
end

all_urn_observations(count) = UrnObservation{Int8,count}.(0:count)
n_observations(::UrnObservation{<:Integer,H}) where {H} = H + 1
representation(o::UrnObservation) = o.observation
index(o::UrnObservation) = o.observation + 1

export UrnObservation, all_urn_observations, n_observations, representation, index

using POMDPs, POMDPModelTools, Distributions

struct UrnPOMDP{T, H, S, O} <: POMDP{UrnState{T, H, S},UrnAction{T, H},UrnObservation{T, O}}
	states::Vector{UrnState{T, H, S}}
	actions::Vector{UrnAction{T, H}}
	observations::Vector{UrnObservation{T, O}}
    r_no_choice::Float64
    r_correct::Float64
    r_incorrect::Float64
    discount::Float64
end

function UrnPOMDP(balls_in_vase::Integer, balls_per_observation::Integer, observations::Integer=1)
	H = Int8(balls_in_vase)
	S = Int8(observations+1)
	O = Int8(balls_per_observation)
	state_vec = all_urn_states(H, S)
	action_vec = all_urn_actions(H)
	observation_vec = all_urn_observations(O)
	return UrnPOMDP{Int8, H, S, O}(
		state_vec,
		action_vec,
		observation_vec,
		0.0, 1.0, -1.0, 1.0
	)
end
export UrnPOMDP

POMDPs.discount(m::UrnPOMDP) = m.discount
POMDPs.isterminal(::UrnPOMDP{T, H, S, O}, s::UrnState{T, H, S}) where {T<:Integer, H,S,O} = end_state(s)
# 
POMDPs.states(pomdp::UrnPOMDP{T, H, S, O}) where {T<:Integer, H,S,O} = pomdp.states
POMDPs.stateindex(::UrnPOMDP{T, H, S, O}, s::UrnState{T, H, S}) where {T<:Integer, H,S,O} = index(s)
POMDPs.initialstate(pomdp::UrnPOMDP{T, H, S, O}) where {T<:Integer, H,S,O} = SparseCat(
	states(pomdp),
	[!isterminal(pomdp, s) && step_num(s) == 0 ? 1.0 / H : 0.0 for s in states(pomdp)]
)

function initialstate_prior(pomdp::UrnPOMDP{T, H, S, O}, prior::AbstractVector{F}) where {T<:Integer, H,S,O, F <: Real}
	return SparseCat(
	    states(pomdp),
	    [!isterminal(pomdp, s) && step_num(s) == 0 ? prior[hypothesis_num(s)+1] : 0.0 for s in states(pomdp)]
        )
end
export initialstate_prior

POMDPs.observations(pomdp::UrnPOMDP{T, H, S, O}) where {T<:Integer, H,S,O} = pomdp.observations
POMDPs.obsindex(::UrnPOMDP{T, H, S, O}, o::UrnObservation{T, O}) where {T<:Integer, H,S,O} = index(o)

function POMDPs.observation(pomdp::UrnPOMDP{T, H, S, O}, s::UrnState{T, H, S}) where {T<:Integer, H,S,O}
	if isterminal(pomdp, s)
		return DiscreteUniform(0, O)
	end
	return Binomial(
		O,
		hypothesis_num(s) / H,
	)
end

POMDPs.actions(pomdp::UrnPOMDP{T, H, S, O}) where {T<:Integer, H,S,O} = pomdp.actions
POMDPs.actionindex(::UrnPOMDP{T, H, S, O}, a::UrnAction{T, H}) where {T<:Integer, H,S,O} = index(a)
function POMDPs.reward(pomdp::UrnPOMDP{T, H, S, O},  s::UrnState{T, H, S}, a::UrnAction{T, H})::Float64 where {T<:Integer, H,S,O}
	if POMDPs.isterminal(pomdp, s)
		return 0
	end

	if a == -1
		return pomdp.r_no_choice
	elseif hypothesis_num(a) == hypothesis_num(a)
		return pomdp.r_correct
	else
		return pomdp.r_incorrect
	end
end

function POMDPs.transition(pomdp::UrnPOMDP{T, H, S, O},  s::UrnState{T, H, S}, ::UrnAction{T, H}) where {T<:Integer, H,S,O}
	return SparseCat(
		states(pomdp),
		collect(next_dist(s))
	)
end

end

