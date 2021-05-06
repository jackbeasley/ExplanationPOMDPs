using Test

@testset "State" begin
	using ExplanationPOMDPs.Urn
	expected_n_hyps = 5
	expected_n_steps = 3
	states = all_urn_states(expected_n_hyps, expected_n_steps)
	# Check static properties
	@test length(states) == n_states(states[1])
	@test expected_n_steps == num_steps(states[1])
	@test expected_n_hyps == n_hyps(states[1])

	# Verify integer values
	@test representation.(states) == collect(-1:expected_n_hyps * expected_n_steps)

	# Verify properties
	@test end_state.(states) == [i == 1 ? true : false for i in 1:length(states)]

	step_one = states[2]
	@test representation(step_one) == 0
	@test step_num(step_one) == 0
	@test hypothesis_num(step_one) == 0
	@test !end_state(step_one)

	step_two = next_state(step_one)
	@test representation(step_two) == 5
	@test step_num(step_two) == 1
	@test hypothesis_num(step_two) == 0
	@test !end_state(step_two)

	step_three = next_state(step_two)
	@test representation(step_three) == 10
	@test !end_state(step_three)
	@test step_num(step_three) == 2
	@test hypothesis_num(step_three) == 0

	last_state = next_state(step_three)
	@test representation(last_state) == -1
	@test end_state(last_state)
end

@testset "Single Observation Simulation" begin
	using ExplanationPOMDPs.Urn
	using POMDPs, POMDPModelTools, BeliefUpdaters, POMDPSimulators, POMDPPolicies
	pomdp = UrnPOMDP(4, 2)

    p = FunctionPolicy(s -> actions(pomdp)[3])

    up = DiscreteUpdater(pomdp)

    hr = HistoryRecorder(max_steps=10)
    history = simulate(hr, pomdp, p, up, initialstate(pomdp))

    # Initial state and state after observation
    @test length(history) == 2
end
