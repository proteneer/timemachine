from hypothesis import given, seed
from hypothesis.strategies import composite, floats, lists, sampled_from

from timemachine.fe.protocol_refinement import greedy_bisection_step

lambdas = floats(0.0, 1.0, allow_subnormal=False)

lambda_schedules = lists(lambdas, min_size=2, unique=True).map(sorted)


@composite
def greedy_bisection_step_args_instances(draw):
    protocol = draw(lambda_schedules)
    worst_pair_lam1 = draw(sampled_from(protocol[:-1]))

    def local_cost(lam1, _):
        return 1.0 if lam1 == worst_pair_lam1 else 0.0

    def make_intermediate(lam1, lam2):
        assert lam1 < lam2
        return draw(floats(lam1, lam2, allow_subnormal=False))

    return protocol, local_cost, make_intermediate


@given(greedy_bisection_step_args_instances())
@seed(2023)
def test_greedy_bisection_step(args):
    protocol, local_cost, make_intermediate = args
    refined_protocol, _ = greedy_bisection_step(protocol, local_cost, make_intermediate)
    assert len(refined_protocol) == len(protocol) + 1
    assert set(refined_protocol).issuperset(set(protocol))
    assert refined_protocol == sorted(refined_protocol)
