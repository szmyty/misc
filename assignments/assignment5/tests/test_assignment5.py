from numpy.random import MT19937
from numpy.random.bit_generator import SeedSequence
from numpy.random.mtrand import RandomState


class TestAssignment5:
    def test_random_state_equality(self):
        seed: int = 42
        seed_alt: int = 55
        random_state1: RandomState = RandomState(MT19937(SeedSequence(seed)))
        random_state2: RandomState = RandomState(MT19937(SeedSequence(seed)))
        random_state3: RandomState = RandomState(MT19937(SeedSequence(seed_alt)))

        assert str(random_state1.get_state()) == str(random_state2.get_state())
        assert str(random_state1.get_state()) != str(random_state3.get_state())
