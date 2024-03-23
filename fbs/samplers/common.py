from typing import NamedTuple
from fbs.typings import FloatScalar, BoolScalar


class MCMCState(NamedTuple):
    acceptance_prob: FloatScalar
    is_accepted: BoolScalar
    prop_log_ell: FloatScalar
    log_ell: FloatScalar
