from datetime import datetime
from typing import Any, List

import gin

from gflownet.api.env_base import EnvBase, TAction, TActionSpace, TState


@gin.configurable()
def get_time_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


@gin.configurable()
def get_str(format: str, values: List[Any]) -> str:
    return format.format(*values)


@gin.configurable()
def reverse(env: EnvBase[TState, TActionSpace, TAction]) -> EnvBase[TState, TActionSpace, TAction]:
    return env.reversed()
