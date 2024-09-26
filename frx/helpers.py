import dataclasses

import beartype

#############
# Constants #
#############


IMAGENET_CHANNEL_MEAN = (0.4632, 0.4800, 0.3762)
IMAGENET_CHANNEL_STD = (0.2375, 0.2291, 0.2474)


#############
# Functions #
#############


@beartype.beartype
def to_aim_value(value: object):
    """
    Recursively converts objects into [Aim](https://github.com/aimhubio/aim)-compatible values.

    As a fallback, tries to call `to_aim_value()` on an object.
    """
    if value is None:
        return value

    if isinstance(value, (str, int, float)):
        return value

    if isinstance(value, list):
        return [to_aim_value(elem) for elem in value]

    if isinstance(value, dict):
        return {to_aim_value(k): to_aim_value(v) for k, v in value.items()}

    if dataclasses.is_dataclass(value):
        return to_aim_value(dataclasses.asdict(value))

    try:
        return value.tolist()
    except AttributeError:
        pass

    try:
        return value.to_aim_value()
    except AttributeError:
        pass

    raise ValueError(f"Could not convert value '{value}' to Aim-compatible value.")


###########
# Classes #
###########


class DummyAimRun:
    def __init__(self, *args, **kwargs):
        self.hash = "dummy-run-hash"

    def track(self, metrics: dict[str, object], step: int):
        pass
