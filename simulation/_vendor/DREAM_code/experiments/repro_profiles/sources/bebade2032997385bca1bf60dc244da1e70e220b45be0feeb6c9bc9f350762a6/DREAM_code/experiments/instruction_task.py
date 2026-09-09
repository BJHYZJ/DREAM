"""Language-only pickup/place contract for the new instruction-driven adapter.

This intentionally has no simulator imports or scene/actor coordinates. The
parser supports a documented English command grammar, not unrestricted language
understanding. Environment recipes must not be passed into the search policy.
"""
from dataclasses import asdict, dataclass
import math
import re


@dataclass(frozen=True)
class PickPlaceInstruction:
    text: str
    pickup_query: str
    placement_query: str
    placement_relation: str
    support_query: str | None

    def policy_payload(self):
        return asdict(self)


def _noun(text):
    text=re.sub(r"\s+", " ", text.strip())
    return re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)


def parse_instruction(text):
    """Keep target attributes and support context; reject unsupported commands.

    Examples: 'Please pick up the tomato and put it in the bowl on the white
    table'; 'pickup the mug and place it onto the plate'. No catalogue lookup
    or hidden coordinate is used to interpret the language.
    """
    normalized=re.sub(r"\s+", " ", text.strip().rstrip(".!"))
    match=re.fullmatch(
        r"(?:please\s+)?(?:pick\s*up|take|grab)\s+(.+?)\s+and\s+"
        r"(?:put|place)\s+(?:it|them)\s+(into|onto|in|on)\s+(.+)",
        normalized, flags=re.I)
    if not match:
        raise ValueError("Expected: pick up <object> and put it in/on <destination> [on <support>]")
    pickup,relation,destination=match.groups()
    support=None
    context=re.fullmatch(r"(.+?)\s+(?:(?:which|that)\s+(?:is\s+)?)?on\s+(.+)",destination,flags=re.I)
    if context:
        destination,support=context.groups()
        support=_noun(support)
    pickup,destination=_noun(pickup),_noun(destination)
    for phrase in (pickup,destination,support):
        if phrase is not None and (not phrase or re.search(r"\b(?:and|then)\b",phrase,re.I)):
            raise ValueError("Only one pickup and one placement target are supported")
    return PickPlaceInstruction(text,pickup,destination,
                                "in" if relation.lower() in ("in","into") else "on",support)


@dataclass(frozen=True)
class VisualDiscovery:
    """A current policy detection, not an evaluator target pose."""
    observation_id: int
    sim_step: int
    query: str
    point_world: tuple[float, float, float]
    confidence: float


class DiscoveryRelocationGate:
    """Environment-side event gate; it never returns the new position.

    The eventual physical runner must additionally verify target identity in
    its independent evaluation. A detector event alone is not proof that the
    correct object was seen. This gate is not yet an end-to-end experiment.
    """
    def __init__(self,pickup_query,minimum_standoff_m=1.,maximum_age_steps=120,minimum_confidence=.25):
        self.pickup_query=pickup_query
        self.minimum_standoff_m=minimum_standoff_m
        self.maximum_age_steps=maximum_age_steps
        self.minimum_confidence=minimum_confidence
        self.discovery=None
        self.started_step=None

    def record(self,detection):
        if (len(detection.point_world)!=3 or not math.isfinite(detection.confidence)
                or not 0<=detection.confidence<=1):
            raise ValueError("Invalid observed detection geometry or confidence")
        if detection.query!=self.pickup_query or detection.confidence<self.minimum_confidence:
            return
        if detection.observation_id<1 or detection.sim_step<0 or not all(map(math.isfinite,detection.point_world)):
            raise ValueError("Invalid observation-grounded discovery")
        if self.discovery is None or detection.sim_step>=self.discovery.sim_step:
            self.discovery=detection

    def should_start(self,*,sim_step,robot_xy,translating,manipulation_started=False):
        if len(robot_xy)!=2 or not all(map(math.isfinite,robot_xy)):
            raise ValueError("Expected finite robot odometry XY")
        if self.started_step is not None or self.discovery is None or not translating or manipulation_started:
            return False
        age=sim_step-self.discovery.sim_step
        if not 0<=age<=self.maximum_age_steps:
            return False
        distance=math.hypot(*(a-b for a,b in zip(self.discovery.point_world[:2],robot_xy)))
        if not math.isfinite(distance) or distance<self.minimum_standoff_m:
            return False
        self.started_step=sim_step
        return True
