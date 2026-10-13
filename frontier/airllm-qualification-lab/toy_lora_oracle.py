"""Independent, SIMULATED LoRA algebra oracle using only Python's standard library.

All matrices are tiny artificial fixtures. The streamed path models recomputation;
it does not offload real tensors, execute a language model, measure memory, or
authorize training. Eligibility and execution-authority fields always stay false.

Prior art: LoRA https://arxiv.org/abs/2106.09685 and activation recomputation
https://arxiv.org/abs/1604.06174. This implementation contains no upstream code.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random


class ContractError(ValueError):
    """An artificial fixture or numerical/planning contract is invalid."""


_BASE = (
    ((0.12, -0.23, 0.34), (0.27, 0.11, -0.18),
     (-0.31, 0.25, 0.09), (0.16, -0.07, 0.29)),
    ((0.15, -0.08, 0.17, 0.03), (-0.21, 0.24, 0.04, -0.11),
     (0.05, 0.19, -0.14, 0.22), (0.23, -0.03, 0.12, 0.18)),
)
_HEAD = ((0.31, -0.12, 0.08, 0.19), (-0.17, 0.28, 0.13, -0.04),
         (0.09, 0.16, -0.26, 0.21), (-0.22, 0.07, 0.18, 0.11),
         (0.14, -0.19, 0.24, -0.15))
SAMPLE_INPUTS = ((0.2, -0.4, 0.7), (-0.3, 0.6, 0.1), (0.8, 0.2, -0.5))
SAMPLE_LABELS = (0, 2, 4)
SAMPLE_ASSISTANT_MASK = (False, True, True)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ContractError("expected a finite non-Boolean number")
    return float(value)


def _matrix(value, rows, columns):
    if type(value) not in (list, tuple) or len(value) != rows:
        raise ContractError("wrong matrix row count")
    result = []
    for row in value:
        if type(row) not in (list, tuple) or len(row) != columns:
            raise ContractError("wrong matrix column count")
        result.append([_number(item) for item in row])
    return result


def _keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ContractError("missing or unexpected contract keys")


def _strict_same(observed, expected):
    """Identity comparisons reject Python's bool/int/float equality aliases."""
    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        return set(observed) == set(expected) and all(
            _strict_same(observed[key], value) for key, value in expected.items())
    if type(expected) in (list, tuple):
        return len(observed) == len(expected) and all(
            _strict_same(left, right) for left, right in zip(observed, expected))
    return observed == expected


def _zeros(rows, columns):
    return [[0.0 for _ in range(columns)] for _ in range(rows)]


@dataclass(frozen=True)
class ToyProfile:
    """Fixed dense algebra fixture; binary64 arithmetic is MODELED, not ML."""

    base: tuple = _BASE
    head: tuple = _HEAD
    rank: int = 2
    alpha: float = 4.0
    dropout_probability: float = 0.0

    def __post_init__(self):
        if type(self.base) is not tuple or len(self.base) != 2:
            raise ContractError("exactly two immutable toy layers required")
        for value in (*self.base, self.head):
            if type(value) is not tuple or any(type(row) is not tuple for row in value):
                raise ContractError("frozen matrices must use tuples")
        _matrix(self.base[0], 4, 3)
        _matrix(self.base[1], 4, 4)
        _matrix(self.head, 5, 4)
        if type(self.rank) is not int or self.rank != 2:
            raise ContractError("this tiny fixture has exactly rank two")
        if _number(self.alpha) <= 0.0:
            raise ContractError("alpha must be positive")
        if not 0.0 <= _number(self.dropout_probability) < 1.0:
            raise ContractError("dropout must be in [0, 1)")

    @property
    def identity(self):
        return {
            "profile": "szl-artificial-dense-lora-v1",
            "arithmetic": "python-binary64",
            "base": [[list(row) for row in layer] for layer in self.base],
            "head": [list(row) for row in self.head],
            "rank": self.rank,
            "alpha": float(self.alpha),
            "dropout_probability": float(self.dropout_probability),
            "adapter_keys": sorted(adapter_shapes(self)),
            "expert_coverage": "NONE_DENSE_TOY_ONLY",
        }


def adapter_shapes(profile):
    shapes = {}
    for index, layer in enumerate(profile.base):
        shapes[f"layer.{index}.A"] = (profile.rank, len(layer[0]))
        shapes[f"layer.{index}.B"] = (len(layer), profile.rank)
    return shapes


def artificial_adapters(profile=None):
    profile = profile or ToyProfile()
    values = {
        "layer.0.A": [[0.05, -0.03, 0.04], [-0.02, 0.06, 0.01]],
        "layer.0.B": [[0.07, -0.04], [0.03, 0.08], [-0.05, 0.02], [0.09, 0.01]],
        "layer.1.A": [[0.04, 0.02, -0.03, 0.05], [-0.01, 0.07, 0.06, -0.02]],
        "layer.1.B": [[0.02, 0.06], [-0.04, 0.09], [0.08, -0.05], [0.03, 0.07]],
    }
    return _adapters(profile, values)


def _adapters(profile, values):
    shapes = adapter_shapes(profile)
    _keys(values, shapes)
    return {key: _matrix(values[key], *shape) for key, shape in shapes.items()}


def validate_batch(inputs, labels, assistant_mask):
    if type(inputs) not in (list, tuple) or not inputs:
        raise ContractError("nonempty artificial batch required")
    values = _matrix(inputs, len(inputs), 3)
    if type(labels) not in (list, tuple) or type(assistant_mask) not in (list, tuple):
        raise ContractError("labels and assistant mask must be sequences")
    if len(labels) != len(values) or len(assistant_mask) != len(values):
        raise ContractError("label/mask length mismatch")
    if any(type(flag) is not bool for flag in assistant_mask):
        raise ContractError("assistant mask must contain exact Booleans")
    for label, selected in zip(labels, assistant_mask):
        if type(label) is not int or not (0 <= label < 5 or (label == -100 and not selected)):
            raise ContractError("invalid label for artificial frozen head")
    if not any(assistant_mask):
        raise ContractError("no supervised assistant tokens survive")
    return values, tuple(labels), tuple(assistant_mask)


def make_dropout_replay(profile, batch_size, seed=17, rng=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ContractError("positive exact batch size required")
    if rng is None:
        if type(seed) is not int:
            raise ContractError("exact integer seed required")
        rng = random.Random(seed)
    probability = profile.dropout_probability
    factor = 1.0 / (1.0 - probability)
    return {
        index: tuple(tuple(1.0 if probability == 0.0 else
                           (factor if rng.random() >= probability else 0.0)
                           for _ in layer[0]) for _ in range(batch_size))
        for index, layer in enumerate(profile.base)
    }


def _replay(profile, batch_size, value):
    if value is None:
        if profile.dropout_probability != 0.0:
            raise ContractError("nonzero dropout requires an explicit replay")
        value = make_dropout_replay(profile, batch_size)
    _keys(value, range(len(profile.base)))
    allowed = {0.0, 1.0 / (1.0 - profile.dropout_probability)}
    if profile.dropout_probability == 0.0:
        allowed = {1.0}
    result = {}
    for index, layer in enumerate(profile.base):
        rows = _matrix(value[index], batch_size, len(layer[0]))
        if any(item not in allowed for row in rows for item in row):
            raise ContractError("dropout replay contains invalid factors")
        result[index] = tuple(tuple(row) for row in rows)
    return result


def _layer_forward(profile, index, hidden, adapters, replay):
    weight = profile.base[index]
    a = adapters[f"layer.{index}.A"]
    b = adapters[f"layer.{index}.B"]
    scale = profile.alpha / profile.rank
    output, cache = [], []
    for x, factors in zip(hidden, replay[index]):
        dropped = [value * factor for value, factor in zip(x, factors)]
        z = [sum(value * coefficient for value, coefficient in zip(dropped, row)) for row in a]
        pre = [sum(value * coefficient for value, coefficient in zip(x, row))
               + scale * sum(value * coefficient for value, coefficient in zip(z, b_row))
               for row, b_row in zip(weight, b)]
        y = [math.tanh(value) for value in pre]
        if any(not math.isfinite(value) for value in pre):
            raise ContractError("nonfinite layer preactivation")
        output.append(y)
        cache.append((x[:], dropped, z, y))
    return output, cache


def _loss_head(profile, hidden, labels, mask):
    count = sum(mask)
    loss = 0.0
    gradient = _zeros(len(hidden), 4)
    for sample, (h, label, selected) in enumerate(zip(hidden, labels, mask)):
        if not selected:
            continue
        logits = [sum(x * w for x, w in zip(h, row)) for row in profile.head]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        denominator = sum(exponentials)
        loss += (maximum + math.log(denominator) - logits[label]) / count
        dlogits = [(value / denominator - float(index == label)) / count
                   for index, value in enumerate(exponentials)]
        gradient[sample] = [sum(d * row[column] for d, row in zip(dlogits, profile.head))
                            for column in range(4)]
    if not math.isfinite(loss):
        raise ContractError("nonfinite artificial loss")
    return loss, gradient


def _layer_backward(profile, index, cache, output_gradient, adapters, replay):
    weight = profile.base[index]
    a = adapters[f"layer.{index}.A"]
    b = adapters[f"layer.{index}.B"]
    scale = profile.alpha / profile.rank
    ga, gb = _zeros(profile.rank, len(weight[0])), _zeros(len(weight), profile.rank)
    input_gradient = []
    for saved, go, factors in zip(cache, output_gradient, replay[index]):
        x, dropped, z, y = saved
        du = [gradient * (1.0 - value * value) for gradient, value in zip(go, y)]
        dz = [scale * sum(du[o] * b[o][r] for o in range(len(weight)))
              for r in range(profile.rank)]
        for o in range(len(weight)):
            for r in range(profile.rank):
                gb[o][r] += scale * du[o] * z[r]
        for r in range(profile.rank):
            for i in range(len(x)):
                ga[r][i] += dz[r] * dropped[i]
        input_gradient.append([
            sum(du[o] * weight[o][i] for o in range(len(weight)))
            + factors[i] * sum(dz[r] * a[r][i] for r in range(profile.rank))
            for i in range(len(x))
        ])
    return input_gradient, {f"layer.{index}.A": ga, f"layer.{index}.B": gb}


@dataclass(frozen=True)
class SimulationResult:
    loss: float
    gradients: dict
    input_gradients: tuple
    identity: dict
    events: tuple
    evidence_class: str = "SIMULATED"
    training_eligible: bool = False
    publication_eligible: bool = False
    execution_authority: bool = False


def resident(profile, adapters, inputs, labels, assistant_mask, dropout_replay=None):
    hidden, labels, mask = validate_batch(inputs, labels, assistant_mask)
    values = _adapters(profile, adapters)
    replay = _replay(profile, len(hidden), dropout_replay)
    caches = []
    for index in range(len(profile.base)):
        hidden, cache = _layer_forward(profile, index, hidden, values, replay)
        caches.append(cache)
    loss, gradient = _loss_head(profile, hidden, labels, mask)
    gradients = {}
    for index in reversed(range(len(profile.base))):
        gradient, local = _layer_backward(profile, index, caches[index], gradient, values, replay)
        gradients.update(local)
    return SimulationResult(loss, gradients, tuple(tuple(row) for row in gradient),
                            profile.identity, ())


def streamed_recompute(profile, adapters, inputs, labels, assistant_mask,
                       dropout_replay=None, recompute_replay=None):
    hidden, labels, mask = validate_batch(inputs, labels, assistant_mask)
    values = _adapters(profile, adapters)
    replay = _replay(profile, len(hidden), dropout_replay)
    backward_replay = replay if recompute_replay is None else _replay(
        profile, len(hidden), recompute_replay)
    if replay != backward_replay:
        raise ContractError("forward and backward dropout replay differ")
    saved_inputs, events = [], []
    for index in range(len(profile.base)):
        saved_inputs.append(deepcopy(hidden))
        events.append(("load", index))
        hidden = _layer_forward(profile, index, hidden, values, replay)[0]
        events.extend((("forward", index), ("evict", index)))
    loss, gradient = _loss_head(profile, hidden, labels, mask)
    gradients = {}
    for index in reversed(range(len(profile.base))):
        events.append(("load", index))
        cache = _layer_forward(profile, index, saved_inputs[index], values, backward_replay)[1]
        events.append(("recompute", index))
        gradient, local = _layer_backward(profile, index, cache, gradient, values, backward_replay)
        gradients.update(local)
        events.extend((("backward", index), ("evict", index)))
    return SimulationResult(loss, gradients, tuple(tuple(row) for row in gradient),
                            profile.identity, tuple(events))


def assert_gradient_close(expected, observed, absolute=1e-10, relative=0.0):
    if _number(absolute) < 0.0 or _number(relative) < 0.0:
        raise ContractError("negative tolerance")
    _keys(observed, expected)
    for key, matrix in expected.items():
        values = _matrix(observed[key], len(matrix), len(matrix[0]))
        for row, actual in zip(matrix, values):
            for reference, measured in zip(row, actual):
                reference = _number(reference)
                if abs(reference - measured) > absolute + relative * abs(reference):
                    raise ContractError(f"gradient mismatch: {key}")


def assert_parity(expected, observed, tolerance=1e-10):
    if _number(tolerance) < 0.0:
        raise ContractError("negative parity tolerance")
    for result in (expected, observed):
        if result.evidence_class != "SIMULATED" or any(
                flag is not False for flag in (result.training_eligible,
                                               result.publication_eligible,
                                               result.execution_authority)):
            raise ContractError("simulation cannot authorize a real action")
    if not _strict_same(observed.identity, expected.identity):
        raise ContractError("profile identity mismatch")
    if abs(_number(expected.loss) - _number(observed.loss)) > tolerance:
        raise ContractError("loss mismatch")
    assert_gradient_close(expected.gradients, observed.gradients, tolerance)
    assert_gradient_close({"input": expected.input_gradients},
                          {"input": observed.input_gradients}, tolerance)


def finite_difference(profile, adapters, inputs, labels, assistant_mask,
                      dropout_replay=None, epsilon=1e-6):
    if _number(epsilon) <= 0.0:
        raise ContractError("positive finite-difference epsilon required")
    values = _adapters(profile, adapters)
    output = {key: _zeros(*shape) for key, shape in adapter_shapes(profile).items()}
    for key, (rows, columns) in adapter_shapes(profile).items():
        for row in range(rows):
            for column in range(columns):
                plus, minus = deepcopy(values), deepcopy(values)
                plus[key][row][column] += epsilon
                minus[key][row][column] -= epsilon
                lp = resident(profile, plus, inputs, labels, assistant_mask, dropout_replay).loss
                lm = resident(profile, minus, inputs, labels, assistant_mask, dropout_replay).loss
                output[key][row][column] = (lp - lm) / (2.0 * epsilon)
    return output


@dataclass(frozen=True)
class AdamWPolicy:
    lr: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self):
        if _number(self.lr) <= 0.0 or _number(self.epsilon) <= 0.0:
            raise ContractError("positive optimizer lr/epsilon required")
        if not 0.0 <= _number(self.beta1) < 1.0 or not 0.0 <= _number(self.beta2) < 1.0:
            raise ContractError("optimizer betas must be in [0, 1)")
        if _number(self.weight_decay) < 0.0:
            raise ContractError("negative weight decay")


def zero_optimizer_state(profile):
    return {"step": 0, "moments": {
        key: {"m": _zeros(*shape), "v": _zeros(*shape)}
        for key, shape in adapter_shapes(profile).items()
    }}


def _optimizer_state(profile, state):
    _keys(state, ("step", "moments"))
    if type(state["step"]) is not int or state["step"] < 0:
        raise ContractError("optimizer step must be a nonnegative exact integer")
    _keys(state["moments"], adapter_shapes(profile))
    result = {"step": state["step"], "moments": {}}
    for key, shape in adapter_shapes(profile).items():
        _keys(state["moments"][key], ("m", "v"))
        m = _matrix(state["moments"][key]["m"], *shape)
        v = _matrix(state["moments"][key]["v"], *shape)
        if any(value < 0.0 for row in v for value in row):
            raise ContractError("negative second moment")
        result["moments"][key] = {"m": m, "v": v}
    return result


def adamw_step(profile, adapters, gradients, state, policy):
    values = _adapters(profile, adapters)
    gradient = _adapters(profile, gradients)
    moments = _optimizer_state(profile, state)
    moments["step"] += 1
    step = moments["step"]
    for key, (rows, columns) in adapter_shapes(profile).items():
        for row in range(rows):
            for column in range(columns):
                g = gradient[key][row][column]
                entry = moments["moments"][key]
                m = policy.beta1 * entry["m"][row][column] + (1.0 - policy.beta1) * g
                v = policy.beta2 * entry["v"][row][column] + (1.0 - policy.beta2) * g * g
                update = (m / (1.0 - policy.beta1 ** step)) / (
                    math.sqrt(v / (1.0 - policy.beta2 ** step)) + policy.epsilon)
                value = values[key][row][column] * (1.0 - policy.lr * policy.weight_decay)
                values[key][row][column] = _number(value - policy.lr * update)
                entry["m"][row][column], entry["v"][row][column] = _number(m), _number(v)
    return values, moments


def _rng_snapshot(rng):
    version, state, gauss = rng.getstate()
    return {"version": version, "state": list(state), "gauss": gauss}


def _rng_restore(snapshot):
    _keys(snapshot, ("version", "state", "gauss"))
    if type(snapshot["version"]) is not int or snapshot["version"] != 3:
        raise ContractError("unsupported artificial RNG state version")
    state = snapshot["state"]
    if type(state) is not list or len(state) != 625 or any(
            type(value) is not int or not 0 <= value < 2 ** 32 for value in state[:-1]):
        raise ContractError("invalid artificial RNG state")
    if type(state[-1]) is not int or not 0 <= state[-1] <= 624:
        raise ContractError("invalid artificial RNG cursor")
    gauss = snapshot["gauss"]
    if gauss is not None:
        gauss = _number(gauss)
    rng = random.Random(0)
    try:
        rng.setstate((3, tuple(state), gauss))
    except (ValueError, TypeError) as error:
        raise ContractError("invalid artificial RNG state") from error
    return rng


class OracleSession:
    """Artificial algebra updates only; no language-model trainer or file I/O."""

    def __init__(self, profile=None, seed=17, policy=None, adapters=None):
        if type(seed) is not int:
            raise ContractError("exact integer seed required")
        self.profile = ToyProfile() if profile is None else profile
        self.policy = AdamWPolicy() if policy is None else policy
        if type(self.profile) is not ToyProfile or type(self.policy) is not AdamWPolicy:
            raise ContractError("explicit toy profile and optimizer policy required")
        self.adapters = _adapters(self.profile, artificial_adapters(self.profile)
                                  if adapters is None else adapters)
        self.optimizer = zero_optimizer_state(self.profile)
        self.rng = random.Random(seed)
        self.dataset_cursor = {"batches_seen": 0, "samples_seen": 0}

    def advance(self, inputs, labels, assistant_mask, mode="resident"):
        values, labels, mask = validate_batch(inputs, labels, assistant_mask)
        if mode not in ("resident", "streamed"):
            raise ContractError("unsupported artificial execution mode")
        local_rng = _rng_restore(_rng_snapshot(self.rng))
        replay = make_dropout_replay(self.profile, len(values), rng=local_rng)
        function = resident if mode == "resident" else streamed_recompute
        result = function(self.profile, self.adapters, values, labels, mask, replay)
        adapters, state = adamw_step(self.profile, self.adapters, result.gradients,
                                     self.optimizer, self.policy)
        self.adapters, self.optimizer, self.rng = adapters, state, local_rng
        self.dataset_cursor["batches_seen"] += 1
        self.dataset_cursor["samples_seen"] += len(values)
        return result

    def checkpoint(self):
        return deepcopy({
            "schema": "szl-simulated-toy-checkpoint-v1",
            "evidence_class": "SIMULATED",
            "training_eligible": False,
            "publication_eligible": False,
            "execution_authority": False,
            "identity": self.profile.identity,
            "optimizer_policy": asdict(self.policy),
            "adapters": self.adapters,
            "optimizer_state": self.optimizer,
            "rng_state": _rng_snapshot(self.rng),
            "dataset_cursor": self.dataset_cursor,
        })

    @classmethod
    def from_checkpoint(cls, profile, policy, checkpoint):
        _keys(checkpoint, ("schema", "evidence_class", "training_eligible",
                           "publication_eligible", "execution_authority", "identity",
                           "optimizer_policy", "adapters", "optimizer_state",
                           "rng_state", "dataset_cursor"))
        if checkpoint["schema"] != "szl-simulated-toy-checkpoint-v1" or checkpoint[
                "evidence_class"] != "SIMULATED":
            raise ContractError("unsupported simulated checkpoint")
        if any(checkpoint[name] is not False for name in (
                "training_eligible", "publication_eligible", "execution_authority")):
            raise ContractError("checkpoint cannot authorize a real action")
        if not _strict_same(checkpoint["identity"], profile.identity) or not _strict_same(
                checkpoint["optimizer_policy"], asdict(policy)):
            raise ContractError("immutable base/adapter/optimizer identity mismatch")
        adapters = _adapters(profile, checkpoint["adapters"])
        optimizer = _optimizer_state(profile, checkpoint["optimizer_state"])
        cursor = checkpoint["dataset_cursor"]
        _keys(cursor, ("batches_seen", "samples_seen"))
        if any(type(value) is not int or value < 0 for value in cursor.values()):
            raise ContractError("invalid dataset cursor")
        if cursor["batches_seen"] != optimizer["step"] or cursor["samples_seen"] < cursor["batches_seen"]:
            raise ContractError("dataset/optimizer cursor mismatch")
        rng = _rng_restore(checkpoint["rng_state"])
        session = cls(profile, policy=policy, adapters=adapters)
        session.optimizer, session.rng, session.dataset_cursor = optimizer, rng, deepcopy(cursor)
        return session
