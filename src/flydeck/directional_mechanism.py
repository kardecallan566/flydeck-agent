from __future__ import annotations

from dataclasses import dataclass
import math

from .market_retina import RetinaStimulus
from .receptive_fields import ReceptiveField


@dataclass(frozen=True, slots=True)
class DirectionalResponse:
    excitation: float
    inhibition: float
    response: float


class SpatialOffsetDirectionalMechanism:
    """T4/T5-like directional subunit from offset excitation and inhibition.

    The mechanism is intentionally small: excitation samples the leading-side
    receptive-field component with a fast response, while inhibition samples
    the anatomically offset inhibitory component with slower temporal decay.
    Direction is therefore produced by the order in which the two spatially
    separated components are stimulated, not by a hard-coded preferred
    direction.

    This is a functional abstraction of the measured T4 mechanism, not a
    biophysical simulation. The receptive-field centers come from the
    connectome-derived anatomical influence estimate.
    """

    def __init__(
        self,
        fields: dict[int, ReceptiveField],
        *,
        inhibition_alpha: float = 0.20,
    ) -> None:
        if not 0.0 < inhibition_alpha <= 1.0:
            raise ValueError("inhibition_alpha must be between 0 and 1")
        self.fields = fields
        self.inhibition_alpha = inhibition_alpha
        self._inhibition_state = {
            index: 0.0
            for index, field in fields.items()
            if field.has_inhibitory_component
        }
        self._previous_stimulus: RetinaStimulus | None = None

    def reset(self) -> None:
        for index in self._inhibition_state:
            self._inhibition_state[index] = 0.0
        self._previous_stimulus = None

    def step(
        self,
        stimulus: RetinaStimulus,
        outputs: tuple[tuple[int, ...], ...],
        *,
        polarity: str,
    ) -> tuple[float, ...]:
        """Return one directional response per output group.

        T4 uses the ON field and T5 uses the OFF field. The current stimulus
        is sampled at the propagated excitatory/inhibitory RF centers. A fast
        excitation is paired with a delayed inhibitory trace, so reversing a
        moving stimulus reverses the temporal order of the two terms.
        """
        if polarity not in {"on", "off"}:
            raise ValueError("polarity must be 'on' or 'off'")
        field = stimulus.on_field if polarity == "on" else stimulus.off_field
        previous_field = (
            None
            if self._previous_stimulus is None
            else self._previous_stimulus.on_field
            if polarity == "on"
            else self._previous_stimulus.off_field
        )

        values = [0.0] * len(self.fields)
        for index, rf in self.fields.items():
            if rf.excitatory_x is None or rf.excitatory_y is None:
                continue
            excitation = _sample(field, rf.excitatory_x, rf.excitatory_y)
            previous_excitation = _sample(previous_field, rf.excitatory_x, rf.excitatory_y)
            fast_excitation = max(0.0, excitation)
            excitation_onset = max(0.0, excitation - previous_excitation)

            if rf.inhibitory_x is None or rf.inhibitory_y is None:
                inhibition = 0.0
            else:
                inhibition_input = _sample(field, rf.inhibitory_x, rf.inhibitory_y)
                previous_inhibition = self._inhibition_state.get(index, 0.0)
                inhibition = previous_inhibition + self.inhibition_alpha * (
                    inhibition_input - previous_inhibition
                )
                self._inhibition_state[index] = inhibition

            # Keep the fast excitatory component and the onset transient. The
            # onset term makes the causal ordering of E and I visible without
            # using the retina's direction metadata.
            values[index] = max(0.0, fast_excitation + excitation_onset - inhibition)

        self._previous_stimulus = stimulus

        group_values: list[float] = []
        for group in outputs:
            group_values.append(
                sum(values[index] for index in group) / max(1, len(group))
            )
        return tuple(group_values)

    def response_for_neuron(
        self,
        neuron_index: int,
        stimulus: RetinaStimulus,
        *,
        polarity: str,
    ) -> DirectionalResponse:
        """Expose the E/I terms for controlled directional diagnostics."""
        if neuron_index not in self.fields:
            raise KeyError(f"no receptive field for neuron {neuron_index}")
        rf = self.fields[neuron_index]
        field = stimulus.on_field if polarity == "on" else stimulus.off_field
        excitation = 0.0
        inhibition = 0.0
        if rf.excitatory_x is not None and rf.excitatory_y is not None:
            excitation = _sample(field, rf.excitatory_x, rf.excitatory_y)
        if rf.inhibitory_x is not None and rf.inhibitory_y is not None:
            inhibition = _sample(field, rf.inhibitory_x, rf.inhibitory_y)
        return DirectionalResponse(
            excitation=excitation,
            inhibition=inhibition,
            response=max(0.0, excitation - inhibition),
        )


def _sample(
    field: tuple[tuple[float, ...], ...] | None,
    x: float | None,
    y: float | None,
) -> float:
    if field is None or not field or not field[0] or x is None or y is None:
        return 0.0
    height = len(field)
    width = len(field[0])
    fx = min(width - 1.0, max(0.0, x * (width - 1)))
    fy = min(height - 1.0, max(0.0, y * (height - 1)))
    x0, y0 = int(math.floor(fx)), int(math.floor(fy))
    x1, y1 = min(width - 1, x0 + 1), min(height - 1, y0 + 1)
    dx, dy = fx - x0, fy - y0
    return (
        field[y0][x0] * (1.0 - dx) * (1.0 - dy)
        + field[y0][x1] * dx * (1.0 - dy)
        + field[y1][x0] * (1.0 - dx) * dy
        + field[y1][x1] * dx * dy
    )
