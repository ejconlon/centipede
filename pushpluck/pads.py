"""Pad interface management for the PushPluck fretboard.

This module manages the visual and interactive aspects of the pad grid,
including color mapping based on musical scales, note triggering,
and coordination with the fretboard and viewport systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from pushpluck.color import Color
from pushpluck.config import ColorScheme, Config, NoteType, PadColorMapper, VisState
from pushpluck.fretboard import BoundedConfig, Fretboard, NoteEffects
from pushpluck.midi import MidiSink
from pushpluck.pos import Pos
from pushpluck.push import PadEvent, PushInterface
from pushpluck.scale import NoteName, Scale, ScaleClassifier, name_and_octave_from_note
from pushpluck.viewport import Viewport


@dataclass(frozen=True)
class PadsConfig:
    """Configuration for the pads interface extracted from the main config.

    Contains the musical information needed to determine pad colors
    and behavior based on scale theory.
    """

    scale: Scale  # The current musical scale
    """The musical scale used for note classification and coloring."""
    root: NoteName  # The root note of the scale
    """The root note of the current scale."""

    @classmethod
    def extract(cls, root_config: Config) -> PadsConfig:
        """Extract pad-relevant configuration from the main config.

        Args:
            root_config: The main application configuration.

        Returns:
            A PadsConfig with the scale and root information.
        """
        return PadsConfig(scale=root_config.scale, root=root_config.root)


@dataclass
class SinglePadState:
    """State information for an individual pad.

    Tracks both the color mapping strategy (based on note type) and
    the current visual state (active, disabled, etc.) for a single pad.
    """

    mapper: PadColorMapper  # Strategy for determining pad color
    """The color mapping strategy for this pad (note, misc, or control)."""
    vis: VisState  # Current visual state
    """The current visual state of this pad (Off, OnPrimary, etc.)."""

    def color(self, scheme: ColorScheme) -> Optional[Color]:
        """Get the current color for this pad.

        Args:
            scheme: The color scheme to use.

        Returns:
            The color this pad should display, or None if off.
        """
        return self.mapper.get_color(scheme, self.vis)


@dataclass
class PadsState:
    """Overall state for the entire pad grid.

    Maintains the state of all individual pads in the 8x8 grid,
    providing a centralized place to manage pad colors and states.
    """

    lookup: Dict[Pos, SinglePadState]  # State for each pad position
    """Dictionary mapping pad positions to their individual states."""

    @classmethod
    def default(cls) -> PadsState:
        """Create a default pad state with all pads off.

        Returns:
            A PadsState with all pads set to non-interactive misc pads
            in the Off visual state.
        """
        return cls(
            {
                pos: SinglePadState(PadColorMapper.misc(False), VisState.Off)
                for pos in Pos.iter_all()
            }
        )


class Pads:
    """Manages the pad grid interface for the PushPluck fretboard.

    This class coordinates between the viewport, fretboard, and visual
    display to provide an interactive musical interface on the Push
    controller's pad grid. It handles note triggering, color updates,
    and configuration changes.
    """

    @classmethod
    def construct(cls, scheme: ColorScheme, root_config: Config) -> Pads:
        """Construct a Pads instance from configuration.

        Args:
            scheme: The color scheme for pad visualization.
            root_config: The main application configuration.

        Returns:
            A new Pads instance configured with the given parameters.
        """
        config = PadsConfig.extract(root_config)
        viewport = Viewport.construct(root_config)
        bounds = viewport.str_bounds()
        bounded_config = BoundedConfig(bounds, root_config)
        fretboard = Fretboard.construct(bounded_config)
        return cls(scheme, config, fretboard, viewport)

    def __init__(
        self,
        scheme: ColorScheme,
        config: PadsConfig,
        fretboard: Fretboard,
        viewport: Viewport,
    ) -> None:
        """Initialize the pads interface.

        Args:
            scheme: Color scheme for pad visualization.
            config: Pad-specific configuration.
            fretboard: Fretboard for note logic.
            viewport: Viewport for coordinate mapping.
        """
        self._scheme = scheme
        self._config = config
        self._fretboard = fretboard
        self._viewport = viewport
        self._state = PadsState.default()
        self._reset_pad_colors()

    def _get_pad_color(self, pos: Pos) -> Optional[Color]:
        pad = self._state.lookup[pos]
        return pad.color(self._scheme)

    def _redraw_pos(self, push: PushInterface, pos: Pos) -> None:
        color = self._get_pad_color(pos)
        if color is None:
            push.pad_led_off(pos)
        else:
            push.pad_set_color(pos, color)

    def redraw(self, push: PushInterface) -> None:
        """Redraw all pads on the Push interface.

        Args:
            push: The Push interface for sending display updates.
        """
        for pos in Pos.iter_all():
            self._redraw_pos(push, pos)

    def _make_pad_color_mapper(
        self, classifier: ScaleClassifier, pos: Pos
    ) -> PadColorMapper:
        str_pos = self._viewport.str_pos_from_pad_pos(pos)
        if str_pos is None:
            return PadColorMapper.misc(False)
        else:
            note = self._fretboard.get_note(str_pos)
            if note is None:
                return PadColorMapper.misc(False)
            else:
                name, _ = name_and_octave_from_note(note)
                note_type: NoteType
                if classifier.is_root(name):
                    note_type = NoteType.Root
                elif classifier.is_member(name):
                    note_type = NoteType.Member
                else:
                    note_type = NoteType.Other
                return PadColorMapper.note(note_type)

    def _reset_pad_colors(self) -> None:
        classifier = self._config.scale.to_classifier(self._config.root)
        for pos in Pos.iter_all():
            mapper = self._make_pad_color_mapper(classifier, pos)
            self._state.lookup[pos].mapper = mapper

    def handle_event(
        self, push: PushInterface, sink: MidiSink, event: PadEvent
    ) -> None:
        """Handle a pad press/release event.

        Args:
            push: The Push interface for visual updates.
            sink: MIDI sink for note output.
            event: The pad event to process.
        """
        str_pos = self._viewport.str_pos_from_pad_pos(event.pos)
        if str_pos is not None:
            fx = self._fretboard.trigger(str_pos, event.velocity)
            self._handle_note_effects(push, sink, fx)

    def handle_config(
        self, push: PushInterface, sink: MidiSink, root_config: Config, reset: bool
    ) -> None:
        """Handle a configuration change.

        Args:
            push: The Push interface for visual updates.
            sink: MIDI sink for note output.
            root_config: The new configuration to apply.
            reset: Whether this is a reset operation.
        """
        unit = self._viewport.handle_config(root_config, reset)
        if unit is not None:
            # If there is an updated config, force reset and redraw of pads
            reset = True
        bounds = self._viewport.str_bounds()
        bounded_config = BoundedConfig(bounds, root_config)
        fx = self._fretboard.handle_config(bounded_config, reset)
        if fx is not None:
            self._handle_note_effects(push, sink, fx)
            # If there are note-offs or updated config, force reset and redraw of pads
            reset = True
        config = PadsConfig.extract(root_config)
        if config != self._config or reset:
            self._config = config
            self._reset_pad_colors()
            self.redraw(push)

    def _handle_note_effects(
        self, push: PushInterface, sink: MidiSink, fx: NoteEffects
    ) -> None:
        # Send notes
        for fret_msg in fx.msgs:
            sink.send_msg(fret_msg.msg)
        # Update display
        for sp, vis in fx.vis.items():
            pad_pos = self._viewport.pad_pos_from_str_pos(sp)
            if pad_pos is not None:
                self._state.lookup[pad_pos].vis = vis
                self._redraw_pos(push, pad_pos)
