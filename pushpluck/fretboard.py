"""Fretboard simulation and note handling for the PushPluck application.

This module implements the core fretboard logic, including string positions,
note triggering, channel mapping, tuning systems, and various play modes
(tap, poly, mono). It provides the bridge between physical pad positions
and musical note output.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set, Tuple, cast, override

from mido.frozen import FrozenMessage

from pushpluck import constants
from pushpluck.base import MatchException
from pushpluck.component import MappedComponent, MappedComponentConfig
from pushpluck.config import ChannelMode, Config, PlayMode, VisState
from pushpluck.midi import is_note_msg, is_note_off_msg, is_note_on_msg


@dataclass(frozen=True)
class StringPos:
    """Represents a position on the fretboard as a string and fret combination.

    This class defines a specific location on the virtual fretboard by identifying
    which string and which fret position. Negative fret values are allowed to
    represent positions below the open string pitch.
    """

    str_index: int
    """The string number (0-based index into the tuning array, 0 to max strings)."""
    fret: int
    """The fret position (semitone offset from open string). Negative values allowed."""


@dataclass(frozen=True)
class NoteGroup:
    """Groups equivalent string positions that produce the same MIDI note.

    Since multiple string/fret combinations can produce the same pitch,
    this class groups them together for visualization and interaction purposes.
    The primary position is typically the one that was actually triggered.
    """

    note: int
    """The MIDI note number (0-127) produced by all positions in this group."""
    primary: Optional[StringPos]
    """The primary/active string position that was triggered, if within fretboard bounds."""
    equivs: List[StringPos]
    """All equivalent string positions that produce the same MIDI note number."""


@dataclass(frozen=True)
class FretboardMessage:
    """A MIDI message associated with a fretboard position and its equivalents.

    This class combines a MIDI message with the fretboard context, including
    the primary string position and all equivalent positions that produce the
    same note. Used for tracking note events and their visual effects.
    """

    str_pos: StringPos
    """The primary string position that triggered this message."""
    equivs: List[StringPos]
    """All equivalent string positions that produce the same note.

    Note: These positions may be on different MIDI channels depending on
    the channel mapping configuration.
    """
    msg: FrozenMessage
    """The MIDI message (note_on, note_off, aftertouch) associated with this fretboard event."""

    @property
    def channel(self) -> int:
        """Get the MIDI channel from the underlying message.

        Returns:
            The MIDI channel number (1-16).
        """
        return cast(int, self.msg.channel)  # pyright: ignore

    @property
    def note(self) -> int:
        """Get the MIDI note number from the underlying message.

        Returns:
            The MIDI note number (0-127).
        """
        return cast(int, self.msg.note)  # pyright: ignore

    @property
    def velocity(self) -> Optional[int]:
        """Get the velocity from the underlying message if it's a note message.

        Returns:
            The MIDI velocity (0-127) if this is a note message, None otherwise.
        """
        if self.is_note():
            return cast(int, self.msg.velocity)  # pyright: ignore
        else:
            return None

    def is_note_on(self) -> bool:
        """Check if this represents a note-on event.

        Returns:
            True if this is a note-on message with velocity > 0.
        """
        return is_note_on_msg(self.msg)

    def is_note_off(self) -> bool:
        """Check if this represents a note-off event.

        Returns:
            True if this is a note-off message or note-on with velocity 0.
        """
        return is_note_off_msg(self.msg)

    def is_note(self) -> bool:
        """Check if this represents any kind of note message.

        Returns:
            True if this is either a note-on or note-off message.
        """
        return is_note_msg(self.msg)

    def make_note_msg(self, velocity: int) -> FretboardMessage:
        """Create a new FretboardMessage with the specified velocity.

        Args:
            velocity: The MIDI velocity (0-127) for the new message.

        Returns:
            A new FretboardMessage with the same positions but different velocity.
        """
        return FretboardMessage(
            self.str_pos,
            self.equivs,
            FrozenMessage(
                type="note_on", channel=self.channel, note=self.note, velocity=velocity
            ),
        )

    def make_note_off_msg(self) -> FretboardMessage:
        """Create a note-off message with the same positions.

        Returns:
            A new FretboardMessage representing a note-off event.
        """
        return self.make_note_msg(0)


@dataclass(frozen=True)
class StringBounds:
    """Defines a rectangular region of the fretboard.

    This class represents a bounded area of the fretboard defined by
    minimum and maximum string positions. It provides iteration and
    containment checking for string positions within the bounds.
    """

    low: StringPos
    """The minimum string position (bottom-left corner of the bounds)."""
    high: StringPos
    """The maximum string position (top-right corner of the bounds)."""

    def __iter__(self) -> Generator[StringPos, None, None]:
        """Iterate over all string positions within the bounds.

        Yields:
            StringPos instances for every combination of string and fret
            within the bounded region.
        """
        for str_index in range(self.low.str_index, self.high.str_index + 1):
            for fret in range(self.low.fret, self.high.fret + 1):
                yield StringPos(str_index=str_index, fret=fret)

    def __contains__(self, cand: StringPos) -> bool:
        """Check if a string position is within these bounds.

        Args:
            cand: The string position to test.

        Returns:
            True if the position is within the bounds, False otherwise.
        """
        return (
            cand.str_index >= self.low.str_index
            and cand.str_index <= self.high.str_index
            and cand.fret >= self.low.fret
            and cand.fret <= self.high.fret
        )


@dataclass(frozen=True)
class NoteEffects:
    """Encapsulates the visual and MIDI effects of a fretboard action.

    This class holds both the visual state changes for the fretboard display
    and the MIDI messages that should be sent as a result of user interaction.
    It's used to batch together related effects from note triggering.
    """

    vis: Dict[StringPos, VisState]
    """Dictionary mapping string positions to their new visual states."""
    msgs: List[FretboardMessage]
    """List of MIDI messages that should be sent to the output."""

    @classmethod
    def empty(cls) -> NoteEffects:
        """Create an empty NoteEffects instance with no changes.

        Returns:
            A NoteEffects instance with no visual changes or MIDI messages.
        """
        return cls({}, [])

    def is_empty(self) -> bool:
        """Check if this NoteEffects instance contains any changes.

        Returns:
            True if there are no visual changes or MIDI messages, False otherwise.
        """
        return len(self.vis) == 0 and len(self.msgs) == 0


class ChannelMapper(metaclass=ABCMeta):
    """Abstract base class for mapping string positions to MIDI channels.

    Different channel mapping strategies can be implemented by subclassing
    this class. Examples include single-channel mapping (all notes on one channel)
    and multi-channel mapping (different strings on different channels).
    """

    @abstractmethod
    def map_channel(self, str_pos: StringPos) -> Optional[int]:
        """Map a string position to a MIDI channel.

        Args:
            str_pos: The string position to map.

        Returns:
            The MIDI channel number (1-16) or None if the position
            should not produce any output.
        """
        raise NotImplementedError()


class SingleChannelMapper(ChannelMapper):
    """Maps all string positions to a single MIDI channel.

    This is the simplest channel mapping strategy, where all notes
    from all strings are sent on the same MIDI channel.
    """

    def __init__(self, channel: int) -> None:
        """Initialize the mapper with a specific channel.

        Args:
            channel: The MIDI channel number (1-16) to use for all notes.
        """
        self._channel = channel

    @override
    def map_channel(self, str_pos: StringPos) -> Optional[int]:
        """Map any string position to the configured channel.

        Args:
            str_pos: The string position (ignored in this implementation).

        Returns:
            The configured MIDI channel number (1-16).
        """
        return self._channel


class MultiChannelMapper(ChannelMapper):
    """Maps different strings to different MIDI channels.

    This mapper assigns each string to its own MIDI channel, starting from
    a base channel. This allows for per-string effects and processing.
    Strings that would map to channels outside the valid range are silenced.
    """

    def __init__(self, base_channel: int, min_channel: int, max_channel: int) -> None:
        """Initialize the multi-channel mapper.

        Args:
            base_channel: The MIDI channel for string 0.
            min_channel: The minimum allowed MIDI channel.
            max_channel: The maximum allowed MIDI channel.
        """
        self._base_channel = base_channel
        self._min_channel = min_channel
        self._max_channel = max_channel

    @override
    def map_channel(self, str_pos: StringPos) -> Optional[int]:
        """Map a string position to its corresponding MIDI channel.

        Args:
            str_pos: The string position to map.

        Returns:
            The MIDI channel number for this string, or None if the
            calculated channel is outside the valid range.
        """
        channel = str_pos.str_index + self._base_channel
        if channel < self._min_channel or channel > self._max_channel:
            return None
        else:
            return channel


class Tuner(metaclass=ABCMeta):
    """Abstract base class for converting string positions to MIDI notes.

    A tuner defines the relationship between fretboard positions and
    MIDI note numbers. Different tuners can implement different string
    tunings and fretboard layouts.
    """

    @abstractmethod
    def get_note(self, str_pos: StringPos) -> Optional[int]:
        """Get the MIDI note number for a string position.

        Args:
            str_pos: The string position to convert.

        Returns:
            The MIDI note number (0-127) or None if the position
            is not valid or out of bounds.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_note_group(self, str_pos: StringPos) -> Optional[NoteGroup]:
        """Get the note group containing all equivalent positions.

        Args:
            str_pos: The string position to analyze.

        Returns:
            A NoteGroup containing the MIDI note and all equivalent
            string positions, or None if the position is invalid.
        """
        raise NotImplementedError()


class FixedTuner(Tuner):
    """A tuner implementation for instruments with a fixed number of strings.

    This tuner uses a predefined list of open string pitches and calculates
    fretted notes by adding the fret number as semitone offsets. It precomputes
    lookup tables for efficiency and supports bounded fretboard regions.

    Note: An alternative design could use interval patterns between strings
    to support an "infinite" number of strings, but this implementation
    uses explicit tuning for a fixed set of strings.
    """

    def __init__(self, tuning: List[int], bounds: Optional[StringBounds]) -> None:
        """Initialize the fixed tuner with string tunings and optional bounds.

        Args:
            tuning: List of MIDI note numbers for each open string.
            bounds: Optional bounding region for the fretboard. If None,
                   all positions are considered invalid.
        """
        self._tuning = tuning
        self._bounds = bounds
        self._note_lookup = self._make_note_lookup()
        self._equivs_lookup = self._make_equivs_lookup()

    def _make_note_lookup(self) -> Dict[StringPos, int]:
        """Build a lookup table from string positions to MIDI note numbers.

        Returns:
            Dictionary mapping valid StringPos instances to their MIDI note numbers.
        """
        lookup: Dict[StringPos, int] = {}
        if self._bounds is not None:
            for str_pos in self._bounds:
                if str_pos.str_index < 0 or str_pos.str_index >= len(self._tuning):
                    pass
                else:
                    lookup[str_pos] = self._tuning[str_pos.str_index] + str_pos.fret
        return lookup

    def _make_equivs_lookup(self) -> Dict[int, List[StringPos]]:
        """Build a lookup table from MIDI notes to equivalent string positions.

        Returns:
            Dictionary mapping MIDI note numbers to lists of StringPos instances
            that produce that note.
        """
        lookup: Dict[int, List[StringPos]] = {}
        if self._bounds is not None:
            for str_pos in self._bounds:
                note = self.get_note(str_pos)
                if note is not None:
                    if note in lookup:
                        lookup[note].append(str_pos)
                    else:
                        lookup[note] = [str_pos]
        return lookup

    @override
    def get_note(self, str_pos: StringPos) -> Optional[int]:
        """Get the MIDI note number for a string position.

        Args:
            str_pos: The string position to convert.

        Returns:
            The MIDI note number or None if the position is invalid.
        """
        return self._note_lookup.get(str_pos)

    @override
    def get_note_group(self, str_pos: StringPos) -> Optional[NoteGroup]:
        """Get the note group for a string position.

        Args:
            str_pos: The string position to analyze.

        Returns:
            A NoteGroup with the note and equivalent positions, or None
            if the position doesn't produce a valid note.
        """
        note = self.get_note(str_pos)
        if note is None:
            return None
        else:
            primary = (
                str_pos
                if self._bounds is not None and str_pos in self._bounds
                else None
            )
            equivs = (
                self._equivs_lookup[note]
                if note is not None and note in self._equivs_lookup
                else []
            )
            return NoteGroup(note, primary, equivs)


class InfiniteTuner(Tuner):
    """A tuner implementation for instruments with infinite string mapping.

    This tuner uses a base tuning pattern and repeat_steps to create
    an infinite mapping from any string index to MIDI notes. It calculates
    notes by taking the string index modulo the tuning length, then adding
    octave offsets based on repeat_steps.
    """

    def __init__(self, tuning: List[int], repeat_steps: int, bounds: Optional[StringBounds]) -> None:
        """Initialize the infinite tuner with base tuning and repeat pattern.

        Args:
            tuning: Base tuning pattern (list of MIDI note numbers).
            repeat_steps: Number of semitones before the pattern repeats.
            bounds: Optional bounding region for valid positions.
        """
        if not tuning:
            raise ValueError("Tuning cannot be empty")
        self._tuning = tuning
        self._repeat_steps = repeat_steps
        self._bounds = bounds

    def _get_note_for_string_index(self, str_index: int) -> int:
        """Calculate the MIDI note for any string index.

        Args:
            str_index: The string index (can be negative or very large).

        Returns:
            The MIDI note number for this string index.
        """
        # Get the base note from the tuning pattern
        base_note = self._tuning[str_index % len(self._tuning)]

        # Calculate how many complete cycles through the tuning we've done
        cycles = str_index // len(self._tuning)

        # Add the repeat_steps offset for each complete cycle
        return base_note + (cycles * self._repeat_steps)

    @override
    def get_note(self, str_pos: StringPos) -> Optional[int]:
        """Get the MIDI note number for a string position.

        Args:
            str_pos: The string position to convert.

        Returns:
            The MIDI note number if valid and within MIDI range, None otherwise.
        """
        # Check bounds if specified
        if self._bounds is not None and str_pos not in self._bounds:
            return None

        # Calculate the base note for this string
        base_note = self._get_note_for_string_index(str_pos.str_index)

        # Add fret offset
        note = base_note + str_pos.fret

        # Check MIDI range (0-127)
        if note < 0 or note > 127:
            return None

        return note

    @override
    def get_note_group(self, str_pos: StringPos) -> Optional[NoteGroup]:
        """Get the note group for a string position.

        For infinite tuning, we don't precompute equivalents since there
        could be infinitely many. We just return the note with empty equivalents.

        Args:
            str_pos: The string position to analyze.

        Returns:
            A NoteGroup with the note and empty equivalents, or None
            if the position doesn't produce a valid note.
        """
        note = self.get_note(str_pos)
        if note is None:
            return None
        else:
            primary = (
                str_pos
                if self._bounds is not None and str_pos in self._bounds
                else None
            )
            # For infinite tuning, we don't compute all equivalents
            # as there could be infinitely many
            equivs: List[StringPos] = []
            return NoteGroup(note, primary, equivs)


class NoteHandler(metaclass=ABCMeta):
    """Abstract base class for handling note triggering in different play modes.

    Note handlers implement different musical behaviors like polyphonic,
    monophonic, or string-specific note handling. They process incoming
    fretboard messages and produce appropriate MIDI output messages.
    """

    @abstractmethod
    def trigger(self, fret_msg: FretboardMessage) -> List[FretboardMessage]:
        """Process a fretboard message and return resulting MIDI messages.

        Args:
            fret_msg: The input fretboard message to process.
                     A velocity of 0 represents a note-off event.

        Returns:
            List of FretboardMessage instances to send as MIDI output.
            May include note-on and note-off messages as appropriate
            for the implemented play mode.
        """
        raise NotImplementedError()


class NoteTracker:
    """Tracks the state of notes and their visual representation on the fretboard.

    This class maintains the current state of which notes are playing on which
    channels and how string positions should be visually represented. It handles
    the coordination between MIDI output and visual feedback.
    """

    def __init__(self, chan_mapper: ChannelMapper) -> None:
        """Initialize the note tracker with a channel mapper.

        Args:
            chan_mapper: The channel mapping strategy to use.
        """
        self._chan_mapper = chan_mapper
        self._notemap: Dict[int, Set[int]] = {}  # Map from channel to set of notes
        self._vis: Dict[StringPos, VisState] = {}

    def is_enabled(self, str_pos: StringPos) -> bool:
        """Check if a string position is enabled for interaction.

        Args:
            str_pos: The string position to check.

        Returns:
            True if the position is enabled (not disabled), False otherwise.
        """
        return self.get_vis(str_pos).enabled

    def get_vis(self, str_pos: StringPos) -> VisState:
        """Get the visual state for a string position.

        Args:
            str_pos: The string position to query.

        Returns:
            The current VisState for this position, defaulting to Off
            if no state has been set.
        """
        vs = self._vis.get(str_pos)
        if vs is None:
            vs = VisState.Off
            self._vis[str_pos] = vs
        return vs

    def _record_note(self, msg: FrozenMessage) -> None:
        """Record a MIDI note message in the internal tracking state.

        Args:
            msg: The MIDI message to record. Should be a note-on or note-off message.
        """
        if is_note_on_msg(msg):
            channel = msg.channel  # pyright: ignore
            note = msg.note  # pyright: ignore
            if channel not in self._notemap:
                self._notemap[channel] = set()
            self._notemap[channel].add(note)
        elif is_note_off_msg(msg):
            channel = msg.channel  # pyright: ignore
            note = msg.note  # pyright: ignore
            notes = self._notemap.get(channel)
            if notes is not None:
                notes.discard(note)

    def record_fx(self, msgs: List[FretboardMessage]) -> NoteEffects:
        """Record the effects of fretboard messages and update visual states.

        This method processes a list of fretboard messages, updating both the
        internal note tracking state and the visual states of affected string
        positions. It handles the complex logic of equivalent positions and
        channel conflicts.

        Args:
            msgs: List of FretboardMessage instances to process.

        Returns:
            NoteEffects containing the visual state changes and messages to output.
        """
        dirty: Set[StringPos] = set()
        for msg in msgs:
            dirty.add(msg.str_pos)
            active = msg.is_note_on()
            vs = VisState.OnPrimary if active else VisState.Off
            self._vis[msg.str_pos] = vs
            for equiv in msg.equivs:
                if equiv == msg.str_pos:
                    continue
                dirty.add(equiv)
                channel = self._chan_mapper.map_channel(equiv)
                if channel is not None:
                    cur_vis = self.get_vis(equiv)
                    if not cur_vis.primary:
                        ws: VisState
                        if active:
                            if channel == msg.channel:
                                ws = VisState.OnDisabled
                            else:
                                ws = VisState.OnLinked
                        else:
                            ws = VisState.Off
                        self._vis[equiv] = ws
            self._record_note(msg.msg)
        vis = {sp: vs for sp, vs in self._vis.items() if sp in dirty}
        return NoteEffects(vis, msgs)

    def clean_fx(self) -> NoteEffects:
        """Create effects to clean up all active notes and reset visual state.

        This method would be used to turn off all active notes and reset
        the visual state, typically during configuration changes or resets.

        Returns:
            NoteEffects that would turn off all notes and reset visuals.
        """
        # Create note-off messages for all active notes
        msgs: List[FretboardMessage] = []
        for channel, notes in self._notemap.items():
            for note in notes:
                # Create a minimal FretboardMessage for note-off
                # We use StringPos(0, 0) as a placeholder since this is cleanup
                msg = FrozenMessage(
                    type="note_off", channel=channel - 1, note=note
                )  # Convert 1-16 to 0-15
                fret_msg = FretboardMessage(
                    str_pos=StringPos(str_index=0, fret=0),
                    equivs=[],
                    msg=msg,
                )
                msgs.append(fret_msg)
        vis = {str_pos: VisState.Off for str_pos in self._vis.keys()}
        return NoteEffects(vis, msgs)


class PolyNoteHandler(NoteHandler):
    """Note handler for polyphonic play mode.

    In polyphonic mode, all notes are played independently without
    interfering with each other. Each note-on and note-off is passed
    through directly.
    """

    @override
    def trigger(self, fret_msg: FretboardMessage) -> List[FretboardMessage]:
        """Pass through the fretboard message unchanged.

        Args:
            fret_msg: The input fretboard message.

        Returns:
            A list containing just the input message, unchanged.
        """
        return [fret_msg]


class MonoNoteHandler(NoteHandler):
    """Note handler for monophonic play mode.

    In monophonic mode, only one note can play at a time. When a new
    note is triggered, the previous note is automatically turned off.
    This creates a legato playing style.
    """

    def __init__(self) -> None:
        """Initialize the monophonic note handler."""
        self._last_off: Optional[FretboardMessage] = None

    @override
    def trigger(self, fret_msg: FretboardMessage) -> List[FretboardMessage]:
        """Handle monophonic note triggering.

        Args:
            fret_msg: The input fretboard message.

        Returns:
            List of messages including note-offs for previous notes
            and note-on for the new note (if applicable).
        """
        msgs: List[FretboardMessage] = []
        if fret_msg.is_note_off():
            if self._last_off is not None and self._last_off == fret_msg:
                msgs.append(self._last_off)
                self._last_off = None
        else:
            if self._last_off is not None:
                msgs.append(self._last_off)
            msgs.append(fret_msg)
            self._last_off = fret_msg.make_note_off_msg()
        return msgs


@dataclass(frozen=True)
class ChokeGroup:
    """Manages a group of notes that can choke each other on a single string.

    In tap/pick mode, multiple frets can be pressed on the same string,
    but only the highest (rightmost) fret should sound. This class tracks
    the active notes on a string and determines which one should be audible.
    """

    note_order: List[int]
    """List of active MIDI note numbers in ascending order."""
    note_info: Dict[int, FretboardMessage]
    """Mapping from MIDI note numbers to their FretboardMessage instances."""

    @classmethod
    def empty(cls) -> ChokeGroup:
        """Create an empty choke group with no active notes.

        Returns:
            A new empty ChokeGroup instance.
        """
        return cls(note_order=[], note_info={})

    def max_msg(self) -> Optional[FretboardMessage]:
        """Get the message for the highest active note.

        Returns:
            The FretboardMessage for the highest note, or None if no notes are active.
        """
        max_note = self.note_order[-1] if len(self.note_order) > 0 else None
        return self.note_info[max_note] if max_note is not None else None

    def trigger(self, fret_msg: FretboardMessage) -> None:
        """Update the choke group with a new fretboard message.

        Args:
            fret_msg: The fretboard message to process (note-on or note-off).
        """
        note_index = bisect_left(self.note_order, fret_msg.note)
        note_exists = (
            len(self.note_order) > note_index
            and note_index >= 0
            and self.note_order[note_index] == fret_msg.note
        )
        if fret_msg.is_note_on():
            if not note_exists:
                self.note_order.insert(note_index, fret_msg.note)
            self.note_info[fret_msg.note] = fret_msg
        else:
            if note_exists:
                del self.note_order[note_index]
            if fret_msg.note in self.note_info:
                del self.note_info[fret_msg.note]


class ChokeNoteHandler(NoteHandler):
    """Note handler implementing string choking behavior for tap/pick mode.

    This handler simulates guitar-like behavior where multiple frets can be
    pressed on the same string, but only the highest fret produces sound.
    It supports hammer-ons and pull-offs by managing the transition between
    active notes on each string.
    """

    def __init__(self, tuning: List[int], repeat_steps: int) -> None:
        """Initialize the choke handler for the computed finite string range.

        Args:
            tuning: Base tuning pattern.
            repeat_steps: Semitone interval for pattern repetition.
        """
        # Calculate the finite range of strings that could produce MIDI notes 0-127
        min_str, max_str = self._calculate_string_range(tuning, repeat_steps)
        self._min_string = min_str
        self._max_string = max_str
        self._fingered = [ChokeGroup.empty() for _ in range(max_str - min_str + 1)]

    def _calculate_string_range(self, tuning: List[int], repeat_steps: int) -> Tuple[int, int]:
        """Calculate the range of string indices that could produce valid MIDI notes.

        Returns:
            Tuple of (min_string_index, max_string_index)
        """
        min_str, max_str = float('inf'), float('-inf')

        for base_idx, base_note in enumerate(tuning):
            for fret in range(-12, 25):  # Reasonable fret range
                if repeat_steps > 0:
                    # Find cycle range that keeps us in MIDI range [0, 127]
                    min_cycles = (0 - base_note - fret) / repeat_steps
                    max_cycles = (127 - base_note - fret) / repeat_steps

                    for cycles in range(int(min_cycles) - 1, int(max_cycles) + 2):
                        note = base_note + fret + cycles * repeat_steps
                        if 0 <= note <= 127:
                            string_idx = cycles * len(tuning) + base_idx
                            min_str = min(min_str, string_idx)
                            max_str = max(max_str, string_idx)

        return int(min_str), int(max_str)

    @override
    def trigger(self, fret_msg: FretboardMessage) -> List[FretboardMessage]:
        """Process a fretboard message with string choking logic.

        Args:
            fret_msg: The input fretboard message to process.

        Returns:
            List of output messages implementing the choking behavior:
            - Single note pluck: note-on for the new note
            - Single note mute: note-off for the previous note
            - Hammer-on/pull-off: note-on for new, note-off for previous
            - Movement on same fret: no output (ignored)
        """
        str_index = fret_msg.str_pos.str_index

        # Check if string index is within our computed range
        if str_index < self._min_string or str_index > self._max_string:
            return []  # String index outside valid range

        # Convert string index to array index
        array_index = str_index - self._min_string
        group = self._fingered[array_index]
        prev_msg = group.max_msg()

        group.trigger(fret_msg)
        cur_msg = group.max_msg()
        out_msgs: List[FretboardMessage] = []
        if cur_msg is None:
            if prev_msg is None:
                pass  # No notes (ignore)
            else:
                out_msgs.append(prev_msg.make_note_off_msg())  # Single note mute
        else:
            if prev_msg is None:
                out_msgs.append(cur_msg)  # Single note pluck
            else:
                if prev_msg == cur_msg:
                    pass  # Movement above fretted string (ignore)
                else:
                    # Hammer-on or pull-off (send on before off to maintain envelope overlap)
                    out_msgs.append(cur_msg)
                    out_msgs.append(prev_msg.make_note_off_msg())
        return out_msgs


@dataclass(frozen=True)
class BoundedConfig:
    """Configuration that includes fretboard bounds along with the main config.

    This class combines a viewport's string bounds with the main application
    configuration, allowing the fretboard to know both its playing constraints
    and its operational parameters.
    """

    bounds: Optional[StringBounds]
    """The visible/active fretboard region, or None if unbounded."""
    config: Config
    """The main application configuration containing tuning, modes, etc."""


@dataclass(frozen=True)
class FretboardConfig(MappedComponentConfig[BoundedConfig]):
    """Fretboard-specific configuration extracted from the main config.

    This class contains only the configuration parameters that are relevant
    to the fretboard component, extracted from the larger application config.
    It implements the MappedComponentConfig pattern for efficient updates.
    """

    chan_mode: ChannelMode
    """How MIDI channels should be assigned (single channel or per-string)."""
    midi_channel: int
    """Base MIDI channel (1-16) for note output."""
    play_mode: PlayMode
    """The play mode (tap, poly, mono) that determines note behavior."""
    tuning: List[int]
    """List of MIDI note numbers for each open string."""
    repeat_steps: int
    """Semitone range for infinite string mapping."""
    min_velocity: int
    """The minimum MIDI velocity for note output."""
    bounds: Optional[StringBounds]
    """The bounded region of the fretboard, or None if unbounded."""

    @classmethod
    def extract(cls, root_config: BoundedConfig) -> FretboardConfig:
        """Extract fretboard-relevant configuration from a BoundedConfig.

        Args:
            root_config: The source BoundedConfig to extract from.

        Returns:
            A new FretboardConfig with the relevant parameters.
        """
        return FretboardConfig(
            chan_mode=root_config.config.chan_mode,
            midi_channel=root_config.config.midi_channel,
            play_mode=root_config.config.play_mode,
            tuning=root_config.config.tuning,
            repeat_steps=root_config.config.repeat_steps,
            min_velocity=root_config.config.min_velocity,
            bounds=root_config.bounds,
        )


def create_tuner(config: FretboardConfig) -> Tuner:
    """Create a tuner instance from fretboard configuration.

    Args:
        config: The fretboard configuration.

    Returns:
        An InfiniteTuner instance configured for infinite string mapping.
    """
    return InfiniteTuner(config.tuning, config.repeat_steps, config.bounds)


def create_chan_mapper(config: FretboardConfig) -> ChannelMapper:
    """Create a channel mapper instance from fretboard configuration.

    Args:
        config: The fretboard configuration.

    Returns:
        Either a SingleChannelMapper or MultiChannelMapper based on the
        channel mode specified in the configuration.
    """
    if config.chan_mode == ChannelMode.Single:
        return SingleChannelMapper(channel=config.midi_channel)
    else:
        return MultiChannelMapper(
            base_channel=config.midi_channel,
            min_channel=constants.MIDI_MIN_CHANNEL,
            max_channel=constants.MIDI_MAX_CHANNEL,
        )


def create_handler(config: FretboardConfig) -> NoteHandler:
    """Create a note handler instance from fretboard configuration.

    Args:
        config: The fretboard configuration.

    Returns:
        A note handler appropriate for the specified play mode:
        - Tap mode: ChokeNoteHandler (guitar-like string behavior)
        - Poly mode: PolyNoteHandler (all notes independent)
        - Mono mode: MonoNoteHandler (one note at a time)

    Raises:
        MatchException: If the play mode is not recognized.
    """
    if config.play_mode == PlayMode.Tap:  # (or Pick mode when implemented)
        return ChokeNoteHandler(config.tuning, config.repeat_steps)
    elif config.play_mode == PlayMode.Poly:
        return PolyNoteHandler()
    elif config.play_mode == PlayMode.Mono:
        return MonoNoteHandler()
    else:
        raise MatchException(config.play_mode)


class Fretboard(MappedComponent[BoundedConfig, FretboardConfig, NoteEffects]):
    """The main fretboard component that coordinates note triggering and tracking.

    This class brings together all the fretboard functionality: tuning, channel
    mapping, note handling, and state tracking. It provides the primary interface
    for triggering notes and handling configuration changes in the fretboard.
    """

    @classmethod
    def construct(cls, root_config: BoundedConfig) -> Fretboard:
        """Construct a Fretboard from a BoundedConfig.

        Args:
            root_config: The bounded configuration to use.

        Returns:
            A new Fretboard instance configured from the input.
        """
        return cls(cls.extract_config(root_config))

    @classmethod
    def extract_config(cls, root_config: BoundedConfig) -> FretboardConfig:
        """Extract fretboard configuration from a BoundedConfig.

        Args:
            root_config: The source bounded configuration.

        Returns:
            A FretboardConfig with the relevant parameters extracted.
        """
        return FretboardConfig.extract(root_config)

    def __init__(self, config: FretboardConfig) -> None:
        """Initialize the fretboard with the given configuration.

        Args:
            config: The fretboard configuration to use.
        """
        super().__init__(config)
        self._mapper = create_chan_mapper(config)
        self._tracker = NoteTracker(self._mapper)
        self._tuner = create_tuner(config)
        self._handler = create_handler(config)

    def _clamp_velocity(self, velocity: int) -> int:
        """Clamp a velocity value to the configured minimum.

        Args:
            velocity: The input velocity (0-127).

        Returns:
            The clamped velocity, with 0 passed through unchanged
            and other values raised to at least the minimum.
        """
        if velocity == 0:
            return 0
        else:
            return max(velocity, self._config.min_velocity)

    def get_note(self, str_pos: StringPos) -> Optional[int]:
        """Get the MIDI note number for a string position.

        Args:
            str_pos: The string position to query.

        Returns:
            The MIDI note number or None if the position is invalid.
        """
        return self._tuner.get_note(str_pos)

    def get_vis(self, str_pos: StringPos) -> VisState:
        """Get the visual state for a string position.

        Args:
            str_pos: The string position to query.

        Returns:
            The current visual state of this position.
        """
        return self._tracker.get_vis(str_pos)

    def trigger(self, str_pos: StringPos, velocity: int) -> NoteEffects:
        """Trigger a note at the specified string position.

        This is the main entry point for note triggering. It coordinates
        the tuner, channel mapper, note handler, and note tracker to
        produce appropriate MIDI output and visual effects.

        Args:
            str_pos: The string position to trigger.
            velocity: The velocity (0-127) for the note. 0 means note-off.

        Returns:
            NoteEffects containing the visual changes and MIDI messages
            resulting from this trigger, or empty effects if the position
            is disabled or invalid.
        """
        if self._tracker.is_enabled(str_pos):
            note_group = self._tuner.get_note_group(str_pos)
            channel = self._mapper.map_channel(str_pos)
            if note_group is not None and channel is not None:
                velocity = self._clamp_velocity(velocity)
                # Debug: Log the channel being used
                import logging

                logging.debug(
                    f"Fretboard.trigger: Using channel {channel} (will send as {channel - 1})"
                )
                fret_msg = FretboardMessage(
                    str_pos=str_pos,
                    equivs=note_group.equivs,
                    msg=FrozenMessage(
                        type="note_on",
                        channel=channel - 1,  # Convert from 1-16 to 0-15 for mido
                        note=note_group.note,
                        velocity=velocity,
                    ),
                )
                out_msgs = self._handler.trigger(fret_msg)
                return self._tracker.record_fx(out_msgs)
        return NoteEffects.empty()

    def handle_mapped_config(self, config: FretboardConfig) -> NoteEffects:
        """Handle a configuration change for the fretboard.

        When the configuration changes, this method cleans up the current
        state and reinitializes all the fretboard components with the
        new configuration.

        Args:
            config: The new fretboard configuration to apply.

        Returns:
            NoteEffects containing cleanup effects from the old state.

        Note:
            Currently the cleanup functionality is not implemented,
            so this will raise an exception. The intent is to send
            note-offs for all active notes and reset visual states.
        """
        fx = self._tracker.clean_fx()
        self._mapper = create_chan_mapper(config)
        self._tracker = NoteTracker(self._mapper)
        self._tuner = create_tuner(config)
        self._handler = create_handler(config)
        return fx
