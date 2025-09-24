from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, NoReturn, Optional

from bad_actor import System, new_system
from minipat.combinators import (
    BundleStreamLike,
    IntStreamLike,
    SymStreamLike,
    bundle_stream,
    channel_stream,
    combine_all,
    control_stream,
    convert_to_int_stream,
    midinote_stream,
    note_stream,
    notename_stream,
    program_stream,
    sound_stream,
    value_stream,
    velocity_stream,
)
from minipat.kit import DEFAULT_KIT, Kit, add_hit
from minipat.live import LiveSystem, Orbit
from minipat.messages import MidiAttrs, MidiBundle, NoteField, NoteKey, TimedMessage
from minipat.midi import start_midi_live_system
from minipat.pat import Pat, SpeedOp
from minipat.stream import MergeStrat, Stream
from minipat.time import (
    Bpc,
    BpcLike,
    Cps,
    CpsLike,
    CycleArcLike,
    CycleDelta,
    CycleDeltaLike,
    CycleTime,
    CycleTimeLike,
    Numeric,
    Tempo,
    TempoLike,
    mk_bpc,
    mk_cps,
    mk_cycle_arc,
    mk_cycle_delta,
    mk_cycle_time,
    mk_tempo,
    numeric_frac,
)
from spiny import PSeq

# =============================================================================
# Type Aliases
# =============================================================================


type FlowLike = Pat[MidiAttrs] | Stream[MidiAttrs] | Flow
"""Types accepted by Flow methods for MIDI patterns.

Accepts:
- Pat[MidiAttrs]: Pattern of MIDI attributes
- Stream[MidiAttrs]: Pre-constructed MIDI stream
- Flow: Another Flow to use
"""


# =============================================================================
# Stream Conversion Helpers
# =============================================================================


def _convert_to_midi_stream(input_val: FlowLike) -> Stream[MidiAttrs]:
    """Convert various input types to a Stream[MidiAttrs]."""
    if isinstance(input_val, Pat):
        return Stream.pat(input_val)
    elif isinstance(input_val, Stream):
        return input_val
    elif isinstance(input_val, Flow):
        return input_val.stream
    else:
        raise ValueError(f"Unsupported type for FlowLike: {type(input_val)}")


def _apply_transpose(attrs: MidiAttrs, transpose_offset: int) -> MidiAttrs:
    """Apply transposition to MIDI attributes.

    Args:
        attrs: Original MIDI attributes containing note
        transpose_offset: Semitone offset to apply

    Returns:
        Transposed MIDI attributes, or attributes with note removed if out of range
    """
    current_note = attrs.lookup(NoteKey())

    if current_note is None:
        return attrs

    # Apply transposition
    new_note = int(current_note) + transpose_offset

    # If result is outside valid MIDI range, remove the note entirely
    if new_note < 0 or new_note > 127:
        # Return attributes without the note (effectively silencing this event)
        return attrs.remove(NoteKey())

    return attrs.put(NoteKey(), NoteField.mk(new_note))


@dataclass(frozen=True, eq=False)
class Flow:
    """A musical flow containing MIDI events over time.

    Flows are the core building blocks for creating patterns in minipat.
    They can be combined, transformed, and played on orbits.

    Examples:
        # Create flows from patterns
        melody = note("c4 d4 e4 f4")
        rhythm = midinote("36 ~ 42 ~")

        # Combine flows
        combined = melody | rhythm  # parallel
        sequence = melody & rhythm  # sequential

        # Transform flows
        fast_melody = melody * 2    # double speed
        quiet = melody >> vel("64")
    """

    stream: Stream[MidiAttrs]

    @staticmethod
    def silent() -> Flow:
        """Create a silent flow with no events.

        Returns:
            A flow that produces no sound.

        Example:
            silent_beat = Flow.silent()
        """
        return Flow(Stream.silent())

    @staticmethod
    def pure(val: MidiAttrs) -> Flow:
        """Create a flow with a single MIDI event.

        Args:
            val: MIDI attributes for the event.

        Returns:
            A flow containing one event.

        Example:
            single_note = Flow.pure(MidiAttrs(note=60, velocity=100))
        """
        return Flow(Stream.pure(val))

    @staticmethod
    def seqs(*flows: FlowLike) -> Flow:
        """Create a sequential flow that plays flows one after another.

        Args:
            *xs: Flows to play in sequence.

        Returns:
            A flow that plays each input flow sequentially.

        Example:
            sequence = Flow.seqs(note("c4"), note("d4"), note("e4"))
            # Equivalent to: note("c4") & note("d4") & note("e4")
        """
        streams = PSeq.mk(_convert_to_midi_stream(x) for x in flows)
        return Flow(Stream.seq(streams))

    @staticmethod
    def pars(*flows: FlowLike) -> Flow:
        """Create a parallel flow that plays flows simultaneously.

        Args:
            *flows: Flows to play in parallel.

        Returns:
            A flow that plays all input flows at the same time.

        Example:
            chord = Flow.pars(note("c4"), note("e4"), note("g4"))
            # Equivalent to: note("c4") | note("e4") | note("g4")
        """
        streams = PSeq.mk(_convert_to_midi_stream(x) for x in flows)
        return Flow(Stream.par(streams))

    @staticmethod
    def rands(*flows: FlowLike) -> Flow:
        """Create a flow that randomly chooses between input flows.

        Args:
            *flows: Flows to choose from randomly.

        Returns:
            A flow that randomly selects one of the input flows each cycle.

        Example:
            random_notes = Flow.rands(note("c4"), note("d4"), note("e4"))
            # Randomly plays c4, d4, or e4 each cycle
        """
        streams = PSeq.mk(_convert_to_midi_stream(x) for x in flows)
        return Flow(Stream.rand(streams))

    @staticmethod
    def alts(*flows: FlowLike) -> Flow:
        """Create a flow that alternates between input flows.

        Args:
            *flows: Flows to alternate between.

        Returns:
            A flow that cycles through input flows, one per cycle.

        Example:
            alternating = Flow.alts(note("c4"), note("d4"))
            # Plays c4 on cycle 1, d4 on cycle 2, c4 on cycle 3, etc.
            # Equivalent to: note("c4") ^ note("d4")
        """
        streams = PSeq.mk(_convert_to_midi_stream(x) for x in flows)
        return Flow(Stream.alt(streams))

    @staticmethod
    def polys(*flows: FlowLike) -> Flow:
        """Create a polymetric flow with different cycle lengths.

        Args:
            *flows: Flows to play polymetrically.

        Returns:
            A flow where each input flow cycles at its own rate.

        Example:
            poly = Flow.polys(note("c4 d4"), note("e4 f4 g4"))
            # First flow cycles every 2 beats, second every 3 beats
        """
        streams = PSeq.mk(_convert_to_midi_stream(x) for x in flows)
        return Flow(Stream.poly(streams, None))

    @staticmethod
    def polysubs(subdiv: int, *flows: FlowLike) -> Flow:
        """Create a polymetric flow with subdivision.

        Args:
            subdiv: Number of subdivisions per cycle.
            *flows: Flows to play polymetrically.

        Returns:
            A polymetric flow with specified subdivision.

        Example:
            poly_sub = Flow.polysubs(4, note("c4 d4"), note("e4 f4 g4"))
            # Subdivides each cycle into 4 parts
        """
        streams = PSeq.mk(_convert_to_midi_stream(x) for x in flows)
        return Flow(Stream.poly(streams, subdiv))

    @staticmethod
    def combines(*flows: FlowLike) -> Flow:
        """Combine flows by merging their MIDI attributes.

        This includes support for bundle flows, which can contain complete
        MIDI messages or sequences of messages.

        Args:
            *flows: Flows to combine (including bundle flows).

        Returns:
            A flow with merged MIDI attributes from all inputs.

        Example:
            combined = Flow.combines(note("c4 d4"), vel("80 100"))
            # Combines note patterns with velocity patterns
            # Equivalent to: note("c4 d4") >> vel("80 100")

            # Can also combine with bundle flows
            program_msg = ProgramMessage(Channel(0), Program(42))
            with_program = Flow.combines(note("c4"), bundle(program_msg))
        """
        streams = [_convert_to_midi_stream(x) for x in flows]
        return Flow(combine_all(streams))

    @staticmethod
    def pat(pattern: Pat[MidiAttrs]) -> Flow:
        """Create a flow from a pattern object.

        Args:
            pattern: A pattern object containing MIDI attributes.

        Returns:
            A flow that plays the pattern.

        Example:
            # Usually used internally; prefer note(), vel() etc.
            flow = Flow.pat(some_pattern)
        """
        return Flow(Stream.pat(pattern))

    def euc(self, hits: int, steps: int, rotation: int = 0) -> Flow:
        """Apply Euclidean rhythm to this flow.

        Args:
            hits: Number of hits (active beats).
            steps: Total number of steps in the rhythm.
            rotation: Number of steps to rotate the pattern (default: 0).

        Returns:
            A flow with Euclidean rhythm applied.

        Example:
            kick = note("c1").euc(3, 8)  # 3 hits over 8 steps
            # Creates rhythm: X..X..X. where X = hit, . = rest
        """
        return Flow(Stream.euc(self.stream, hits, steps, rotation))

    def _speed(self, operator: SpeedOp, factor: Fraction) -> Flow:
        return Flow(Stream.speed(self.stream, operator, factor))

    def fast(self, factor: Numeric) -> Flow:
        """Speed up events by a factor.

        Args:
            factor: Speed multiplier (2 = twice as fast).

        Returns:
            A flow with events sped up.

        Example:
            fast_melody = note("c4 d4 e4 f4").fast(2)
            # Plays twice as fast
            # Equivalent to: note("c4 d4 e4 f4") * 2
        """
        return self._speed(SpeedOp.Fast, numeric_frac(factor))

    def slow(self, factor: Numeric) -> Flow:
        """Slow down events by a factor.

        Args:
            factor: Slowdown factor (2 = half as fast).

        Returns:
            A flow with events slowed down.

        Example:
            slow_melody = note("c4 d4 e4 f4").slow(2)
            # Plays twice as slow
            # Equivalent to: note("c4 d4 e4 f4") / 2
        """
        return self._speed(SpeedOp.Slow, numeric_frac(factor))

    def stretch(self, count: Numeric) -> Flow:
        """Stretch the flow over a longer duration.

        Args:
            count: Stretch factor (2 = twice as long).

        Returns:
            A flow stretched to the specified duration.

        Example:
            stretched = note("c4 d4").stretch(2)
            # Takes 2 cycles instead of 1
        """
        return Flow(Stream.stretch(self.stream, numeric_frac(count)))

    def prob(self, chance: Numeric) -> Flow:
        """Make events in the flow probabilistic.

        Args:
            chance: Probability of each event occurring (0.0-1.0).

        Returns:
            A flow where events occur randomly based on probability.

        Example:
            sparse_hits = note("c1").prob(0.5)
            # Each note has 50% chance of playing
        """
        return Flow(Stream.prob(self.stream, numeric_frac(chance)))

    def repeat(self, count: Numeric) -> Flow:
        """Repeat each event in the flow.

        Args:
            count: Number of times to repeat each event.

        Returns:
            A flow with repeated events.

        Example:
            repeated = note("c4 d4").repeat(3)
            # Becomes: c4 c4 c4 d4 d4 d4
            # Equivalent to: note("c4 d4") ** 3
        """
        return Flow(Stream.repeat(self.stream, numeric_frac(count)))

    def map(self, func: Callable[[MidiAttrs], MidiAttrs]) -> Flow:
        """Transform each event in the flow with a function.

        Args:
            func: Function that takes MidiAttrs and returns modified MidiAttrs.

        Returns:
            A flow with transformed events.

        Example:
            # Transpose all notes up by an octave
            transposed = note("c4 d4").map(lambda m: m.with_note(m.note + 12))
        """
        return Flow(self.stream.map(func))

    def filter(self, predicate: Callable[[MidiAttrs], bool]) -> Flow:
        """Filter events in the flow based on a condition.

        Args:
            predicate: Function that returns True to keep events.

        Returns:
            A flow containing only events that match the predicate.

        Example:
            # Keep only high velocity notes
            loud_only = flow.filter(lambda m: m.velocity > 100)
        """
        return Flow(self.stream.filter(predicate))

    def bind(self, merge_strat: MergeStrat, func: Callable[[MidiAttrs], Flow]) -> Flow:
        """Bind a flow with a merge strategy."""

        def stream_func(x: MidiAttrs) -> Stream[MidiAttrs]:
            return func(x).stream

        return Flow(self.stream.bind(merge_strat, stream_func))

    def apply(
        self,
        merge_strat: MergeStrat,
        func: Callable[[MidiAttrs, MidiAttrs], MidiAttrs],
        other: Flow,
    ) -> Flow:
        """Apply a function across two flows."""
        return Flow(self.stream.apply(merge_strat, func, other.stream))

    def shift(self, delta: CycleDelta) -> Flow:
        """Shift all events in time by a specified amount.

        Args:
            delta: Time offset to apply (positive = later, negative = earlier).

        Returns:
            A flow with time-shifted events.

        Example:
            delayed = note("c4 d4").shift(0.25)  # Quarter beat later
        """
        return Flow(self.stream.shift(delta))

    def early(self, delta: CycleDelta) -> Flow:
        """Shift events earlier in time.

        Args:
            delta: Amount to shift earlier (positive value).

        Returns:
            A flow with events shifted earlier.

        Example:
            early_beat = note("c1").early(0.1)  # Play 0.1 beats early
        """
        return self.shift(CycleDelta(-delta))

    def late(self, delta: CycleDelta) -> Flow:
        """Shift events later in time.

        Args:
            delta: Amount to shift later (positive value).

        Returns:
            A flow with events shifted later.

        Example:
            late_beat = note("c1").late(0.1)  # Play 0.1 beats late
        """
        return self.shift(delta)

    def transpose(self, transpose_input: IntStreamLike) -> Flow:
        """Transpose notes by semitones specified in a pattern.

        Args:
            transpose_input: Pattern string, list of integers, or stream containing
                           semitone transposition values (integers).

        Returns:
            A flow with transposed notes.

        Example:
            melody = note("c4 d4 e4 f4")
            transposed = melody.transpose("12")     # Up one octave
            varying = melody.transpose("0 5 7")     # Different transposition per note
            # Various patterns possible
        """
        transpose_stream = convert_to_int_stream(transpose_input)

        return Flow(
            self.stream.apply(MergeStrat.Inner, _apply_transpose, transpose_stream)
        )

    def par(self, other: FlowLike) -> Flow:
        """Combine this flow with another in parallel."""
        return Flow.pars(self, other)

    def seq(self, other: FlowLike) -> Flow:
        """Combine this flow with another sequentially."""
        return Flow.seqs(self, other)

    def combine(self, other: FlowLike) -> Flow:
        """Combine this flow with another by merging attributes."""
        return Flow.combines(self, other)

    def __or__(self, other: FlowLike) -> Flow:
        """Operator overload for parallel combination."""
        return self.par(other)

    def __and__(self, other: FlowLike) -> Flow:
        """Operator overload for sequential combination."""
        return self.seq(other)

    def __rshift__(self, other: FlowLike) -> Flow:
        """Operator overload for combining flows.

        The >> operator combines flows by merging their MIDI attributes.
        This works with all flow types including bundle flows.

        Examples:
            note("c4 d4") >> vel("80 100")  # Combine notes with velocities
            note("c4") >> bundle(program_msg)  # Combine note with program change
        """
        return Flow.combines(self, other)

    def __lshift__(self, other: FlowLike) -> Flow:
        """Operator overload for combining flows (flipped)."""
        return Flow.combines(other, self)

    def __mul__(self, factor: Numeric) -> Flow:
        """Operator overload for fast (speed up by factor)."""
        return self.fast(factor)

    def __truediv__(self, factor: Numeric) -> Flow:
        """Operator overload for slow (slow down by factor)."""
        return self.slow(factor)

    def __xor__(self, other: FlowLike) -> Flow:
        """Operator overload for alternating flows."""
        return Flow.alts(self, other)

    def __pow__(self, count: Numeric) -> Flow:
        """Operator overload for repeating flow."""
        return self.repeat(count)

    @staticmethod
    def compose(*sections: tuple[CycleArcLike, FlowLike]) -> Flow:
        """Create a flow that composes multiple sections with infinite looping.

        The composition loops infinitely - if the sections span (0, 4), then
        querying (4, 8) returns the same pattern shifted by 4, (-4, 0) same, etc.

        Args:
            *sections: Variable number of (arc, flow) pairs defining the composition

        Returns:
            A flow containing the infinitely looping composition

        Examples:
            # Create a simple verse-chorus structure that loops forever
            verse = note("c4 d4 e4 f4")
            chorus = note("g4 a4 b4 c5")
            song = Flow.compose(
                (arc(0, 2), verse),
                (arc(2, 4), chorus)
            )
        """
        # Convert arc sections to CycleArc sections
        cycle_sections = []
        for arc_like, flow_like in sections:
            cycle_arc = mk_cycle_arc(arc_like)
            cycle_sections.append((cycle_arc, _convert_to_midi_stream(flow_like)))

        return Flow(Stream.compose(cycle_sections))


def notename(input_val: SymStreamLike) -> Flow:
    """Create a flow from note names.

    Args:
        input_val: Pattern string or stream containing
                  musical note names with octaves

    Returns:
        A Flow containing MIDI note attributes

    Examples:
        note("c4 d4 e4")         # C major scale fragment
        note("c4 ~ g4")          # C4, rest, G4
        note("[c4,e4,g4]")       # C major chord (simultaneous)
    """
    return Flow(notename_stream(input_val))


def note(input_val: SymStreamLike) -> Flow:
    """Create a flow from notes with optional chord symbols.

    Args:
        input_val: Pattern string or stream containing notes and chord symbols

    Returns:
        A Flow containing simultaneous MIDI note attributes for each chord

    Examples:
        note("c4'maj7 f4'min")      # C major 7th, F minor
        note("c4'maj7 ~ f4'min")    # C major 7th, rest, F minor
        note("[c4'maj7,f4'min]")    # Layered chords (simultaneous)
    """
    return Flow(note_stream(input_val))


def midinote(input_val: IntStreamLike) -> Flow:
    """Create a flow from numeric MIDI notes.

    Args:
        input_val: Pattern string or stream containing
                  numeric MIDI note values (0-127)

    Returns:
        A Flow containing MIDI note attributes

    Examples:
        midinote("60 62 64")     # C4, D4, E4 (C major triad)
        midinote("36 ~ 42")      # Kick, rest, snare pattern
        midinote("[60,64,67]")   # C major chord (simultaneous)
    """
    return Flow(midinote_stream(input_val))


def vel(input_val: IntStreamLike) -> Flow:
    """Create a flow from velocity values.

    Args:
        input_val: Pattern string or stream containing
                  MIDI velocity values (0-127)

    Returns:
        A Flow containing MIDI velocity attributes

    Examples:
        vel("64 80 100")         # Medium, loud, very loud
        vel("127 0 64")          # Loud, silent, medium
        vel("100*8")             # Repeat loud velocity 8 times
    """
    return Flow(velocity_stream(input_val))


def program(input_val: IntStreamLike) -> Flow:
    """Create a flow from program values.

    Args:
        input_val: Pattern string or stream containing
                  MIDI program values (0-127)

    Returns:
        A Flow containing MIDI program attributes

    Examples:
        program("0 1 40")         # Piano, Bright Piano, Violin
        program("128 ~ 0")        # Invalid program, rest, Piano (will error on 128)
        program("1*4")            # Repeat Bright Piano 4 times
    """
    return Flow(program_stream(input_val))


def control(input_val: IntStreamLike) -> Flow:
    """Create a flow from control number values.

    Args:
        input_val: Pattern string or stream containing
                  MIDI control numbers (0-127)

    Returns:
        A Flow containing MIDI control number attributes

    Examples:
        control("1 7 10")         # Modulation, Volume, Pan
        control("64 ~ 1")         # Sustain, rest, Modulation
        control("7*8")            # Repeat Volume control 8 times
    """
    return Flow(control_stream(input_val))


def value(input_val: IntStreamLike) -> Flow:
    """Create a flow from control value values.

    Args:
        input_val: Pattern string or stream containing
                  MIDI control values (0-127)

    Returns:
        A Flow containing MIDI control value attributes

    Examples:
        value("0 64 127")         # Min, center, max values
        value("127 ~ 0")          # Max, rest, min
        value("64*8")             # Repeat center value 8 times
    """
    return Flow(value_stream(input_val))


def channel(input_val: IntStreamLike) -> Flow:
    """Create a flow from channel values.

    Creates a Flow from MIDI channel values. If channel is not specified
    in patterns, the orbit number will be used as the default channel.

    Args:
        input_val: Pattern string or stream containing
                  MIDI channel values (0-15)

    Returns:
        A Flow containing MIDI channel attributes

    Examples:
        channel("0 1 9")          # Channels 1, 2, 10 (drums)
        channel("15 ~ 0")         # Channel 16, rest, Channel 1
        channel("9*4")            # Repeat Channel 10 (drums) 4 times
    """
    return Flow(channel_stream(input_val))


def bundle(input_val: BundleStreamLike) -> Flow:
    """Create a flow from a MIDI message bundle pattern or stream.

    Takes a Pat or Stream of MidiBundle values and creates a flow
    that adds each bundle to the MIDI attributes. Bundles allow you
    to group multiple MIDI messages together to be sent as a unit.

    Args:
        input_val: A Pat[MidiBundle] or Stream[MidiBundle] containing
                   one or more MidiMessage objects per bundle

    Returns:
        A Flow containing the bundle attribute that can be combined
        with other flows or played on an orbit

    Examples:
        from minipat.messages import NoteOnMessage, ProgramMessage, Channel, Note, Velocity, Program
        from minipat.pat import Pat
        from spiny.seq import PSeq

        # Single message bundle in a pattern
        note_msg = NoteOnMessage(Channel(0), Note(60), Velocity(100))
        bundle_pat = Pat.pure(note_msg)
        flow = bundle(bundle_pat)

        # Multiple message bundle in a pattern
        note_msg = NoteOnMessage(Channel(0), Note(60), Velocity(100))
        program_msg = ProgramMessage(Channel(0), Program(42))
        multi_bundle = PSeq.mk([note_msg, program_msg])
        bundle_pat = Pat.pure(multi_bundle)
        flow = bundle(bundle_pat)

        # Combine with other flows
        combined = note("c4 d4") >> bundle(bundle_pat)
    """
    return Flow(bundle_stream(input_val))


@dataclass(eq=False)
class Nucleus:
    """The core control system for minipat.

    The Nucleus manages the entire minipat system, including timing, playback,
    orbit management, and all global state for minipat (including drum kits).
    Create one with Nucleus.boot() and use it to control your live coding session.

    Properties:
        running: True if the system is running
        playing: Get/set playback state (True = playing, False = paused)
        cps: Get/set cycles per second (tempo control)
        tempo: Get/set beats per minute (alternative tempo control)
        cycle: Get/set current cycle position
        bpc: Get/set beats per cycle
        kit: Get/set the current kit

    Examples:
        # Boot the system
        n = Nucleus.boot()

        # Control playback
        n.playing = True
        n.tempo = 140  # Set to 140 BPM

        # Play patterns on orbits
        n[0] = note("c4 d4 e4 f4")  # Orbit 0
        n[1] = n.sound("bd ~ sd ~")     # Orbit 1 with drums

        # Manage kit
        n.add_hit("crash2", 49)  # Add custom hit

        # Control orbits
        n[0].mute()      # Mute orbit 0
        n[1].solo()      # Solo orbit 1

        # Stop everything and exit
        n.panic()        # Emergency stop
        n.exit()         # Shutdown
    """

    sys: System
    live: LiveSystem[MidiAttrs, TimedMessage]
    kit: Kit

    @staticmethod
    def boot(
        port_name: Optional[str] = None,
        sys_name: Optional[str] = None,
        init_bpm: Optional[int] = None,
        init_bpc: Optional[int] = None,
        log_path: Optional[str] = None,
        log_level: Optional[str] = None,
    ) -> Nucleus:
        log_path = log_path or os.environ.get("MINIPAT_LOG_PATH", "/tmp/minipat.log")
        assert log_path is not None
        log_level = log_level or os.environ.get("MINIPAT_LOG_LEVEL", "INFO")
        assert log_level is not None
        port_name = port_name or os.environ.get("MINIPAT_PORT", "minipat")
        assert port_name is not None
        sys_name = sys_name or os.environ.get("MINIPAT_SYS", "system")
        assert sys_name is not None
        init_bpm = init_bpm or int(os.environ.get("MINIPAT_BPM", "120"))
        assert init_bpm is not None
        init_bpc = init_bpc or int(os.environ.get("MINIPAT_BPC", "4"))
        assert init_bpc is not None
        init_cps = Cps(Fraction(init_bpm, init_bpc * 60))
        logging.basicConfig(
            filename=log_path, filemode="w", level=getattr(logging, log_level)
        )
        sys = new_system(sys_name)
        live = start_midi_live_system(sys, port_name, init_cps, mk_bpc(init_bpc))
        return Nucleus(sys, live, DEFAULT_KIT)

    def stop(self) -> int:
        self.live.panic()
        self.sys.stop()
        saved_excs = self.sys.wait()
        if len(saved_excs) > 0:
            sys.stderr.write(f"System exceptions ({len(saved_excs)})\n")
            for exc in saved_excs:
                sys.stderr.write(f"{exc}\n")
        return len(saved_excs)

    def exit(self) -> NoReturn:
        ret = self.stop()
        sys.exit(ret)

    def panic(self) -> None:
        """Emergency stop - pause, reset cycle, and clear all patterns.

        This immediately stops all sounds, resets the cycle position to 0,
        and clears all orbit patterns. The system remains running.
        """
        self.live.panic()

    def clear(self) -> None:
        """Clear all orbit patterns.

        Removes all patterns from all orbits but keeps them playing.
        Use this to clear everything and start fresh.
        """
        self.live.clear_orbits()

    @property
    def running(self) -> bool:
        """Check if the actor system is currently running.

        Returns:
            True if the system is running, False if it has stopped or is stopping.
        """
        return self.sys.running()

    @property
    def playing(self) -> bool:
        """True if patterns are currently playing, False if paused.

        Examples:
            n.playing = True     # Start playback
            n.playing = False    # Pause playback
            if n.playing: print("Music is playing")
        """
        return self.live.playing()

    @playing.setter
    def playing(self, value: bool) -> None:
        """Set the playing state (True to play, False to pause)."""
        self.live.play(value)

    @property
    def cps(self) -> Cps:
        """Cycles per second (tempo control).

        This is the fundamental tempo unit in minipat. Higher values = faster.
        Prefer using .tempo for BPM-based control.

        Examples:
            n.cps = 2.0      # 2 cycles per second
            current = n.cps  # Get current CPS
        """
        return self.live.get_cps()

    @cps.setter
    def cps(self, value: CpsLike) -> None:
        """Set the cycles per second (tempo)."""
        self.live.set_cps(mk_cps(value))

    @property
    def cycle(self) -> CycleTime:
        """Current cycle position.

        Cycles are the fundamental time unit. Use this to jump to specific
        positions or reset the timeline.

        Examples:
            n.cycle = 0      # Reset to beginning
            n.cycle = 4.5    # Jump to middle of cycle 5
            pos = n.cycle    # Get current position
        """
        return self.live.get_cycle()

    @cycle.setter
    def cycle(self, value: CycleTimeLike) -> None:
        """Set the current cycle position."""
        self.live.set_cycle(mk_cycle_time(value))

    @property
    def bpc(self) -> Bpc:
        """Beats per cycle.

        Defines how many beats fit in one cycle. This affects how tempo
        relates to BPM: BPM = CPS * BPC * 60.

        Examples:
            n.bpc = 4        # 4/4 time
            n.bpc = 3        # 3/4 time
            beats = n.bpc    # Get current BPC
        """
        return self.live.get_bpc()

    @bpc.setter
    def bpc(self, value: BpcLike) -> None:
        """Set the beats per cycle."""
        self.live.set_bpc(mk_bpc(value))

    @property
    def tempo(self) -> Tempo:
        """Tempo in beats per minute (BPM).

        This is the most familiar tempo control. Setting this adjusts CPS
        while keeping beats per cycle constant.

        Examples:
            n.tempo = 120    # Standard tempo
            n.tempo = 140    # Faster tempo
            bpm = n.tempo    # Get current BPM
        """
        return Tempo(Fraction(self.cps) * self.bpc * 60)

    @tempo.setter
    def tempo(self, value: TempoLike) -> None:
        """Set the tempo in beats per minute, adjusting cps while keeping bpc fixed."""
        tempo_val = mk_tempo(value)
        new_cps = Fraction(tempo_val) / (self.bpc * 60)
        self.cps = new_cps

    def once(
        self,
        flow: FlowLike,
        length: Optional[CycleDeltaLike] = None,
        aligned: Optional[bool] = None,
    ) -> None:
        cycle_length = mk_cycle_delta(length) if length is not None else None
        self.live.once(
            _convert_to_midi_stream(flow),
            length=cycle_length,
            aligned=aligned,
            orbit=None,
        )

    def bundle(
        self,
        b: MidiBundle,
        aligned: Optional[bool] = None,
    ) -> None:
        """Send a MIDI bundle immediately using once.

        Takes a MidiBundle (single MidiMessage or sequence of MidiMessages)
        and sends it immediately using the once method.

        Args:
            b: A MidiBundle containing one or more MidiMessage objects
            aligned: Whether to align to cycle boundaries (default: True)

        Examples:
            from minipat.messages import NoteOnMessage, ProgramMessage, Channel, Note, Velocity, Program
            from spiny.seq import PSeq

            # Send a single message
            note_msg = NoteOnMessage(Channel(0), Note(60), Velocity(100))
            n.bundle(note_msg)

            # Send multiple messages as a bundle
            note_msg = NoteOnMessage(Channel(0), Note(60), Velocity(100))
            program_msg = ProgramMessage(Channel(0), Program(42))
            multi_bundle = PSeq.mk([note_msg, program_msg])
            n.bundle(multi_bundle)
        """
        self.once(bundle_stream(Stream.pure(b)), aligned=aligned)

    def orbital(self, num: int) -> Orbital:
        return Orbital(self, Orbit(num))

    def __getitem__(self, num: int) -> Orbital:
        return self.orbital(num)

    def __setitem__(self, num: int, flow: Optional[FlowLike]) -> None:
        o = self.orbital(num)
        if flow is None:
            o.clear()
        else:
            o.every(flow)

    def __delitem__(self, num: int) -> None:
        self.orbital(num).clear()

    def __or__(self, flow: FlowLike) -> None:
        self.once(flow)

    def add_hit(
        self,
        identifier: str,
        note: int,
        velocity: Optional[int] = None,
        channel: Optional[int] = None,
    ) -> None:
        """Add a new hit to the current kit.

        Creates a Sound object with validation and adds it to the kit.

        Args:
            identifier: String identifier for the hit
            note: MIDI note number (0-127)
            velocity: Optional default velocity (0-127)
            channel: Optional default channel (0-15)

        Raises:
            ValueError: If note, velocity, or channel values are out of range

        Example:
            n.add_hit("crash2", 49, 100)
            n[0] = n.sound("bd crash2 sd")
        """
        sound = add_hit(note, velocity, channel)
        self.kit = self.kit.put(identifier, sound)

    def sound(self, input_val: SymStreamLike) -> Flow:
        """Create a flow from kit hit identifiers using this nucleus's kit.

        Creates a Flow from hit identifiers using this nucleus's kit mapping.
        Supports all standard drum notation like "bd" for bass drum, "sd" for snare, etc.

        Args:
            input_val: Pattern string (or Stream) containing hit identifiers

        Returns:
            A Flow containing MIDI note attributes for hits

        Examples:
            n.sound("bd sd bd sd")       # Bass drum, snare, bass drum, snare
            n.sound("bd ~ sd ~")         # Bass drum, rest, snare, rest
            n.sound("[bd,sd,hh]")        # Bass drum + snare + hi-hat (simultaneous)
            n.sound("hh*8")              # Hi-hat repeated 8 times
            n.sound("bd sd:2 hh:3")      # Different speeds for each element
        """
        return Flow(sound_stream(self.kit, input_val))

    def preview(self, arc: CycleArcLike) -> None:
        """Render and play a preview of the current orbits over the given arc.

        Sends the current orbit patterns over the specified arc directly to the
        MIDI output port.

        Args:
            arc: The time arc to preview. Can be either a numeric arc (using numeric times)
                 or a CycleArc (using cycle times). Use arc(start, end) to create one.

        Returns:
            An Event that will be set when playback completes

        Examples:
            # Preview 2 cycles of the current pattern
            n[0] = note("c4 d4 e4 f4")
            n.preview(arc(0, 2))

            # Preview with multiple orbits
            n[0] = note("c4 e4 g4")
            n[1] = midinote("36 42")
            n.preview(arc(0, 4))
        """
        arc = mk_cycle_arc(arc)
        self.playing = False
        self.live.preview(arc)


@dataclass(frozen=True, eq=False)
class Orbital:
    """A single orbit (channel) for playing patterns.

    Orbits are independent channels that can each play one pattern.
    Access them through the Nucleus using n[orbit_number].

    Methods:
        once(flow): Play a flow once
        every(flow): Set the repeating pattern for this orbit
        mute(): Mute this orbit
        unmute(): Unmute this orbit
        solo(): Solo this orbit (mute all others)
        unsolo(): Unsolo this orbit
        clear(): Remove the pattern from this orbit

    Examples:
        n[0].every(note("c4 d4 e4 f4"))    # Set repeating pattern
        n[0].once(note("c5"))             # Play once
        n[0].mute()                       # Mute orbit
        n[0].solo()                       # Solo orbit
        n[0].clear()                      # Clear pattern
    """

    nucleus: Nucleus
    num: Orbit

    def once(
        self,
        flow: FlowLike,
        length: Optional[CycleDeltaLike] = None,
        aligned: Optional[bool] = None,
    ) -> None:
        """Play a flow once on this orbit.

        Args:
            flow: The flow to play.
            length: Duration in cycles (default: flow's natural length).
            aligned: Whether to align to cycle boundaries (default: True).

        Example:
            n[0].once(note("c4 d4 e4"))  # Play melody once
            n[0] | note("c5")            # Shorthand using | operator
        """
        cycle_length = mk_cycle_delta(length) if length is not None else None
        self.nucleus.live.once(
            _convert_to_midi_stream(flow),
            length=cycle_length,
            aligned=aligned,
            orbit=self.num,
        )

    def every(self, flow: FlowLike) -> None:
        """Set the repeating pattern for this orbit.

        Args:
            flow: The flow to repeat continuously.

        Example:
            n[0].every(note("c4 d4 e4 f4"))  # Repeating melody
            n[0] = note("c4 d4 e4 f4")       # Shorthand using assignment
        """
        self.nucleus.live.set_orbit(self.num, _convert_to_midi_stream(flow))

    def mute(self, value: bool = True) -> None:
        """Mute this orbit.

        Args:
            value: True to mute, False to unmute (default: True).

        Example:
            n[0].mute()        # Mute orbit 0
            n[0].mute(False)   # Unmute orbit 0
        """
        self.nucleus.live.mute(self.num, value)

    def unmute(self) -> None:
        """Unmute this orbit.

        Example:
            n[0].unmute()  # Make orbit 0 audible again
        """
        self.nucleus.live.mute(self.num, False)

    def solo(self, value: bool = True) -> None:
        """Solo this orbit (mute all others).

        Args:
            value: True to solo, False to unsolo (default: True).

        Example:
            n[0].solo()        # Solo orbit 0 (mute others)
            n[0].solo(False)   # Unsolo orbit 0
        """
        self.nucleus.live.solo(self.num, value)

    def unsolo(self) -> None:
        """Unsolo this orbit.

        Example:
            n[0].unsolo()  # Remove solo from orbit 0
        """
        self.nucleus.live.solo(self.num, False)

    def clear(self) -> None:
        """Remove the pattern from this orbit.

        The orbit will become silent but remain available for new patterns.

        Example:
            n[0].clear()  # Stop orbit 0
            del n[0]      # Shorthand using del
        """
        self.nucleus.live.set_orbit(self.num, None)

    def __matmul__(self, flow: FlowLike) -> None:
        self.once(flow)

    def __or__(self, flow: FlowLike) -> None:
        self.once(flow)
