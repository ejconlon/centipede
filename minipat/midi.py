"""MIDI functionality for the minipat pattern system.

This module provides both high-level pattern-based MIDI functionality and low-level
MIDI message handling utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from logging import Logger
from threading import Event
from typing import (
    Any,
    List,
    Optional,
    cast,
    override,
)

import mido
from mido.frozen import FrozenMessage, freeze_message

from bad_actor import (
    Actor,
    ActorEnv,
    Callback,
    Mutex,
    Nursery,
    Sender,
    System,
    Task,
    new_system,
)
from minipat.common import PosixTime, current_posix_time
from minipat.ev import EvHeap
from minipat.live import (
    BackendEvents,
    BackendMessage,
    BackendPlay,
    BackendTiming,
    Instant,
    LiveSystem,
    Orbit,
    Processor,
    Timing,
)
from minipat.messages import (
    DEFAULT_VELOCITY,
    Channel,
    ChannelField,
    ChannelKey,
    MidiAttrs,
    MidiMessage,
    MsgHeap,
    MsgTypeField,
    NoteField,
    TimedMessage,
    Velocity,
    msg_note_off,
)
from spiny.seq import PSeq


@dataclass
class SenderState:
    """State shared between MIDI backend actor and sender task."""

    timing: Timing
    msg_heap: MsgHeap

    @staticmethod
    def initial(timing: Timing) -> SenderState:
        return SenderState(timing, MsgHeap.empty())


class MidiSenderTask(Task):
    """Background task that sends scheduled MIDI messages at the correct time.

    Runs in a separate thread and continuously pops messages from a shared
    heap, sending them to a MIDI output at the
    appropriate timestamps. Uses sleep intervals based on current tempo and generation rate.
    """

    def __init__(
        self,
        state: Mutex[SenderState],
        output: mido.ports.BaseOutput,
    ):
        """Initialize the MIDI sender task.

        Args:
            state: Shared state for scheduled messages
            output: MIDI output port
        """
        self._state = state
        self._output = output

    @override
    def run(self, logger: Logger, halt: Event) -> None:
        """Execute the task using the actor system's threading model.

        Args:
            logger: Logger for the task to use.
            halt: Event that will be set when the task should stop.
        """
        logger.debug("MIDI sender task started")
        while True:
            current_time = current_posix_time()

            # Get the next message that's ready to send
            with self._state as st:
                sleep_interval = st.timing.sleep_interval
                msgs = st.msg_heap.pop_all_before(current_time)

            if msgs:
                for timed_msg in msgs:
                    self._output.send(timed_msg.message)

            # Sleep based on current timing configuration
            # Use halt.wait() instead of sleep() to be responsive to shutdown
            if halt.wait(timeout=sleep_interval):
                break

        logger.debug("MIDI sender task stopped")


# =============================================================================
# Pattern System Integration
# =============================================================================


def parse_messages(
    orbit: Optional[Orbit],
    attrs: MidiAttrs,
    default_velocity: Optional[Velocity] = None,
) -> List[FrozenMessage]:
    """Parse MIDI attributes into multiple FrozenMessages

    Converts a set of MIDI attributes into multiple MIDI message types using
    the higher-level typed message system. The function extracts all valid
    message types present in the attributes:

    - If note is present: creates a note_on message (note required, velocity optional)
    - If program is present: creates a program_change message (program required only)
    - If control attributes are present: creates a control_change message (both control_num and control_val required)

    This allows mixing different message types in the same attributes, e.g.:
    note('c4 c5') >> program('0 1') would create both note and program change messages.

    Args:
        orbit: The orbit number, used as MIDI channel (must be 0-15), or None if not specified
        attrs: MIDI attributes containing the message parameters
        default_velocity: Default velocity to use when not specified in attributes

    Returns:
        A list of FrozenMessage objects representing the parsed MIDI messages

    Raises:
        ValueError: If the orbit is outside valid MIDI channel range (0-15),
                   or if neither orbit nor channel attribute is provided,
                   or if incomplete control change attributes are present
    """
    channel = attrs.lookup(ChannelKey())
    if channel is None and orbit is not None:
        orbit_value = int(orbit)
        if not (0 <= orbit_value <= 15):
            raise ValueError(
                f"Orbit {orbit_value} out of valid MIDI channel range (0-15)"
            )
        attrs = attrs.put(ChannelKey(), Channel(orbit_value))

    # Get typed messages and render them to FrozenMessage objects
    typed_messages = MidiMessage.parse_attrs(attrs)
    velocity = default_velocity if default_velocity is not None else DEFAULT_VELOCITY
    return [msg.render_midi(velocity) for msg in typed_messages]


# =============================================================================
# MIDI Processor (Pattern System to MIDI Messages)
# =============================================================================


class MidiProcessor(Processor[MidiAttrs, TimedMessage]):
    """Processor that converts MidiAttrs to MIDI messages."""

    def __init__(self, default_velocity: Optional[Velocity] = None):
        """Initialize the MIDI processor.

        Args:
            default_velocity: Default velocity to use when not specified
        """
        self._default_velocity = (
            default_velocity if default_velocity is not None else DEFAULT_VELOCITY
        )

    @override
    def process(
        self, instant: Instant, orbit: Optional[Orbit], events: EvHeap[MidiAttrs]
    ) -> PSeq[TimedMessage]:
        """Process MIDI events into timed MIDI messages."""
        timed_messages = []

        for span, ev in events:
            # Parse messages using parse_messages (supports multiple message types)
            try:
                msgs = parse_messages(orbit, ev.val, self._default_velocity)
            except ValueError as e:
                # Log the error and skip this event
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    "Skipping MIDI event due to parse error at cycle %s: %s",
                    instant.cycle_time,
                    e,
                )
                continue

            # Determine event timing
            event_start = span.whole is None or span.active.start == span.whole.start
            event_end = span.whole is None or span.active.end == span.whole.end

            # Process each message (already FrozenMessage objects)
            for midi_msg in msgs:
                msg_type = MsgTypeField.get(midi_msg)
                if msg_type == "note_on":
                    # For note messages, we need to handle timing for note_on/note_off pairs
                    note = NoteField.get(midi_msg)
                    channel = ChannelField.get(midi_msg)

                    if event_start:
                        # Calculate timestamp for the start of the event
                        timestamp = PosixTime(
                            instant.posix_start
                            + (float(span.active.start) / float(instant.cps))
                        )
                        # Debug: Log instant values and calculation
                        # with open("midi_debug.log", "a") as f:
                        #     f.write(f"Note Event: span.start={float(span.active.start):.6f} span.end={float(span.active.end):.6f} instant.cps={float(instant.cps):.6f} posix_start={instant.posix_start:.6f} calculated_timestamp={timestamp:.6f}\n")
                        timed_messages.append(TimedMessage(timestamp, midi_msg))

                    if event_end:
                        # Create note off message (at end of span)
                        # Use whole arc end if present, otherwise active arc end
                        note_end_time = (
                            span.whole.end
                            if span.whole is not None
                            else span.active.end
                        )
                        note_off_timestamp = PosixTime(
                            instant.posix_start
                            + (float(note_end_time) / float(instant.cps))
                        )
                        note_off_msg = msg_note_off(channel=channel, note=note)
                        timed_messages.append(
                            TimedMessage(note_off_timestamp, note_off_msg)
                        )

                else:
                    # For non-note messages (program_change, control_change), send only at event start
                    if event_start:
                        timestamp = PosixTime(
                            instant.posix_start
                            + (float(span.active.start) / float(instant.cps))
                        )
                        timed_messages.append(TimedMessage(timestamp, midi_msg))

        return PSeq.mk(timed_messages)


# =============================================================================
# MIDI Actor (Message Output)
# =============================================================================


class MidiBackendActor(Actor[BackendMessage[TimedMessage]]):
    """Actor that queues MIDI messages for scheduled sending.

    Instead of sending MIDI messages immediately, this actor queues them
    in a shared message heap where a MidiSenderTask will send them at the
    appropriate time. Also handles timing configuration updates.
    """

    def __init__(
        self,
        state: Mutex[SenderState],
        output: mido.ports.BaseOutput,
    ):
        """Initialize the MIDI backend actor.

        Args:
            state: Shared state for scheduled messages
            output: MIDI output port
        """
        self._state = state
        self._output = output
        self._playing = False
        self._reset = False

    @override
    def on_stop(self, logger: Logger) -> None:
        """Reset the MIDI output when stopping."""
        logger.debug("Resetting MIDI output port on stop")

        if not self._reset:
            self._output.reset()

    @override
    def on_message(self, env: ActorEnv, value: BackendMessage[TimedMessage]) -> None:
        self._reset = False
        match value:
            case BackendPlay(playing):
                self._playing = playing
                if playing:
                    env.logger.info("MIDI: Playing")
                else:
                    env.logger.info(
                        "MIDI: Pausing - clearing queued messages and resetting port"
                    )
                    # Clear buffer
                    with self._state as st:
                        st.msg_heap.clear()
                    # Reset to send "Reset All Controllers" and "All Notes Off"
                    self._output.reset()
                    # Track that the last thing we did was reset
                    self._reset = True
            case BackendEvents(msgs):
                if self._playing:
                    env.logger.debug("MIDI: Pushing %d messages", len(msgs))
                    with self._state as st:
                        for msg in msgs:
                            st.msg_heap.push(msg)
                else:
                    env.logger.debug("MIDI: Ignoring events while stopped")
            case BackendTiming(timing):
                with self._state as st:
                    st.timing = timing
                env.logger.debug(
                    "MIDI: Updated timing config - CPS: %s, Gens/Cycle: %d, Wait Factor: %s",
                    timing.cps,
                    timing.generations_per_cycle,
                    timing.wait_factor,
                )
            case _:
                env.logger.warning("Unknown MIDI message type: %s", type(value))


# =============================================================================
# Echo system
# =============================================================================


class SendActor(Actor[FrozenMessage]):
    """Actor that sends raw MIDI messages to an output port."""

    def __init__(self, port: mido.ports.BaseOutput):
        self._port = port

    @override
    def on_message(self, env: ActorEnv, value: FrozenMessage) -> None:
        self._port.send(value)

    @override
    def on_stop(self, logger: Logger) -> None:
        self._port.close()


def _recv_cb(sender: Sender[FrozenMessage], msg: Any) -> None:
    """Callback for receiving MIDI messages."""
    fmsg = cast(FrozenMessage, freeze_message(msg))
    sender.send(fmsg)


class RecvCallback(Callback[FrozenMessage]):
    """Callback for receiving MIDI messages from an input port."""

    def __init__(self, port: mido.ports.BaseInput):
        self._port = port

    @override
    def register(self, sender: Sender[FrozenMessage]) -> None:
        self._port.callback = partial(_recv_cb, sender)  # pyright: ignore

    @override
    def deregister(self) -> None:
        self._port.callback = None  # pyright: ignore


def echo_system(in_port_name: str, out_port_name: str) -> System:
    """Create a system that echoes MIDI input to output."""
    system = new_system("echo")
    in_port = mido.open_input(name=in_port_name, virtual=True)  # pyright: ignore
    out_port = mido.open_output(name=out_port_name, virtual=True)  # pyright: ignore
    recv_callback = RecvCallback(in_port)
    send_actor = SendActor(out_port)
    system.spawn_callback("recv", send_actor, recv_callback)
    return system


# =============================================================================
# Live system
# =============================================================================


class MidiNursery(Nursery[BackendMessage[TimedMessage]]):
    """Nursery for MIDI backend components."""

    def __init__(self, timing: Timing, output: mido.ports.BaseOutput) -> None:
        self._timing = timing
        self._output = output

    def initialize(self, env: ActorEnv) -> Sender[BackendMessage[TimedMessage]]:
        """Initialize MIDI backend actor and sender task."""
        state = Mutex(SenderState.initial(self._timing))

        # Create the timing-oblivious MIDI backend actor
        midi_backend = MidiBackendActor(state, self._output)
        backend_sender = env.control.spawn_actor("midi_backend", midi_backend)

        # Create the sender task with timing awareness
        sender_task = MidiSenderTask(state, self._output)
        env.control.spawn_task("midi_sender", sender_task)

        return backend_sender


def start_midi_live_system(
    system: System,
    out_port_name: str,
    cps: Optional[Fraction] = None,
    beats_per_cycle: Optional[int] = None,
) -> LiveSystem[MidiAttrs, TimedMessage]:
    """Start a LiveSystem with MIDI components.

    Creates a complete MIDI live system with:
    - MidiProcessor for converting pattern events to MIDI messages
    - MidiBackendActor for queuing messages and handling timing updates
    - MidiSenderTask for sending scheduled messages with timing-aware sleep intervals
    - Shared message heap and timing configuration for coordination

    Args:
        system: The actor system to use
        out_port_name: Name of the MIDI output port
        cps: Optional initial cycles per second (default ~120 BPM)
        beats_per_cycle: Optional beats per cycle (default 4)

    Returns:
        A started LiveSystem configured for MIDI output.
    """
    # Create MIDI output port and protect with mutex
    existing = mido.get_output_names()  # pyright: ignore
    virtual = out_port_name not in existing
    output = mido.open_output(name=out_port_name, virtual=virtual)  # pyright: ignore

    # Create shared timing configuration
    timing = Timing.initial(cps, beats_per_cycle)

    backend_sender = system.spawn_nursery("midi_output", MidiNursery(timing, output))

    # Create MIDI processor
    processor = MidiProcessor()

    # Start the live system with MIDI processor and backend
    return LiveSystem.start(system, processor, backend_sender, cps)
