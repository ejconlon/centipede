"""MIDI monitor for minipat."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any

import mido


def signal_handler(sig: Any, frame: Any) -> None:
    """Handler for Ctrl-C."""
    sys.exit(0)


def format_message_csv(msg: mido.Message, delimiter: str) -> str:
    """Format a MIDI message as delimited values."""
    timestamp_ns = time.time_ns()
    wall_time = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # millisecond precision

    if msg.type == "note_on":
        return delimiter.join(
            [
                str(timestamp_ns),
                wall_time,
                msg.type,
                str(msg.channel),
                str(msg.note),
                str(msg.velocity),
                "",
                "",
                "",
                "",
            ]
        )
    elif msg.type == "note_off":
        return delimiter.join(
            [
                str(timestamp_ns),
                wall_time,
                msg.type,
                str(msg.channel),
                str(msg.note),
                str(msg.velocity),
                "",
                "",
                "",
                "",
            ]
        )
    elif msg.type == "control_change":
        return delimiter.join(
            [
                str(timestamp_ns),
                wall_time,
                msg.type,
                str(msg.channel),
                "",
                "",
                str(msg.control),
                str(msg.value),
                "",
                "",
            ]
        )
    elif msg.type == "program_change":
        return delimiter.join(
            [
                str(timestamp_ns),
                wall_time,
                msg.type,
                str(msg.channel),
                "",
                "",
                "",
                "",
                str(msg.program),
                "",
            ]
        )
    elif msg.type == "pitchwheel":
        return delimiter.join(
            [
                str(timestamp_ns),
                wall_time,
                msg.type,
                str(msg.channel),
                "",
                "",
                "",
                "",
                "",
                str(msg.pitch),
            ]
        )
    elif msg.type == "aftertouch":
        return delimiter.join(
            [
                str(timestamp_ns),
                wall_time,
                msg.type,
                str(msg.channel),
                "",
                "",
                "",
                str(msg.value),
                "",
                "",
            ]
        )
    elif msg.type == "polytouch":
        return delimiter.join(
            [
                str(timestamp_ns),
                wall_time,
                msg.type,
                str(msg.channel),
                str(msg.note),
                "",
                "",
                str(msg.value),
                "",
                "",
            ]
        )
    elif msg.type == "sysex":
        data_str = " ".join(f"{b:02x}" for b in msg.data[:16])
        if len(msg.data) > 16:
            data_str += "..."
        return delimiter.join(
            [str(timestamp_ns), wall_time, msg.type, "", "", "", "", "", "", data_str]
        )
    else:
        return delimiter.join(
            [str(timestamp_ns), wall_time, msg.type, "", "", "", "", "", "", str(msg)]
        )


def pretty_print_message(
    msg: mido.Message,
    last_global_timestamp_ns: int | None,
    channel_timestamps: dict[int | str, int],
) -> int:
    """Pretty print a MIDI message with both global and per-channel deltas.

    Args:
        msg: The MIDI message to print
        last_global_timestamp_ns: Last timestamp across all channels
        channel_timestamps: Dictionary mapping channel numbers (or "NC") to last timestamp

    Returns:
        Current timestamp in nanoseconds
    """
    timestamp_ns = time.time_ns()
    # Format with millisecond precision: HH:MM:SS.mmm
    wall_time = time.strftime("%H:%M:%S") + f".{(timestamp_ns // 1_000_000) % 1000:03d}"

    # Get channel or "NC" for messages without channel
    channel_key: int | str = msg.channel if hasattr(msg, "channel") else "NC"

    # Calculate global delta
    if last_global_timestamp_ns is not None:
        gdelta_ns = timestamp_ns - last_global_timestamp_ns
        gdelta_secs = gdelta_ns / 1_000_000_000  # Convert to seconds
        gdelta_str = f" | gdel: {gdelta_secs:.3f}"
    else:
        gdelta_str = " | gdel: 0.000"

    # Calculate channel delta
    if channel_key in channel_timestamps:
        cdelta_ns = timestamp_ns - channel_timestamps[channel_key]
        cdelta_secs = cdelta_ns / 1_000_000_000  # Convert to seconds
        cdelta_str = f" | cdel: {cdelta_secs:.3f}"
    else:
        cdelta_str = " | cdel: 0.000"

    # Update timestamp for this channel
    channel_timestamps[channel_key] = timestamp_ns

    # Get channel string - padded to 2 digits if present, NC if not
    channel_str = f"[{msg.channel:02d}]" if hasattr(msg, "channel") else "[NC]"

    # Map message types to abbreviated versions for display
    type_abbrev = {
        "control_change": "control",
        "program_change": "program",
        "pitchwheel": "pitch",
    }
    display_type = type_abbrev.get(msg.type, msg.type)

    if msg.type == "note_on":
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | note: {msg.note:3d} | vel: {msg.velocity:3d}"
        )
    elif msg.type == "note_off":
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | note: {msg.note:3d} | vel: {msg.velocity:3d}"
        )
    elif msg.type == "control_change":
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | control: {msg.control:3d} | value: {msg.value:3d}"
        )
    elif msg.type == "program_change":
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | program: {msg.program:3d}"
        )
    elif msg.type == "pitchwheel":
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | pitch: {msg.pitch:6d}"
        )
    elif msg.type == "aftertouch":
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | value: {msg.value:3d}"
        )
    elif msg.type == "polytouch":
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | note: {msg.note:3d} | value: {msg.value:3d}"
        )
    elif msg.type == "sysex":
        data_str = " ".join(f"{b:02x}" for b in msg.data[:8])
        if len(msg.data) > 8:
            data_str += "..."
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | data: {data_str}"
        )
    else:
        print(
            f"[{wall_time}] {channel_str} {display_type:<9}{gdelta_str}{cdelta_str} | {msg}"
        )

    return timestamp_ns


def list_ports() -> None:
    """List available MIDI input ports."""
    print("\nAvailable MIDI input ports:", file=sys.stderr)
    input_ports = mido.get_input_names()
    for i, name in enumerate(input_ports):
        print(f"  {i:2d}: {name}", file=sys.stderr)
    print(file=sys.stderr)


def find_port(device_id: str) -> str | None:
    """Find a MIDI port by device ID or name."""
    input_ports = mido.get_input_names()

    # Try to parse as integer first
    try:
        port_idx = int(device_id)
        if 0 <= port_idx < len(input_ports):
            return str(input_ports[port_idx])
        else:
            print(
                f"Error: Port ID {port_idx} out of range (0-{len(input_ports) - 1})",
                file=sys.stderr,
            )
            return None
    except ValueError:
        pass

    # Try exact match first
    if device_id in input_ports:
        return device_id

    # Try partial match (starts with)
    for port_name in input_ports:
        if port_name.lower().startswith(device_id.lower()):
            return str(port_name)

    # Try substring match
    for port_name in input_ports:
        if device_id.lower() in port_name.lower():
            return str(port_name)

    print(f"Error: Port '{device_id}' not found", file=sys.stderr)
    return None


def monitor_port(port_name: str, delimiter: str | None) -> None:
    """Monitor a MIDI input port."""
    try:
        print(f"Monitoring MIDI port: {port_name}", file=sys.stderr)
        print("Press Ctrl+C to stop", file=sys.stderr)
        print(file=sys.stderr)

        use_delimited = delimiter is not None

        if use_delimited and delimiter is not None:
            # Output CSV header to stdout
            header = delimiter.join(
                [
                    "timestamp_ns",
                    "wall_time",
                    "type",
                    "channel",
                    "note",
                    "velocity",
                    "control",
                    "value",
                    "program",
                    "data",
                ]
            )
            print(header)

        inport = mido.open_input(port_name)

        with inport:
            last_global_timestamp_ns: int | None = None
            channel_timestamps: dict[int | str, int] = {}
            for msg in inport:
                if use_delimited and delimiter is not None:
                    csv_line = format_message_csv(msg, delimiter)
                    print(csv_line)
                else:
                    last_global_timestamp_ns = pretty_print_message(
                        msg, last_global_timestamp_ns, channel_timestamps
                    )
    except KeyboardInterrupt:
        print("\nStopped monitoring", file=sys.stderr)
    except OSError as e:
        print(f"Error opening port '{port_name}': {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="MIDI monitor for minipat")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available MIDI input ports"
    )
    parser.add_argument(
        "-p",
        "--port",
        metavar="ID",
        help="MIDI port ID (number) or name (string) to monitor",
    )
    parser.add_argument(
        "-d",
        "--delimiter",
        metavar="DELIM",
        help="Delimiter for output values (enables delimited mode)",
    )

    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    if args.port:
        port_name = find_port(args.port)
        if port_name:
            monitor_port(port_name, args.delimiter)
    else:
        parser.print_help(file=sys.stderr)


if __name__ == "__main__":
    main()
