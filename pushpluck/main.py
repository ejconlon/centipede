"""Main entry point for the PushPluck application.

This module contains the main function and command-line argument handling
for the PushPluck guitar fretboard application. It sets up logging,
connects to MIDI ports, and runs the main event loop.
"""

import logging
from argparse import ArgumentParser

from pushpluck import constants
from pushpluck.config import default_scheme, init_config
from pushpluck.menu import default_menu_layout
from pushpluck.plucked import Plucked
from pushpluck.push import PushOutput, PushPorts, match_event, push_ports_context
from pushpluck.shadow import PushShadow


def main_with_ports(ports: PushPorts, min_velocity: int) -> None:
    """Run the main application with the given MIDI ports.

    This function initializes the application components, sets up the controller,
    and runs the main event loop that processes MIDI input from the Push controller.

    Args:
        ports: The MIDI port connections to use.
        min_velocity: The minimum MIDI velocity for note output.
    """
    scheme = default_scheme()
    layout = default_menu_layout()
    config = init_config(min_velocity)
    push = PushOutput(ports.midi_out)
    shadow = PushShadow(push)
    # Start with a clean slate
    logging.info("resetting push")
    push.reset()
    try:
        plucked = Plucked(shadow, ports.midi_processed, scheme, layout, config)
        logging.info("resetting controller")
        plucked.reset()
        logging.info("controller ready")
        while True:
            msg = ports.midi_in.recv_msg()
            event = match_event(msg)
            if event is not None:
                plucked.handle_event(event)
    except KeyboardInterrupt:
        pass
    finally:
        # Send all notes off
        logging.info("final all notes off")
        ports.midi_processed.reset()
        # End with a clean slate
        logging.info("final reset of push")
        push.reset()


def make_parser() -> ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        An ArgumentParser configured with all the command-line options
        for the PushPluck application.
    """
    parser = ArgumentParser()
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--push-delay", type=float, default=constants.DEFAULT_PUSH_DELAY
    )
    parser.add_argument("--push-port", default=constants.DEFAULT_PUSH_PORT_NAME)
    parser.add_argument(
        "--processed-port", default=constants.DEFAULT_PROCESSED_PORT_NAME
    )
    parser.add_argument("--min-velocity", type=int, default=20)
    return parser


def configure_logging(log_level: str) -> None:
    """Configure the logging system with the specified log level.

    Args:
        log_level: The logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
        level=log_level,
    )


def main() -> None:
    """Main entry point for the PushPluck application.

    Parses command-line arguments, configures logging, opens MIDI ports,
    and starts the main application loop.
    """
    parser = make_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    with push_ports_context(
        push_port_name=args.push_port,
        processed_port_name=args.processed_port,
        delay=args.push_delay,
    ) as ports:
        main_with_ports(ports, args.min_velocity)
    logging.info("done")


if __name__ == "__main__":
    main()
