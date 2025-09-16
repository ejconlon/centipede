"""Port manager for handling dynamic MIDI output port switching.

This module provides functionality to enumerate available MIDI output ports,
manage the active output port, and handle port switching while keeping the
default 'pushpluck' port always available.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import mido

from pushpluck.midi import MidiOutput


class PortManager:
    """Manages MIDI output ports and handles dynamic port switching.

    The PortManager maintains a collection of open MIDI output ports and
    allows switching between them dynamically. It ensures the 'pushpluck'
    port is always available and handles opening/closing ports as needed.
    """

    def __init__(self, default_port_name: str = "pushpluck"):
        """Initialize the port manager.

        Args:
            default_port_name: The name of the default port to keep open.
        """
        self._default_port_name = default_port_name
        self._ports: Dict[str, MidiOutput] = {}
        self._active_port_name = default_port_name

        # Create the default virtual port once and keep it for the program lifetime
        try:
            default_port = MidiOutput.open(default_port_name, virtual=True)
            self._ports[default_port_name] = default_port
            logging.info(f"Created virtual MIDI output port: {default_port_name}")
        except Exception as e:
            logging.error(
                f"Failed to create default MIDI output port {default_port_name}: {e}"
            )
            raise

    def get_available_port_names(self) -> List[str]:
        """Get a list of available MIDI output port names.

        Returns:
            List of available MIDI output port names, with the default port first.
        """
        try:
            # Get system MIDI output ports
            system_ports = mido.get_output_names()

            # Always include the default port at the beginning
            available_ports = [self._default_port_name]

            # Add system ports that aren't already in the list
            for port in system_ports:
                if port not in available_ports:
                    available_ports.append(port)

            return available_ports
        except Exception as e:
            logging.warning(f"Failed to enumerate MIDI ports: {e}")
            return [self._default_port_name]

    def get_active_port_name(self) -> str:
        """Get the name of the currently active port.

        Returns:
            The name of the active MIDI output port.
        """
        return self._active_port_name

    def get_active_port(self) -> MidiOutput:
        """Get the currently active MIDI output port.

        Returns:
            The active MidiOutput instance.
        """
        return self._ports[self._active_port_name]

    def set_active_port(self, port_name: str) -> bool:
        """Set the active MIDI output port.

        Args:
            port_name: The name of the port to make active.

        Returns:
            True if the port was successfully set as active, False otherwise.
        """
        try:
            # Ensure the requested port is open
            if self._ensure_port_open(port_name):
                self._active_port_name = port_name
                logging.info(f"Switched to MIDI output port: {port_name}")
                return True
            else:
                logging.warning(f"Failed to open MIDI output port: {port_name}")
                return False
        except Exception as e:
            logging.error(f"Error switching to port {port_name}: {e}")
            return False

    def _ensure_port_open(self, port_name: str) -> bool:
        """Ensure a MIDI output port is open.

        Args:
            port_name: The name of the port to open.

        Returns:
            True if the port is now open, False if it failed to open.
        """
        if port_name in self._ports:
            return True

        # Default port should never need to be reopened
        if port_name == self._default_port_name:
            logging.error(f"Default port {port_name} should already be open")
            return False

        try:
            # Non-default ports are system MIDI ports (not virtual)
            midi_output = MidiOutput.open(port_name, virtual=False)
            self._ports[port_name] = midi_output
            logging.info(f"Opened MIDI output port: {port_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to open MIDI output port {port_name}: {e}")
            return False

    def cleanup_unused_ports(self) -> None:
        """Close MIDI output ports that are not the active or default port."""
        ports_to_close = []

        for port_name in self._ports:
            if (
                port_name != self._active_port_name
                and port_name != self._default_port_name
            ):
                ports_to_close.append(port_name)

        for port_name in ports_to_close:
            try:
                self._ports[port_name].close()
                del self._ports[port_name]
                logging.info(f"Closed unused MIDI output port: {port_name}")
            except Exception as e:
                logging.warning(f"Error closing port {port_name}: {e}")

    def close_all(self) -> None:
        """Close all open MIDI output ports."""
        for port_name, port in self._ports.items():
            try:
                port.close()
                logging.info(f"Closed MIDI output port: {port_name}")
            except Exception as e:
                logging.warning(f"Error closing port {port_name}: {e}")

        self._ports.clear()
