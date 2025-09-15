"""Main controller class that coordinates all PushPluck components.

This module contains the Plucked class, which serves as the central
controller for the PushPluck application. It coordinates between
the menu system, fretboard/pad interface, and MIDI processing.
"""

import logging

from pushpluck.base import Resettable
from pushpluck.config import ColorScheme, Config
from pushpluck.constants import ButtonCC
from pushpluck.menu import Menu, MenuLayout
from pushpluck.midi import MidiSink
from pushpluck.pads import Pads
from pushpluck.push import ButtonEvent, PadEvent, PushEvent
from pushpluck.shadow import PushShadow


class Plucked(Resettable):
    """Main controller that coordinates menu, pads, and MIDI processing.

    This class serves as the central hub for the PushPluck application,
    managing the interaction between the menu system, the fretboard pads,
    and MIDI output. It processes events from the Push controller and
    coordinates the appropriate responses.
    """

    def __init__(
        self,
        shadow: PushShadow,
        midi_processed: MidiSink,
        scheme: ColorScheme,
        layout: MenuLayout,
        config: Config,
    ) -> None:
        """Initialize the Plucked controller.

        Args:
            shadow: Push display shadow for efficient updates.
            midi_processed: MIDI sink for processed note output.
            scheme: Color scheme for visual elements.
            layout: Menu layout configuration.
            config: Initial application configuration.
        """
        self._shadow = shadow
        self._midi_processed = midi_processed
        self._pads = Pads.construct(scheme, config)
        self._menu = Menu(layout, config)

    def handle_event(self, event: PushEvent) -> None:
        """Handle an event from the Push controller.

        Routes events to the appropriate handler (pads or menu) and
        coordinates any necessary updates between components.

        Args:
            event: The Push controller event to process.
        """
        with self._shadow.context() as push:
            if isinstance(event, PadEvent):
                self._pads.handle_event(push, self._midi_processed, event)
            elif isinstance(event, ButtonEvent) and event.button == ButtonCC.Accent:
                if event.pressed:
                    self.reset()
            elif isinstance(event, ButtonEvent) and event.button == ButtonCC.Master:
                if event.pressed:
                    self.redraw()
            else:
                config = self._menu.handle_event(push, event)
                if config is not None:
                    self._pads.handle_config(
                        push, self._midi_processed, config, reset=False
                    )

    def redraw(self) -> None:
        """Redraw the entire Push interface.

        Resets the shadow state and redraws both the menu and pad displays.
        Called when a complete refresh is needed.
        """
        logging.info("plucked redrawing")
        self._shadow.reset()
        with self._shadow.context() as push:
            self._menu.redraw(push)
            self._pads.redraw(push)

    def reset(self) -> None:
        """Reset the controller to its initial state.

        Resets both the menu and pad configurations and redraws the interface.
        This is typically called on startup or when the Accent button is pressed.
        """
        logging.info("plucked resetting")
        self._shadow.reset()
        with self._shadow.context() as push:
            config = self._menu.handle_reset(push)
            self._pads.handle_config(push, self._midi_processed, config, reset=True)
