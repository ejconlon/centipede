# Project TODOs

This file tracks project tasks. We are maintaining 4 "lanes" of tasks in this file:

- Active: tasks currently in progress
- Upcoming: future tasks from next priority to last priority
- Done: finished tasks from most recent to least recent
- Archive: unscheduled tasks from most recent to least recent

Work on tasks in the Active lane. Move tasks from one lane to the other as their state changes.

Each task has a unique ID, and we track the next available task ID on its own line below.

## Lanes

Next available task ID: {#10}

### Active

- **Add "kit" functionality for drum sound patterns** {#5} {patterns} {nucleus} {drums}
  - Implement sound pattern mapping from string identifiers to MIDI parameters
  - Support drum kit notation (e.g., "bd" → bass drum, "sd" → snare drum, "hh" → hi-hat)
  - Create well-typed mapping to (channel, note, velocity, duration) tuples
  - Enable pattern sequences using drum sound names instead of raw MIDI values
  - Consider extensible kit definitions for different drum machines/samples

### Upcoming

- **Add optional "backend" config to specify backend command** {#2} {plugin} {neovim} {config}
  - Allow users to configure a custom backend process command
  - Add this process to the plugin lifecycle management
  - Use pidfiles to control execution and track process state
  - Enable integration with external tools and custom backends

- **Add new minipat pane for backend output streams** {#3} {plugin} {neovim} {ui}
  - Create a dedicated pane to display stdout/stderr from the backend command
  - Provide real-time monitoring of backend process output
  - Support scrolling and log viewing functionality

- **Add commands to manage backend pane and process** {#4} {plugin} {neovim} {commands}
  - Commands to start/stop/restart the backend process
  - Commands to show/hide/toggle the backend output pane
  - Commands to clear backend logs or save them to file
  - Integration with existing plugin command structure

- **Performance optimizations for high-frequency MIDI events** {#6} {performance} {midi}

- **Enhanced error handling and recovery mechanisms** {#7} {reliability} {error-handling}

- **Plugin configuration validation and user feedback** {#8} {plugin} {config} {validation}

- **Documentation updates for new features** {#9} {docs}

### Done

- **Investigate timing issues in test_midi_integration** {#1} {testing} {midi} {timing}
  - The test occasionally fails due to timing variations in note durations
  - Consider improving timing tolerance or using mock time for more deterministic tests
  - May need to address system-level timing jitter in MIDI message processing

### Archive
