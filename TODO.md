# Project TODOs

This file tracks project tasks in several lanes:

- Active (a): tasks currently in progress
- Review (r): tasks for review on the way from Active to Done
- Upcoming (u): future Active tasks from highest to lowest priority
- Done (d): finished Review tasks from most to least recent
- Backlog (b): unscheduled Upcoming tasks from most to least recent

Pull new tasks into Active from the Upcoming lane.
Work on tasks in the Active lane.
Put Active tasks into Review to allow manual marking as Done.
Create Backlog tasks to allow manual selection into Upcoming.

Each task has a unique ID in the format {#id} where id is a number.
Tasks can also have optional tags in the format {tag} for categorization or special handling.
Tasks can reference other task IDs in the format {@id} where id is a number.
For example:
- {#1} Implement feature X {feature} {release}
- {#2} Fix bug Y {@1} {bugfix}

We track the next available task ID on its own line below.

## Lanes

Next available task ID: {#43}

### Active


### Review


### Upcoming

- {#40} Implement O(log n) versions of PSeq.take, drop, and split_at operations
  - Current implementations are O(n) using iteration/uncons
  - Need proper finger tree-based splitting that operates on tree structure
  - Should leverage existing finger tree invariants for logarithmic complexity
  - Reference Hinze-Paterson paper for proper splitting algorithms
- {#25} Add `Flow.beat(self, beat_str: str, steps: int) -> Flow` to repeat flow in the given pattern

### Done

- {#42} Engrave event heaps with lilypond, including tab information
  - ✓ Implemented `render_lilypond()` function in minipat/offline.py
  - ✓ Takes EvHeap[MidiAttrs] with tab information and renders to LilyPond format
  - ✓ Supports TabInstKey, TabStringKey, TabFretKey attributes for tablature notation
  - ✓ Handles multiple instruments (guitar, bass, ukulele, mandolin, banjo)
  - ✓ Generates both standard notation and tablature staff
  - ✓ One cycle equals one bar, bpc is numerator of time signature
  - ✓ Outputs .ly files that compile successfully with lilypond binary
  - ✓ All tests and precommit checks pass
- {#41} Add guitar tab parsing
  - **Restructured**: Renamed to comprehensive tablature system supporting multiple instruments
  - Created tab.py module with TabInst enum and TabConfig dataclass system
  - Supports multiple instruments: StandardGuitar, DropDGuitar, StandardBass, Ukulele, Mandolin, Banjo, etc.
  - Added TabInstKey, TabStringKey, TabFretKey MIDI message fields for tab metadata
  - Renamed guit() to tab() and guitar_tab_stream to tab_stream for consistency
  - Created TabElemParser and TabBinder classes that generate full tab attributes
  - Format: "#frets" where frets are continuous digits/x, spaces separate tab entries
  - Examples: "#x32010" (C chord), "#320003" (G chord), "#x32010 #320003" (sequence)
  - Tab attributes include: Note, TabInst, string number (1-based), fret number
  - Configurable string ordering (HIGH_TO_LOW default for guitar)
  - Comprehensive test suite with 20 tests covering all functionality
  - All tests pass and precommit checks succeed
- {#38} Create invN dropN macros to modify chord voicings
  - Implemented chord inversion functionality with `apply_inversion()` function
  - Implemented drop voicing functionality with `apply_drop_voicing()` function
  - Extended ChordElemParser to parse voicing modifiers in chord notation
  - Supports pseudo-grammar: ``note ` chord [` (inv | drop) int ]*``
  - Examples: ``c4`maj7`inv1`` (first inversion), ``c4`maj7`drop2`` (drop2 voicing), ``c4`maj7`inv1`drop2`` (combined)
  - Added comprehensive test suite in test_chord_voicings.py with 11 test cases
- {#39} Unknown chords, notes, or sounds should throw errors in streams
  - Fixed ChordBinder to propagate ValueError instead of silencing unknown chords
  - Verified that unknown notes already raise ValueError
  - Verified that unknown sounds already raise ValueError
  - Added comprehensive test suite in test_stream_errors.py
- {#37} Change chord separator from single quote to backtick
- {#36} Add midi file playing with fluidsynth to minipat/offline.py
  - Implemented play_midi function that saves MIDI file to temp directory
  - Added FluidSynth command execution with platform-specific audio driver selection
  - Included error handling for missing FluidSynth installation and soundfont
- {#35} Add Stream/Flow.opt_apply and clear invalid transposed notes
- {#34} Change Iso -> BiArrow
- {#33} Make strict checks default and rename check to precommit
- {#32} Use new-style dict/list/tuple types
- {#31} Support chord patterns
- {#30} Add message bundle support to MidiAttrs
  - Updated MidiMessage.parse_attrs to handle BundleKey
  - Added bundle_stream function to combinators
  - Added bundle function to DSL
- {#29} Midi message bundles
- {#28} Create patch list of ~/.local/share/minipat/sf/default.sf2 using sf2dump output and add to minipat.hw.fluid
  - Created minipat/hw/fluid.py module with FluidR3_GM preset list
  - Extracted 189 presets using sf2dump with proper (bank, preset, name) format
  - Includes multiple banks: 0 (GM standard), 8 (variations), 9, 16, 128 (drum kits)
  - Added utility functions: get_preset(), find_preset(), list_presets(), search_presets(), get_banks(), get_gm_preset()
  - Follows General MIDI standard: Bank 0, Preset 0 = "Yamaha Grand Piano"
- {#27} Add minipat.fluid to manage fluidsynth config and resources
  - Created minipat/fluid.py module with FluidSynthConfig class
  - Created ~/.local/share/minipat/sf directory for soundfonts
  - Downloaded fluid-soundfont from Ubuntu archive (134MB)
  - Created symlink ~/.local/share/minipat/sf/default.sf2 -> FluidR3_GM.sf2
  - Created config files in ~/.local/config/minipat
- {#24} Expand accepted argument types to Flow methods
  - When accepting pattern strings, also accept Stream[T] or Flow
- {#23} Add `Flow.transpose(self, pat_str: str) -> Flow` to transpose notes
  - Implemented transpose method that takes a pattern string of semitone offset integers
  - Uses custom parser to handle positive and negative integers (e.g., "5 -3 7")
  - Applies transpose offsets using MergeStrat.Inner with the original note stream
  - Silences notes that would go out of valid MIDI range (0-127) by removing the note attribute
  - Added comprehensive tests covering basic transpose, patterns, negative values, mixed values, out-of-range handling, and preservation of other MIDI attributes
- {#26} Fix event generation by making whole required and correcting constructor sites in streams
- {#21} Configure "generations ahead" to give more generation time
  - Added _DEFAULT_GENERATIONS_AHEAD = 2 static configuration in live.py
  - Updated once() method to use generation_cycle_length * _DEFAULT_GENERATIONS_AHEAD for minimum_future_time
  - Updated preview() method to use the same multiplier for consistency
  - This gives 2x more generation time (2 generation cycles ahead instead of 1)
  - All tests pass with the new configuration
- {#22} Add " | delta: [fractional seconds since last column msg]" column to minipat nvim monitor output
  - Modified pretty_print_message() to track and display both global and per-channel delta times
  - Added two delta columns: gdel (global) and cdel (channel-specific)
  - Both deltas display with 3 decimal places (e.g., "gdel: 0.123 | cdel: 1.456")
  - First message shows "gdel: 0.000 | cdel: 0.000"
  - Updated monitor_port() to maintain both last_global_timestamp_ns and channel_timestamps dictionary
- {#20} Add `Nucleus.preview(arc: CycleArcLike) -> Event` to render and play midi file
  - Added preview() method to Nucleus class that renders current orbit patterns
  - Updated render_midi_file() to set type=0 for single track MIDI files
  - Tested functionality with multiple orbits and patterns
- {#18} Render EvHeap to midi file
  - Added render_evheap_to_midi_file() function to midi.py for rendering event heaps to MIDI files
  - Function supports configurable tempo (cps), MIDI resolution (ticks_per_beat), and default velocity
  - Handles note_on/note_off pairs, program changes, and control changes properly
  - Automatically assigns default channel (orbit 0) when no channel is specified in events
  - Tested with both simple and complex compositions including multiple channels and overlapping notes
- {#19} Repeat composition stream, move into stream.py and add to combinators {python}
  - Added Stream.repeat_compose() static method to stream.py for repeating composition structures
  - Added repeat_compose() function to combinators.py that delegates to Stream.repeat_compose()
  - Supports both full and fractional repetitions with proper timing calculations
  - Tested functionality with various repetition counts and verified all existing tests pass
- {#16} Create minipat.compose module for composition primitives {python}
- {#15} Pass only necessary state to window and process functions {nvim}
  - We pass whole application state around, but I would like the window and
    process modules to be more self-contained.
- {#14} Add cheatsheet param to config for custom cheatsheet path (default nil) {nvim}
- {#13} Reflow windows when hiding/showing {nvim}
- {#12} Look for minipat_config.lua in cwd to override default args -> plugin args -> local args {nvim}
- {#11} Add Cheatsheet (c) group to minipat-nvim that opens buffer containing CHEATSHEET.md
- {#10} Create CHEATSHEET.md with examples of minipat dsl
- {#4} Add commands to manage backend pane and process {plugin} {neovim} {commands}
  - Commands to start/stop/restart the backend process
  - Commands to show/hide/toggle the backend output pane
  - Commands to clear backend logs or save them to file
  - Integration with existing plugin command structure
- {#3} Add new minipat pane for backend output streams {plugin} {neovim} {ui}
  - Create a dedicated pane to display stdout/stderr from the backend command
  - Provide real-time monitoring of backend process output
  - Support scrolling and log viewing functionality

### Backlog


