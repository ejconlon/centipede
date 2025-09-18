# Minipat DSL Cheatsheet

Minipat is a Python-based live coding language for creating musical patterns.
This cheatsheet covers the essential DSL syntax and functionality.

## Basic Setup

```python
# `n` is a `Nucleus` instance
# Use dir() to see what else is in scope

n.tempo = 120  # Set tempo in BPM

# Play patterns on orbits
n[0] = note("c4 d4 e4 f4")  # Orbit 0
n[1] = kit("bd ~ sd ~")     # Orbit 1 with drums
```

## Flow Creation Functions

### `note(pattern)` - Musical Notes
```python
note("c4 d4 e4")         # C major scale fragment
note("c4 ~ g4")          # C4, rest, G4
note("[c4,e4,g4]")       # C major chord (simultaneous)
note("c4:2 d4:0")        # Sample selection (c4 sample 2, d4 sample 0)
```

### `midinote(pattern)` - MIDI Note Numbers
```python
midinote("60 62 64")     # C4, D4, E4 (C major triad)
midinote("36 ~ 42")      # Kick, rest, snare pattern
midinote("[60,64,67]")   # C major chord (simultaneous)
```

### `kit(pattern)` - Drum Sounds
```python
# Use nucleus method for better control
n.kit("bd sd bd sd")     # Bass drum, snare, bass drum, snare
n.kit("bd ~ sd ~")       # Bass drum, rest, snare, rest
n.kit("[bd,sd,hh]")      # Bass drum + snare + hi-hat (simultaneous)
n.kit("hh*8")            # Hi-hat repeated 8 times
```

### `vel(pattern)` - Velocity
```python
vel("64 80 100")         # Medium, loud, very loud
vel("127 0 64")          # Loud, silent, medium
vel("100*8")             # Repeat loud velocity 8 times
```

### `program(pattern)` - MIDI Program Change
```python
program("0 1 40")        # Piano, Bright Piano, Violin
program("1*4")           # Repeat Bright Piano 4 times
```

### `control(pattern)` & `value(pattern)` - MIDI Control
```python
control("1 7 10")        # Modulation, Volume, Pan
value("0 64 127")        # Min, center, max values
```

### `channel(pattern)` - MIDI Channel
```python
channel("0 1 9")         # Channels 1, 2, 10 (drums)
channel("9*4")           # Repeat Channel 10 (drums) 4 times
```

## Pattern Syntax

### Basic Elements
```python
"bd"                     # Single symbol
"~"                      # Silence/rest
"bd sd hh"               # Sequence (space-separated)
"bd~sd"                  # Sequence without spaces
"bd:2"                   # Sample selection (symbol:selector)
```

### Grouping
```python
"[bd sd] cp"             # Sequential group [a b] then c
"[bd,sd,cp]"             # Parallel (simultaneous) [a,b,c]
"[bd|sd|cp]"             # Random choice [a|b|c]
"<bd sd cp>"             # Alternating <a b c> (cycles through)
"{bd, sd}"               # Polymetric {a, b} (different cycle lengths)
```

### Timing Modifiers
```python
"bd*2"                   # Fast (double speed) - multiply operator
"bd/2"                   # Slow (half speed) - divide operator
"bd*2/4"                 # Chained: fast then slow
"bd_"                    # Stretch by 2 (underscore)
"bd__"                   # Stretch by 3 (multiple underscores)
"bd@3"                   # Stretch by 3 (@ notation)
"bd _ _"                 # Stretch by 3 (spaced underscores)
```

### Euclidean Rhythms
```python
"bd(3,8)"                # 3 hits over 8 steps
"bd(3,8,1)"              # 3 hits over 8 steps, rotated by 1
```

### Probability
```python
"bd?"                    # 50% chance
"bd?0.8"                 # 80% chance
```

### Complex Patterns
```python
"bd*2 [sd cp] ~ {hh, oh}"         # Mixed timing and grouping
"[[bd sd] cp] hh"                 # Nested groups
"bd(3,8) sd*2 [hh,oh]?"           # Euclidean + speed + probability
```

## Flow Operators and Methods

### Combination Operators
```python
flow1 | flow2            # Parallel (same as Flow.pars())
flow1 & flow2            # Sequential (same as Flow.seqs())
flow1 >> flow2           # Combine (merge attributes)
flow1 ^ flow2            # Alternating (same as Flow.alts())
```

### Speed Operators
```python
flow * 2                 # Fast (double speed)
flow / 2                 # Slow (half speed)
flow ** 3                # Repeat 3 times
```

### Transformation Methods
```python
flow.fast(2)             # Speed up by factor of 2
flow.slow(2)             # Slow down by factor of 2
flow.stretch(2)          # Stretch over 2 cycles
flow.repeat(3)           # Repeat each event 3 times
flow.prob(0.5)           # 50% probability for each event
flow.shift(0.25)         # Shift by quarter beat
flow.early(0.1)          # Play 0.1 beats early
flow.late(0.1)           # Play 0.1 beats late
```

### Euclidean Method
```python
flow.euc(3, 8)           # Apply 3 hits over 8 steps
flow.euc(3, 8, 2)        # With rotation of 2
```

### Functional Operations
```python
# Transform each event
flow.map(lambda m: m.with_note(m.note + 12))  # Transpose up octave

# Filter events
flow.filter(lambda m: m.velocity > 100)       # Keep only loud notes
```

## Static Flow Constructors

```python
Flow.silent()                   # Silent flow
Flow.pure(MidiAttrs(...))       # Single MIDI event
Flow.seqs(flow1, flow2, flow3)  # Sequential flows
Flow.pars(flow1, flow2, flow3)  # Parallel flows
Flow.alts(flow1, flow2, flow3)  # Alternating flows
Flow.rands(flow1, flow2, flow3) # Random choice flows
Flow.polys(flow1, flow2)        # Polymetric flows
Flow.combines(flow1, flow2)     # Combined flows
```

## Nucleus Control

### Playback Control
```python
n.playing = True        # Start/stop playback
n.tempo = 140           # Set BPM
n.cps = 2.0             # Set cycles per second
n.bpc = 4               # Set beats per cycle
n.cycle = 0             # Reset/jump to cycle position
```

### Emergency Controls
```python
n.panic()               # Emergency stop (pause, reset, clear)
n.clear()               # Clear all orbit patterns
n.stop()                # Stop system
n.exit()                # Stop and exit
```

### Orbit Management
```python
n[0] = flow             # Set repeating pattern on orbit 0
n[0] | flow             # Play flow once on orbit 0
n[0].once(flow)         # Play flow once
n[0].every(flow)        # Set repeating pattern
n[0].mute()             # Mute orbit
n[0].solo()             # Solo orbit (mute others)
n[0].clear()            # Clear orbit pattern
del n[0]                # Clear orbit (shorthand)
```

### Drum Kit Management
```python
n.add_drum_sound("crash2", 49)      # Add new drum sound
n.remove_drum_sound("crash2")       # Remove drum sound
sounds = n.list_drum_sounds()       # List all sounds
n.reset_kit()                       # Reset to default kit
n.drum_kit = custom_kit             # Set custom kit
```

## Combining Patterns

### Common Combinations
```python
# Melody with rhythm
melody = note("c4 d4 e4 f4")
rhythm = n.kit("bd ~ sd ~")
combined = melody >> rhythm

# Chord progressions
bass = note("c2 f2 g2 c2")
chords = note("[c4,e4,g4] [f4,a4,c5] [g4,b4,d5] [c4,e4,g4]")
progression = bass | chords

# Layered drums
kick = n.kit("bd ~ ~ ~")
snare = n.kit("~ ~ sd ~")
hihat = n.kit("hh hh hh hh")
drums = kick | snare | hihat
```

### Performance Patterns
```python
# Build up pattern
n[0] = n.kit("bd")                         # Start with kick
n[0] = n.kit("bd ~ sd ~")                  # Add snare
n[0] = n.kit("bd ~ sd ~") | n.kit("hh*8")  # Add hi-hat

# Variations
base = note("c4 d4 e4 f4")
n[0] = base                           # Basic pattern
n[0] = base.fast(2)                   # Double time
n[0] = base.prob(0.7)                 # Sparse version
n[0] = base >> vel("80 100 60 90")    # Dynamic version
```

## Tips

- Use `~` for rests/silence in patterns
- Combine patterns with `|` (parallel), `&` (sequential), `>>` (merge)
- Use `.prob()` to add randomness and variation
- Use Euclidean rhythms for interesting polyrhythmic patterns
- Layer multiple orbits for complex arrangements
- Use `n.panic()` for emergency stops during live performance
