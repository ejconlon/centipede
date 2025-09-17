# minipat-nvim

A Neovim plugin for minipat (Python version)

Based on [tidal.nvim](https://github.com/ryleelyman/tidal.nvim) by Rylee Alanza Lyman,
which is MIT-licensed.

To use this, add it to your `lazy.nvim` plugins:

    {
      dir = 'path/to/minipat-nvim',
      lazy = true,
      ft = { 'minipat' },
      dependencies = {
        'nvim-treesitter/nvim-treesitter',
      },
      init = function()
        vim.filetype.add { extension = { minipat = 'minipat' } }
      end,
      opts = {
        config = {
          -- source_path: path to the minipat source directory
          -- nil (default): uses parent directory of plugin
          -- relative path: relative to the plugin directory
          -- absolute path: use as-is
          source_path = nil,
          -- command_prefix: prefix for vim commands (default "Mp")
          -- creates commands like :MpBoot and :MpQuit
          command_prefix = 'Mp',
          -- autoclose: whether to close the REPL buffer when minipat exits
          -- true (default): close buffer automatically
          -- false: preserve buffer for reviewing output
          autoclose = true,
          -- nucleus_var: name of the nucleus variable in minipat
          -- used for the exit command (e.g., "n.exit()")
          nucleus_var = 'n',
          -- exit_wait: milliseconds to wait for graceful exit (default 1000)
          -- after sending exit command, before forcing termination
          exit_wait = 1000,
          -- minipat: process-specific options
          minipat = {
            -- port: MIDI port name (sets MINIPAT_PORT environment variable, default: "minipat")
            port = "minipat",
          },
        }
      }
    },

The plugin will be lazy-loaded when you open a `*.minipat` file.

## Commands

- `:MpBoot` - Boot the minipat REPL
- `:MpQuit` - Quit the minipat instance (exits the REPL)
- `:MpStop` - Panic minipat (pause, reset cycle, clear patterns)
- `:MpToggle` - Toggle playback (toggles n.playing property)
- `:MpAt <code>` - Send Python code to minipat (boots minipat if not running)
- `:MpMon` - Toggle MIDI monitor for all ports (no arguments)
- `:MpMod` - Monitor the configured minipat MIDI port
- `:MpLogs` - Toggle log viewer with tail -f behavior
- `:MpHide` - Hide all minipat buffers (REPL, monitor, logs)
- `:MpShow` - Show all minipat buffers
- `:MpConfig` - Edit the minipat configuration file
- `:MpHelp` - Show help with current keybindings and configuration
- `:MpBackendStart` - Start the backend process
- `:MpBackendStop` - Stop the backend process
- `:MpBackendRestart` - Restart the backend process
- `:MpBackendStatus` - Show backend process status

Commands are customizable via the `command_prefix` config option.

## Key Mappings

### Global Key Mappings

By default, global keymaps are available under `<localleader>n`:

- `<localleader>nb` - Boot minipat REPL
- `<localleader>nq` - Quit minipat instance
- `<localleader>np` - Panic (stop playback)
- `<localleader>nk` - Toggle playback (n.playing)
- `<localleader>nm` - Toggle MIDI monitor
- `<localleader>nl` - Toggle log viewer
- `<localleader>nh` - Hide all minipat buffers
- `<localleader>nw` - Show all minipat buffers
- `<localleader>nc` - Edit configuration
- `<localleader>n?` - Show help
- `<localleader>nn` - Send code to minipat (MpAt command)

Global keymaps can be customized via the `global_keymaps` config option.

### Buffer-Specific Key Mappings

In `*.minipat` files:

- `<C-L>` - Send the current line/selection to minipat
- `<C-H>` - Panic (pause, reset cycle, clear patterns)

Buffer keymaps can be customized via the `keymaps` config option.

## Configuration Options

The plugin offers extensive configuration options:

### Basic Configuration

- `command_prefix` (default: "Mp") - Prefix for all commands
- `autoclose` (default: true) - Auto-close REPL buffer when minipat exits
- `nucleus_var` (default: "n") - Name of the nucleus variable in minipat
- `exit_wait` (default: 1000) - Milliseconds to wait for graceful exit

### Source Path

- `source_path` (default: nil) - Path to minipat source directory
  - nil: uses parent directory of plugin
  - relative path: relative to plugin directory
  - absolute path: used as-is

### Process Configuration

The `minipat` table contains process-specific options:

- `port` (default: "minipat") - MIDI port name
- `log_path` (default: "/tmp/minipat.log") - Path to minipat log file
- `log_level` (default: "INFO") - Python logging level
- `bpm` (default: 120) - Initial tempo in beats per minute
- `bpc` (default: 4) - Initial beats per cycle

These map to environment variables:
- `minipat.port` → `MINIPAT_PORT`
- `minipat.log_path` → `MINIPAT_LOG_PATH`
- `minipat.log_level` → `MINIPAT_LOG_LEVEL`
- `minipat.bpm` → `MINIPAT_BPM`
- `minipat.bpc` → `MINIPAT_BPC`

### Backend Configuration

The `backend` table allows you to configure a custom backend process that runs alongside minipat:

```lua
backend = {
  enabled = true,                              -- Enable custom backend process
  command = "my-backend --port 8080",          -- Command to run (required if enabled)
  cwd = nil,                                   -- Working directory (defaults to source_path)
  env = { MY_VAR = "value" },                  -- Additional environment variables
  pidfile = "/tmp/my-backend.pid",             -- Path to pidfile (defaults to /tmp/minipat-backend.pid)
  autostart = true,                            -- Start backend when plugin loads
  autostop = true,                             -- Stop backend when Neovim exits
  restart_on_exit = false,                     -- Restart backend if it exits unexpectedly
}
```

#### Backend Configuration Options

- `enabled` (default: false) - Enable/disable the backend functionality
- `command` (required if enabled) - Shell command to start the backend process
- `cwd` (default: source_path) - Working directory for the backend process
  - Relative paths are resolved relative to `source_path`
  - Absolute paths are used as-is
- `env` (default: {}) - Additional environment variables to pass to the backend
- `pidfile` (default: "/tmp/minipat-backend.pid") - Path to store the backend process ID
  - Relative paths are resolved relative to `cwd`
  - Used for process tracking and cleanup
- `autostart` (default: true) - Automatically start the backend when the plugin loads
- `autostop` (default: true) - Automatically stop the backend when Neovim exits
- `restart_on_exit` (default: false) - Restart the backend if it exits unexpectedly (non-zero exit code)

#### Backend Process Management

The plugin provides commands to manage the backend process lifecycle:

- `:MpBackendStart` - Start the backend (if not already running)
- `:MpBackendStop` - Stop the backend process
- `:MpBackendRestart` - Stop and restart the backend
- `:MpBackendStatus` - Check if the backend is running

The backend process is tracked using pidfiles and can be managed independently of the minipat REPL.

### Key Mappings Configuration

#### Buffer Keymaps (`keymaps` table)

- `send_line` (default: "<C-L>") - Send current line to minipat
- `send_visual` (default: "<C-L>") - Send selection to minipat (visual mode)
- `panic` (default: "<C-H>") - Panic (pause, reset cycle, clear patterns)

#### Global Keymaps (`global_keymaps` table)

- `leader_prefix` (default: "<localleader>n") - Prefix for global keymaps
- `boot` (default: "b") - Boot minipat
- `quit` (default: "q") - Quit minipat
- `panic` (default: "p") - Panic (stop playback)
- `toggle` (default: "k") - Toggle playback (n.playing)
- `monitor` (default: "m") - Toggle MIDI monitor
- `logs` (default: "l") - Toggle log viewer
- `hide` (default: "h") - Hide minipat buffers
- `show` (default: "w") - Show minipat buffers
- `config` (default: "c") - Edit configuration
- `help` (default: "?") - Show help
- `at` (default: "n") - Send code to minipat (MpAt)

## Source Path and Virtual Environment

The plugin needs to know where your minipat source code is located:

1. **Default behavior** (`source_path = nil`): Uses the parent directory of the plugin
   (assumes plugin is inside the minipat repository).

2. **Custom source path**: Set `source_path` to either:
   - A relative path (relative to the plugin directory)
   - An absolute path to your minipat source directory

The plugin expects a Python virtual environment at `source_path/.venv`. If it doesn't
exist but a `pyproject.toml` is found, it will attempt to create the venv using `uv sync`.

**Manual installation**: If not using `uv`, install minipat with:

    pip install -e /path/to/centipede

Check the source (`lua/minipat.lua`) for additional configurable options and defaults.
