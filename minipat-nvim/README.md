# minipat-nvim

A Neovim plugin for minipat (Python version)

Based on [tidal.nvim](https://github.com/ryleelyman/tidal.nvim) by Rylee Alanza Lyman,
which is MIT-licensed.

To use this, add it to your `lazy.nvim` plugins:

    {
      dir = 'minipat-nvim',
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
- `:MpStop` - Stop minipat playback (stops all sounds)
- `:MpAt <code>` - Send Python code directly to the running minipat instance
- `:MpMon [args]` - Monitor MIDI messages (lists devices if no args, otherwise passes args to minipat.mon)
- `:MpMod` - Monitor the configured minipat MIDI port
- `:MpHelp` - Show help with current keybindings and configuration

Commands are customizable via the `command_prefix` config option.

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
