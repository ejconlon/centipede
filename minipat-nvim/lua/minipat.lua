-- ==============================================================================
-- Minipat Neovim Plugin
-- ==============================================================================

local M = {}
local process_manager = require("minipat.process")
local window = require("minipat.window")

-- Component configuration: name -> type mapping
local COMPONENTS = {
  repl = "process",
  backend = "process",
  logs = "process",
  monitor = "process",
  cheatsheet = "buffer"
}

-- Helper function to get component names
local function get_component_names()
  local names = {}
  for name, _ in pairs(COMPONENTS) do
    table.insert(names, name)
  end
  return names
end

-- Helper function to get component type
local function get_component_type(name)
  return COMPONENTS[name]
end

-- ==============================================================================
-- Configuration and Defaults
-- ==============================================================================

local DEFAULTS = {
  config = {
    source_path = nil, -- Optional path to minipat project root
    file_ext = "minipat", -- File extension to trigger this plugin
    split = nil, -- Whether to split vertical (v) or horizontal (h) - nil to autodetect
    command_prefix = "Mp", -- prefix for the Boot, Quit commands, etc
    autoclose = true, -- Close the buffer on exit
    exit_wait = 500, -- milliseconds to wait for process to exit gracefully
    debug = false, -- set to true to see debug messages
    minipat = {
      nucleus_var = "n", -- name of the Nucleus variable in the REPL
      port = "minipat", -- MIDI port name for minipat (sets MINIPAT_PORT env var)
      log_path = "/tmp/minipat.log", -- Log file path for minipat (sets MINIPAT_LOG_PATH env var)
      log_level = "INFO", -- Log level for minipat (sets MINIPAT_LOG_LEVEL env var)
      bpm = 120, -- Initial BPM for minipat (sets MINIPAT_BPM env var)
      bpc = 4, -- Initial beats per cycle for minipat (sets MINIPAT_BPC env var)
    },
    backend = {
      command = nil, -- Custom backend command to run (e.g., "my-backend --port 8080")
      cwd = nil, -- Working directory for backend (defaults to source_path)
      env = {}, -- Additional environment variables for backend
      pidfile = nil, -- Path to pidfile for tracking backend process (defaults to /tmp/minipat-backend.pid)
      autostart = true, -- Start backend automatically when plugin loads
      autostop = true, -- Stop backend when Neovim exits
      restart_on_exit = false, -- Restart backend if it exits unexpectedly
    },
  },
  keymaps = {
    send_line = "<C-L>",
    send_visual = "<C-L>",
    panic = "<C-H>",
  },
  global_keymaps = {
    leader_prefix = "<localleader>n",
    -- REPL commands (<leader>nr prefix)
    repl_hide = "rh", -- <localleader>nrh - toggle show/hide REPL
    repl_start = "rs", -- <localleader>nrs - (re)start REPL
    repl_quit = "rq", -- <localleader>nrq - quit REPL
    repl_status = "ri", -- <localleader>nri - REPL status
    repl_only = "ro", -- <localleader>nro - show only REPL (hide others)
    -- Monitor commands (<leader>nm prefix)
    monitor_hide = "mh", -- <localleader>nmh - toggle show/hide monitor
    monitor_start = "ms", -- <localleader>nms - (re)start monitor
    monitor_quit = "mq", -- <localleader>nmq - quit monitor
    monitor_status = "mi", -- <localleader>nmi - monitor status
    monitor_only = "mo", -- <localleader>nmo - show only monitor (hide others)
    -- Backend commands (<leader>nb prefix)
    backend_hide = "bh", -- <localleader>nbh - toggle show/hide backend
    backend_start = "bs", -- <localleader>nbs - (re)start backend
    backend_quit = "bq", -- <localleader>nbq - quit backend
    backend_status = "bi", -- <localleader>nbi - backend status
    backend_only = "bo", -- <localleader>nbo - show only backend (hide others)
    -- Logs commands (<leader>nl prefix)
    logs_hide = "lh", -- <localleader>nlh - toggle show/hide logs
    logs_start = "ls", -- <localleader>nls - (re)start logs viewer
    logs_quit = "lq", -- <localleader>nlq - quit logs viewer
    logs_status = "li", -- <localleader>nli - logs status
    logs_only = "lo", -- <localleader>nlo - show only logs (hide others)
    -- Cheatsheet commands (<leader>nc prefix)
    cheatsheet_hide = "ch", -- <localleader>nch - toggle show/hide cheatsheet
    cheatsheet_start = "cs", -- <localleader>ncs - (re)start cheatsheet viewer
    cheatsheet_quit = "cq", -- <localleader>ncq - quit cheatsheet viewer
    cheatsheet_status = "ci", -- <localleader>nci - cheatsheet status
    cheatsheet_only = "co", -- <localleader>nco - show only cheatsheet (hide others)
    -- Other commands
    start = "s", -- <localleader>ns - start backend (if configured) and REPL
    quit = "q", -- <localleader>nq - quit all processes
    hide = "h", -- <localleader>nh - toggle show/hide all buffers
    all = "a", -- <localleader>na - show all started components
    info = "i", -- <localleader>ni - show info/status for all components
    panic = "p", -- <localleader>np - panic (stop playback)
    toggle = "k", -- <localleader>nk - toggle playback
    help = "?", -- <localleader>n? - show help
    at = "n", -- <localleader>nn - send code
  },
}

local KEYMAPS = {
  send_line = {
    mode = "n",
    action = "Vy<cmd>lua require('minipat').send_reg()<CR><ESC>",
    description = "send line to Minipat",
  },
  send_visual = {
    mode = "v",
    action = "y<cmd>lua require('minipat').send_reg()<CR>",
    description = "send selection to Minipat",
  },
  panic = {
    mode = "n",
    action = nil, -- Will be set dynamically with config
    description = "send panic command to Minipat",
  },
}

-- ==============================================================================
-- State Management
-- ==============================================================================
local state = {
  -- Subprocesses: each has { buffer, process }
  subprocesses = {
    repl = { buffer = nil, process = nil },
    monitor = { buffer = nil, process = nil },
    backend = { buffer = nil, process = nil },
    logs = { buffer = nil, process = nil },
    cheatsheet = { buffer = nil, process = nil }, -- cheatsheet is buffer-only (process = nil)
  },
  -- Shared state
  resolved_python = nil,
  resolved_source_path = nil,
  -- Buffer group management
  group_hidden = false,
  group_split_dir = "v", -- Direction for the group split
  group_win = nil, -- Main group window
  previously_visible_components = {}, -- Components that were visible before hiding
  -- Configuration cache
  cached_config = nil,
  initial_user_args = nil,
}

-- ==============================================================================
-- Utility Functions
-- ==============================================================================

-- Safe window navigation
local function safe_set_current_win(win)
  if win and vim.api.nvim_win_is_valid(win) then
    vim.api.nvim_set_current_win(win)
    return true
  end
  return false
end

-- File system utilities
local function file_exists(path)
  local stat = vim.loop.fs_stat(path)
  return stat and stat.type == "file"
end

local function dir_exists(path)
  local stat = vim.loop.fs_stat(path)
  return stat and stat.type == "directory"
end

-- Plugin directory utilities
local function get_plugin_dir()
  local source = debug.getinfo(1, "S").source:sub(2)
  local plugin_lua_dir = vim.fn.fnamemodify(source, ":p:h")
  return vim.fn.fnamemodify(plugin_lua_dir, ":h")
end

-- ==============================================================================
-- Window and Buffer Management (using window submodule)
-- ==============================================================================

-- ==============================================================================
-- Process Management
-- ==============================================================================
local function get_subprocess_status(name)
  local subprocess = state.subprocesses[name]
  local component_type = get_component_type(name)

  if component_type == "buffer" then
    -- Buffer-only component, check if buffer exists and is valid
    return (subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer)) and "open" or "closed"
  elseif component_type == "process" and subprocess and subprocess.process then
    local status_info = subprocess.process:status()
    return status_info.is_running and "running" or "stopped"
  end
  return "not started"
end

local function quit_subprocess(name)
  local subprocess = state.subprocesses[name]
  if not subprocess then
    return false
  end

  local component_type = get_component_type(name)

  -- Stop the process (only for process components)
  if component_type == "process" and subprocess.process then
    subprocess.process:stop()
    subprocess.process = nil
  end

  -- Close buffer windows
  if subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
    local wins = vim.fn.win_findbuf(subprocess.buffer)
    for _, win in ipairs(wins) do
      if vim.api.nvim_win_is_valid(win) then
        vim.api.nvim_win_close(win, false)
      end
    end
    -- Delete buffer (for buffer components, just close windows; for process components, force delete)
    if component_type == "buffer" then
      -- For buffer components, just close the windows, the buffer can remain
      subprocess.buffer = nil
    else
      -- For process components, force delete the buffer
      pcall(vim.api.nvim_buf_delete, subprocess.buffer, { force = true })
      subprocess.buffer = nil
    end
  end

  return true
end

local function start_subprocess(name, args, command, cwd, env, on_exit_callback)
  local subprocess = state.subprocesses[name]
  local original_win = vim.api.nvim_get_current_win()

  -- Clean up any existing process for this subprocess
  if subprocess.process then
    quit_subprocess(name)
    vim.wait(100) -- Allow time for cleanup
  end

  -- Set up window layout for the new subprocess buffer
  if window.ensure_group_visible(state, args) then
    -- Group window exists, add a split within it
    local group_buffers = window.get_group_buffers(state)
    if #group_buffers > 0 then
      -- Find any visible window from the group to split from
      local split_from_win = nil
      for _, buf in ipairs(group_buffers) do
        local wins = vim.fn.win_findbuf(buf)
        if #wins > 0 and vim.api.nvim_win_is_valid(wins[1]) then
          split_from_win = wins[1]
          break
        end
      end

      if split_from_win then
        safe_set_current_win(split_from_win)
      else
        safe_set_current_win(state.group_win)
      end

      local split_dir = window.get_split_direction(args.config.split)
      vim.cmd(window.create_split_command(split_dir, true))
    end
  else
    -- No group window, create it
    safe_set_current_win(original_win)
    window.create_group_split(state, args)
  end

  -- Create and display new buffer for subprocess
  subprocess.buffer = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_set_current_buf(subprocess.buffer)

  -- Start the actual process
  subprocess.process = process_manager.new({
    name = name,
    command = command,
    cwd = cwd,
    env = env or {},
    debug = args.config.debug,
    autoclose = args.config.autoclose,
    on_exit = function(job_id, exit_code, event_type)
      -- Clean up references when process exits
      if subprocess.scroll_timer then
        subprocess.scroll_timer:stop()
        subprocess.scroll_timer:close()
        subprocess.scroll_timer = nil
      end
      subprocess.auto_scroll_enabled = nil
      subprocess.user_navigating = nil
      subprocess.buffer = nil
      subprocess.process = nil
      if on_exit_callback then
        on_exit_callback(job_id, exit_code, event_type)
      end
    end,
  })

  subprocess.process:start(subprocess.buffer)

  -- Trigger reflow after starting process
  vim.defer_fn(function()
    window.reflow_windows_in_group(state)
  end, 150)

  -- Set up smart auto-scrolling for this buffer
  if subprocess.buffer then
    -- Track auto-scroll state and user interaction
    subprocess.auto_scroll_enabled = true
    subprocess.user_navigating = false
    subprocess.last_line_count = 0

    local function is_at_bottom(win)
      if not vim.api.nvim_win_is_valid(win) then
        return false
      end

      local line_count = vim.api.nvim_buf_line_count(subprocess.buffer)
      local cursor_pos = vim.api.nvim_win_get_cursor(win)

      -- Only consider at bottom if cursor is on the actual last line
      return cursor_pos[1] >= line_count
    end

    local function scroll_to_bottom()
      local current_line_count = vim.api.nvim_buf_line_count(subprocess.buffer)

      -- Only scroll if new content was added and user isn't navigating
      if current_line_count <= subprocess.last_line_count then
        return
      end
      subprocess.last_line_count = current_line_count

      -- Only auto-scroll the current window if auto-scroll is enabled and user isn't navigating
      local current_win = vim.api.nvim_get_current_win()
      local wins = vim.fn.win_findbuf(subprocess.buffer)

      for _, win in ipairs(wins) do
        if
          vim.api.nvim_win_is_valid(win)
          and win == current_win
          and subprocess.auto_scroll_enabled
          and not subprocess.user_navigating
        then
          vim.api.nvim_win_call(win, function()
            vim.api.nvim_win_set_cursor(win, { current_line_count, 0 })
          end)
        end
      end
    end

    local function on_cursor_moved()
      local current_win = vim.api.nvim_get_current_win()

      -- Mark as user navigating to pause auto-scroll temporarily
      subprocess.user_navigating = true

      -- Check if user moved to bottom, if so re-enable auto-scroll
      if is_at_bottom(current_win) then
        subprocess.auto_scroll_enabled = true
      else
        subprocess.auto_scroll_enabled = false
      end

      -- Clear navigation flag after a short delay
      vim.defer_fn(function()
        subprocess.user_navigating = false
      end, 200)
    end

    -- Set up periodic auto-scrolling with a timer (slower interval)
    subprocess.scroll_timer = vim.loop.new_timer()
    subprocess.scroll_timer:start(
      500,
      500,
      vim.schedule_wrap(function()
        if not vim.api.nvim_buf_is_valid(subprocess.buffer) then
          subprocess.scroll_timer:stop()
          subprocess.scroll_timer:close()
          subprocess.scroll_timer = nil
          return
        end
        scroll_to_bottom()
      end)
    )

    -- Track cursor movement to enable/disable auto-scroll
    vim.api.nvim_create_autocmd({ "CursorMoved", "CursorMovedI" }, {
      buffer = subprocess.buffer,
      callback = on_cursor_moved,
    })

    -- Enable auto-scroll when entering the buffer
    vim.api.nvim_create_autocmd({ "BufEnter", "TermEnter", "WinEnter" }, {
      buffer = subprocess.buffer,
      callback = function()
        local current_win = vim.api.nvim_get_current_win()
        -- When entering the buffer, enable auto-scroll and go to bottom
        subprocess.auto_scroll_enabled = true
        subprocess.user_navigating = false
        vim.api.nvim_win_call(current_win, function()
          local line_count = vim.api.nvim_buf_line_count(subprocess.buffer)
          vim.api.nvim_win_set_cursor(current_win, { line_count, 0 })
        end)
      end,
    })
  end

  -- Return to original window if it's still valid
  safe_set_current_win(original_win)
  return subprocess
end

-- ==============================================================================
-- Configuration Resolution
-- ==============================================================================

-- Load configuration from minipat_config.lua in cwd if it exists
local function load_cwd_config()
  local cwd = vim.fn.getcwd()
  local config_path = cwd .. "/minipat_config.lua"

  if not file_exists(config_path) then
    return nil
  end

  -- Try to load the config file
  local ok, config = pcall(dofile, config_path)
  if not ok then
    vim.notify("Error loading minipat_config.lua: " .. tostring(config), vim.log.levels.ERROR)
    return nil
  end

  if type(config) ~= "table" then
    vim.notify("minipat_config.lua must return a table", vim.log.levels.ERROR)
    return nil
  end

  return config
end

local function resolve_source_path(source_path)
  local plugin_dir = get_plugin_dir()

  if source_path == nil or source_path == "" then
    -- Default: parent directory of plugin
    return vim.fn.fnamemodify(plugin_dir, ":h")
  elseif vim.fn.fnamemodify(source_path, ":p") == source_path then
    -- Absolute path
    return source_path
  else
    -- Relative path to plugin directory
    return plugin_dir .. "/" .. source_path
  end
end

-- Resolve backend configuration paths and set defaults
local function resolve_backend_config(backend_config, source_path)
  if not backend_config.command then
    return backend_config
  end

  local resolved_config = vim.tbl_deep_extend("force", {}, backend_config)

  -- Set default working directory to source_path if not specified
  if not resolved_config.cwd then
    resolved_config.cwd = source_path
  elseif not vim.fn.fnamemodify(resolved_config.cwd, ":p") == resolved_config.cwd then
    -- Relative path - make it relative to source_path
    resolved_config.cwd = source_path .. "/" .. resolved_config.cwd
  end

  -- Set default pidfile path if not specified
  if not resolved_config.pidfile then
    resolved_config.pidfile = "/tmp/minipat-backend.pid"
  elseif not vim.fn.fnamemodify(resolved_config.pidfile, ":p") == resolved_config.pidfile then
    -- Relative path - make it relative to cwd
    resolved_config.pidfile = resolved_config.cwd .. "/" .. resolved_config.pidfile
  end

  return resolved_config
end

-- Validate backend configuration
local function validate_backend_config(backend_config)
  if not backend_config.command or backend_config.command == "" then
    return true, nil
  end

  -- Check if cwd exists
  if backend_config.cwd and not dir_exists(backend_config.cwd) then
    return false, "Backend working directory does not exist: " .. backend_config.cwd
  end

  -- Check if pidfile directory exists
  if backend_config.pidfile then
    local pidfile_dir = vim.fn.fnamemodify(backend_config.pidfile, ":h")
    if not dir_exists(pidfile_dir) then
      return false, "Backend pidfile directory does not exist: " .. pidfile_dir
    end
  end

  return true, nil
end

-- Get current configuration (reload from file if needed)
local function get_current_config(force_reload)
  if not force_reload and state.cached_config then
    return state.cached_config
  end

  -- Start with defaults and initial user args
  local args = vim.tbl_deep_extend("force", DEFAULTS, state.initial_user_args or {})

  -- Load and apply local configuration from minipat_config.lua in cwd if it exists
  local cwd_config = load_cwd_config()
  if cwd_config then
    args = vim.tbl_deep_extend("force", args, cwd_config)
  end

  -- Resolve and validate backend configuration
  local resolved_source_path = resolve_source_path(args.config.source_path)
  local backend_config = resolve_backend_config(args.config.backend, resolved_source_path)
  local backend_valid, backend_error = validate_backend_config(backend_config)

  if not backend_valid then
    vim.notify("Backend configuration error: " .. backend_error, vim.log.levels.ERROR)
    backend_config.command = nil
  end

  -- Store resolved backend config
  args.config.backend = backend_config

  -- Cache the result
  state.cached_config = args

  return args
end

-- Invalidate cached configuration (force reload on next use)
local function invalidate_config_cache()
  state.cached_config = nil
  -- Also clear resolved paths to force re-resolution
  state.resolved_python = nil
  state.resolved_source_path = nil
end


-- Get the python binary from source_path/.venv
local function get_python_binary(source_path)
  local venv_path = source_path .. "/.venv"

  -- Check if venv exists, if not try to create it
  if not dir_exists(venv_path) then
    local pyproject_file = source_path .. "/pyproject.toml"
    if file_exists(pyproject_file) and vim.fn.executable("uv") == 1 then
      print("Creating virtual environment with uv sync in " .. source_path .. "...")
      local result = vim.fn.system('cd "' .. source_path .. '" && uv sync')
      if vim.v.shell_error == 0 and dir_exists(venv_path) then
        print("Virtual environment created successfully")
      else
        vim.notify("Failed to create virtual environment: " .. result, vim.log.levels.ERROR)
        return "python" -- Fallback to system python
      end
    else
      vim.notify("No virtual environment found at " .. venv_path, vim.log.levels.ERROR)
      return "python" -- Fallback to system python
    end
  end

  local python_bin = venv_path .. "/bin/python"
  if file_exists(python_bin) then
    return python_bin
  end

  -- Fallback to system python
  return "python"
end

-- ==============================================================================
-- Subprocess Starters
-- ==============================================================================

local function start_buffer_component(component, args)
  local subprocess = state.subprocesses[component]
  local original_win = vim.api.nvim_get_current_win()

  -- Clean up any existing buffer for this component
  if subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
    local wins = vim.fn.win_findbuf(subprocess.buffer)
    for _, win in ipairs(wins) do
      if vim.api.nvim_win_is_valid(win) then
        vim.api.nvim_win_close(win, false)
      end
    end
    pcall(vim.api.nvim_buf_delete, subprocess.buffer, { force = true })
    subprocess.buffer = nil
  end

  -- Set up window layout for the new buffer component
  if window.ensure_group_visible(state, args) then
    -- Group window exists, add a split within it
    local group_buffers = window.get_group_buffers(state)
    if #group_buffers > 0 then
      -- Find any visible window from the group to split from
      local split_from_win = nil
      for _, buf in ipairs(group_buffers) do
        local wins = vim.fn.win_findbuf(buf)
        if #wins > 0 and vim.api.nvim_win_is_valid(wins[1]) then
          split_from_win = wins[1]
          break
        end
      end

      if split_from_win then
        safe_set_current_win(split_from_win)
      else
        safe_set_current_win(state.group_win)
      end

      local split_dir = window.get_split_direction(args.config.split)
      vim.cmd(window.create_split_command(split_dir, true))
    end
  else
    -- No group window, create it
    vim.api.nvim_set_current_win(original_win)
    window.create_group_split(state, args)
  end

  -- Handle different buffer components
  if component == "cheatsheet" then
    -- Find and open the cheatsheet file
    local resolved_source_path = state.resolved_source_path or resolve_source_path(args.config.source_path)
    local cheatsheet_path = resolved_source_path .. "/CHEATSHEET.md"

    if not file_exists(cheatsheet_path) then
      vim.notify("CHEATSHEET.md not found at: " .. cheatsheet_path, vim.log.levels.WARN)
      vim.api.nvim_set_current_win(original_win)
      return nil
    end

    -- Create and display new buffer for cheatsheet
    subprocess.buffer = vim.api.nvim_create_buf(false, false)
    vim.api.nvim_set_current_buf(subprocess.buffer)

    -- Load the cheatsheet file
    vim.cmd("edit " .. vim.fn.fnameescape(cheatsheet_path))
    subprocess.buffer = vim.api.nvim_get_current_buf()

    -- Set buffer options for read-only viewing
    vim.api.nvim_buf_set_option(subprocess.buffer, "readonly", true)
    vim.api.nvim_buf_set_option(subprocess.buffer, "modifiable", false)
  end

  safe_set_current_win(original_win) -- Return to original window

  -- Trigger reflow after creating buffer component
  vim.defer_fn(function()
    window.reflow_windows_in_group(state)
  end, 150)

  return subprocess
end

local function start_component(component, args, extra_args)
  if component == "repl" then
    -- Resolve source path and python binary if not already done
    if not state.resolved_python or not state.resolved_source_path then
      state.resolved_source_path = resolve_source_path(args.config.source_path)
      if args.config.debug then
        print("Resolved source_path: " .. state.resolved_source_path)
      end
      state.resolved_python = get_python_binary(state.resolved_source_path)
      if args.config.debug then
        print("Using python: " .. state.resolved_python)
      end
    end

    -- Build the command with resolved python binary
    local cmd = state.resolved_python .. " -i -m minipat.boot"
    if extra_args then
      cmd = cmd .. " " .. extra_args
    end
    if args.config.debug then
      print("Executing command: " .. cmd .. " in " .. state.resolved_source_path)
    end

    -- Build environment variables
    local env = { PYTHON_GIL = "0" }
    if args.config.minipat then
      if args.config.minipat.port then
        env.MINIPAT_PORT = tostring(args.config.minipat.port)
      end
      if args.config.minipat.log_path then
        env.MINIPAT_LOG_PATH = tostring(args.config.minipat.log_path)
      end
      if args.config.minipat.log_level then
        env.MINIPAT_LOG_LEVEL = tostring(args.config.minipat.log_level)
      end
      if args.config.minipat.bpm then
        env.MINIPAT_BPM = tostring(args.config.minipat.bpm)
      end
      if args.config.minipat.bpc then
        env.MINIPAT_BPC = tostring(args.config.minipat.bpc)
      end
    end

    return start_subprocess("repl", args, cmd, state.resolved_source_path, env)
  elseif component == "monitor" then
    -- Resolve source path and python binary if not already done
    if not state.resolved_python or not state.resolved_source_path then
      state.resolved_source_path = resolve_source_path(args.config.source_path)
      state.resolved_python = get_python_binary(state.resolved_source_path)
    end

    -- Build the command
    local cmd = state.resolved_python .. " -m minipat.mon"
    if extra_args and extra_args ~= "" then
      cmd = cmd .. " " .. extra_args
    else
      local port = args.config.minipat and args.config.minipat.port or "minipat"
      cmd = cmd .. " -p " .. port
    end

    return start_subprocess("monitor", args, cmd, state.resolved_source_path, { PYTHON_GIL = "0" })
  elseif component == "backend" then
    if not args.config.backend or not args.config.backend.command then
      return nil
    end

    return start_subprocess(
      "backend",
      args,
      args.config.backend.command,
      args.config.backend.cwd,
      args.config.backend.env or {}
    )
  elseif component == "logs" then
    local log_path = args.config.minipat and args.config.minipat.log_path or "/tmp/minipat.log"

    -- Check if log file exists
    if not file_exists(log_path) then
      vim.notify("Log file not found: " .. log_path, vim.log.levels.WARN)
      return nil
    end

    local cmd = "tail -f " .. vim.fn.shellescape(log_path)
    return start_subprocess("logs", args, cmd)
  else
    local component_type = get_component_type(component)
    if component_type == "buffer" then
      return start_buffer_component(component, args)
    end
  end

  return nil
end

-- Graceful REPL quit using unified system
local function quit_repl_gracefully(config, callback)
  local repl = state.subprocesses.repl
  if not repl or not repl.process then
    if callback then
      callback()
    end
    return
  end

  -- Send exit command to gracefully quit the REPL
  M.send(config.minipat.nucleus_var .. ".exit()")

  -- Wait for process to exit gracefully
  local wait_time = config.exit_wait or 1000

  vim.defer_fn(function()
    -- Force quit if still running
    quit_subprocess("repl")
    if callback then
      callback()
    end
  end, wait_time)
end

-- Helper function to start backend if configured and not already running
local function ensure_backend_started(args)
  if not args.config.backend.command then
    return -- No backend configured
  end

  if state.subprocesses.backend and state.subprocesses.backend.process then
    return -- Backend already running
  end

  if args.config.debug then
    print("Auto-starting backend: " .. args.config.backend.command)
  end

  return start_component("backend", args)
end

local function boot_minipat_repl(args, extra_args)
  local original_win = vim.api.nvim_get_current_win()

  local do_boot = function()
    -- Start backend first if configured
    ensure_backend_started(args)

    -- Add small delay to ensure previous process is fully cleaned up
    vim.defer_fn(function()
      -- Ensure group window exists and is visible
      if window.ensure_group_visible(state, args) then
        -- If this is the first buffer in the group, we're already in the group window
        -- If there are already buffers, split within the group
        local group_buffers = window.get_group_buffers(state)
        if #group_buffers > 0 then
          vim.api.nvim_set_current_win(state.group_win)
          local split_dir = window.get_split_direction(args.config.split)
          vim.cmd(window.create_split_command(split_dir, true))
        end
      else
        -- Group is hidden, create the group split from original window
        safe_set_current_win(original_win)
        window.create_group_split(state, args)
      end
      start_component("repl", args, extra_args)
      -- Return to original editor window
      safe_set_current_win(original_win)
    end, 200) -- 200ms delay
  end

  -- If already running, quit the existing process first
  local repl = state.subprocesses.repl
  if repl and repl.process then
    if args.config.debug then
      print("Existing minipat process found, quitting it first...")
    end
    quit_subprocess("repl")
    vim.defer_fn(do_boot, 200)
  else
    do_boot()
  end
end

local function key_map(key, mapping, config)
  local action = KEYMAPS[key].action
  -- Special handling for panic command to use nucleus_var
  if key == "panic" then
    action = function()
      M.send(config.minipat.nucleus_var .. ".panic()")
    end
  end
  vim.keymap.set(KEYMAPS[key].mode, mapping, action, {
    buffer = true,
    desc = KEYMAPS[key].description,
  })
end

function M.send(text)
  local repl = state.subprocesses.repl
  if not repl or not repl.process then
    return
  end
  repl.process:send(text)
end

function M.send_reg(register)
  if not register then
    register = ""
  end
  local text = table.concat(vim.fn.getreg(register, 1, true), "\n")
  M.send(text)
end

local function help_minipat()
  local current_config = get_current_config(false)
  local prefix = current_config.config.command_prefix
  local help_text = {
    "Minipat Neovim Plugin Help",
    "===========================",
    "",
    "Commands:",
    "",
    "Main Commands:",
    "  :" .. prefix .. "Start   - Start backend (if configured) and REPL",
    "  :" .. prefix .. "Quit    - Quit all processes (REPL, Monitor, Backend, Logs)",
    "  :" .. prefix .. "Info    - Show status of all components",
    "  :" .. prefix .. "Hide    - Toggle show/hide all buffers",
    "",
    "REPL Commands:",
    "  :" .. prefix .. "ReplStart   - (Re)start the minipat REPL",
    "  :" .. prefix .. "ReplQuit    - Quit minipat (sends " .. current_config.config.minipat.nucleus_var .. ".exit())",
    "  :" .. prefix .. "ReplHide    - Toggle show/hide REPL buffer",
    "  :" .. prefix .. "ReplStatus  - Show REPL status",
    "",
    "Monitor Commands:",
    "  :" .. prefix .. "MonitorStart  - (Re)start MIDI monitor for minipat port",
    "  :" .. prefix .. "MonitorQuit   - Quit MIDI monitor",
    "  :" .. prefix .. "MonitorHide   - Toggle show/hide monitor buffer",
    "  :" .. prefix .. "MonitorStatus - Show monitor status",
    "",
    "Backend Commands:",
    "  :" .. prefix .. "BackendStart   - (Re)start backend process",
    "  :" .. prefix .. "BackendQuit    - Quit backend process",
    "  :" .. prefix .. "BackendHide    - Toggle show/hide backend buffer",
    "  :" .. prefix .. "BackendStatus  - Show backend process status",
    "  :" .. prefix .. "BackendRestart - Restart the backend process",
    "  :" .. prefix .. "BackendOutput  - Show backend output stream pane",
    "  :" .. prefix .. "BackendClear   - Clear backend output buffer",
    "  :" .. prefix .. "BackendSave [file] - Save backend output to file",
    "",
    "Logs Commands:",
    "  :" .. prefix .. "LogsStart   - (Re)start log viewer with tail -f behavior",
    "  :" .. prefix .. "LogsQuit    - Quit log viewer",
    "  :" .. prefix .. "LogsHide    - Toggle show/hide logs buffer",
    "  :" .. prefix .. "LogsStatus  - Show logs status",
    "",
    "Other Commands:",
    "  :" .. prefix .. "Panic   - Panic minipat (sends " .. current_config.config.minipat.nucleus_var .. ".panic())",
    "  :"
      .. prefix
      .. "Toggle  - Toggle playback ("
      .. current_config.config.minipat.nucleus_var
      .. ".playing = not "
      .. current_config.config.minipat.nucleus_var
      .. ".playing)",
    "  :" .. prefix .. "At <code> - Send Python code to minipat (boots if needed)",
    "  :" .. prefix .. "Config  - Edit the minipat configuration file",
    "  :" .. prefix .. "Help    - Show this help",
    "",
    "Keybindings (in *." .. current_config.config.file_ext .. " files):",
    "  " .. current_config.keymaps.send_line .. "  - Send current line to minipat",
    "  " .. current_config.keymaps.send_visual .. "  - (Visual mode) Send selection to minipat",
    "  "
      .. current_config.keymaps.panic
      .. "  - Panic (pause, reset cycle, clear patterns) - "
      .. current_config.config.minipat.nucleus_var
      .. ".panic()",
    "",
    "Global Keybindings:",
    "  Main:",
    "    "
      .. current_config.global_keymaps.leader_prefix
      .. current_config.global_keymaps.start
      .. "  - Start backend (if configured) and REPL",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.quit .. "  - Quit all processes",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.hide .. "  - Toggle show/hide all buffers",
    "    "
      .. current_config.global_keymaps.leader_prefix
      .. current_config.global_keymaps.info
      .. "  - Show info/status for all components",
    "  REPL:",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.repl_hide .. "  - Toggle show/hide REPL",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.repl_start .. "  - (Re)start REPL",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.repl_quit .. "  - Quit REPL",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.repl_status .. "  - REPL status",
    "  Monitor:",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.monitor_hide .. "  - Toggle show/hide monitor",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.monitor_start .. "  - (Re)start monitor",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.monitor_quit .. "  - Quit monitor",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.monitor_status .. "  - Monitor status",
    "  Backend:",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.backend_hide .. "  - Toggle show/hide backend",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.backend_start .. "  - (Re)start backend",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.backend_quit .. "  - Quit backend",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.backend_status .. "  - Backend status",
    "  Logs:",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.logs_hide .. "  - Toggle show/hide logs",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.logs_start .. "  - (Re)start logs viewer",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.logs_quit .. "  - Quit logs viewer",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.logs_status .. "  - Logs status",
    "  Other:",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.panic .. "  - Panic (stop playback)",
    "    " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.toggle .. "  - Toggle playback",
    "  " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.config .. "  - Edit config file",
    "  " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.help .. "  - Show this help",
    "  " .. current_config.global_keymaps.leader_prefix .. current_config.global_keymaps.at .. "  - Send code to minipat (MpAt)",
    "",
    "Configuration:",
    "  Source path: " .. (current_config.config.source_path or "(auto-detected)"),
    "  Split mode: "
      .. (window.get_split_direction(current_config.config.split) == "v" and "vertical" or "horizontal")
      .. (current_config.config.split == nil and " (auto)" or ""),
    "  Autoclose: " .. tostring(current_config.config.autoclose),
    "  Debug mode: " .. tostring(current_config.config.debug),
    "",
    "Minipat Config:",
    "  Nucleus var: " .. (current_config.config.minipat and current_config.config.minipat.nucleus_var or "(not set)"),
    "  Port: " .. (current_config.config.minipat and current_config.config.minipat.port or "(not set)"),
    "  Log path: " .. (current_config.config.minipat and current_config.config.minipat.log_path or "(not set)"),
    "  Log level: " .. (current_config.config.minipat and current_config.config.minipat.log_level or "(not set)"),
    "  BPM: " .. (current_config.config.minipat and current_config.config.minipat.bpm and tostring(current_config.config.minipat.bpm) or "(not set)"),
    "  BPC: " .. (current_config.config.minipat and current_config.config.minipat.bpc and tostring(current_config.config.minipat.bpc) or "(not set)"),
    "",
    "Backend Config:",
    "  Command: " .. (current_config.config.backend and current_config.config.backend.command or "(not set)"),
    "  Working directory: " .. (current_config.config.backend and current_config.config.backend.cwd or "(not set)"),
    "  Pidfile: " .. (current_config.config.backend and current_config.config.backend.pidfile or "(not set)"),
    "  Autostart: " .. (current_config.config.backend and tostring(current_config.config.backend.autostart) or "false"),
    "  Autostop: " .. (current_config.config.backend and tostring(current_config.config.backend.autostop) or "false"),
    "  Restart on exit: " .. (current_config.config.backend and tostring(current_config.config.backend.restart_on_exit) or "false"),
  }

  -- Create a floating window for the help text
  local buf = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, help_text)
  vim.api.nvim_buf_set_option(buf, "modifiable", false)

  local width = 60
  local height = #help_text
  local win = vim.api.nvim_open_win(buf, true, {
    relative = "editor",
    width = width,
    height = height,
    col = (vim.o.columns - width) / 2,
    row = (vim.o.lines - height) / 2,
    style = "minimal",
    border = "rounded",
  })

  -- Set up key mapping to close the window
  vim.api.nvim_buf_set_keymap(buf, "n", "q", ":close<CR>", { noremap = true, silent = true })
  vim.api.nvim_buf_set_keymap(buf, "n", "<Esc>", ":close<CR>", { noremap = true, silent = true })

  -- Auto-close when losing focus
  vim.api.nvim_create_autocmd({ "WinLeave", "BufLeave" }, {
    buffer = buf,
    once = true,
    callback = function()
      if vim.api.nvim_win_is_valid(win) then
        vim.api.nvim_win_close(win, true)
      end
    end,
  })
end

local function panic_minipat(config)
  local repl = state.subprocesses.repl
  if not repl or not repl.process then
    return
  end
  M.send(config.minipat.nucleus_var .. ".panic()")
end

local function monitor_midi(args, extra_args)
  local original_win = vim.api.nvim_get_current_win()

  -- If monitor is already running, toggle its visibility
  if state.monitor and vim.api.nvim_buf_is_valid(state.monitor) then
    local wins = vim.fn.win_findbuf(state.monitor)
    if #wins > 0 then
      -- Buffer is visible, hide it
      for _, win in ipairs(wins) do
        if vim.api.nvim_win_is_valid(win) then
          vim.api.nvim_win_close(win, false)
        end
      end
      return
    else
      -- Buffer exists but not visible, show it in group
      if window.ensure_group_visible(state, args) then
        local group_buffers = window.get_group_buffers(state)
        if #group_buffers > 0 then
          vim.api.nvim_set_current_win(state.group_win)
          local split_dir = window.get_split_direction(args.config.split)
          vim.cmd(window.create_split_command(split_dir, true))
        end
      else
        vim.api.nvim_set_current_win(original_win)
        window.create_group_split(state, args)
      end
      vim.api.nvim_set_current_buf(state.monitor)
      vim.api.nvim_set_current_win(original_win)
      return
    end
  end

  -- Resolve source path and python binary if not already done
  if not state.resolved_python or not state.resolved_source_path then
    state.resolved_source_path = resolve_source_path(args.config.source_path)
    if args.config.debug then
      print("Resolved source_path: " .. state.resolved_source_path)
    end
    state.resolved_python = get_python_binary(state.resolved_source_path)
    if args.config.debug then
      print("Using python: " .. state.resolved_python)
    end
  end

  -- Create new buffer and split within group
  if window.ensure_group_visible(state, args) then
    local group_buffers = window.get_group_buffers(state)
    if #group_buffers > 0 then
      vim.api.nvim_set_current_win(state.group_win)
      local split_dir = window.get_split_direction(args.config.split)
      vim.cmd(window.create_split_command(split_dir, true))
    end
  else
    vim.api.nvim_set_current_win(original_win)
    window.create_group_split(state, args)
  end
  state.monitor = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_set_current_buf(state.monitor)

  -- Build the command with resolved python binary
  local full_cmd = state.resolved_python .. " -m minipat.mon"
  if extra_args ~= nil and extra_args ~= "" then
    full_cmd = full_cmd .. " " .. extra_args
  else
    local port_name = args.config.minipat and args.config.minipat.port or "minipat"
    full_cmd = full_cmd .. " -p " .. port_name
  end

  -- Create process
  state.monitor_process = process_manager.new({
    name = "monitor",
    command = full_cmd,
    cwd = state.resolved_source_path,
    env = { PYTHON_GIL = "0" },
    debug = args.config.debug,
    autoclose = args.config.autoclose,
    on_exit = function(job_id, exit_code, event_type)
      state.monitor = nil
      state.monitor_process = nil
    end,
  })

  state.monitor_process:start(state.monitor)
  vim.api.nvim_set_current_win(original_win)
end

local function open_logs(args)
  local original_win = vim.api.nvim_get_current_win()
  local log_path = args.config.minipat and args.config.minipat.log_path or "/tmp/minipat.log"

  -- If logs buffer is already open, toggle its visibility
  if state.logs and vim.api.nvim_buf_is_valid(state.logs) then
    local wins = vim.fn.win_findbuf(state.logs)
    if #wins > 0 then
      -- Buffer is visible, hide it
      for _, win in ipairs(wins) do
        if vim.api.nvim_win_is_valid(win) then
          vim.api.nvim_win_close(win, false)
        end
      end
      return
    else
      -- Buffer exists but not visible, show it in group
      if window.ensure_group_visible(state, args) then
        local group_buffers = window.get_group_buffers(state)
        if #group_buffers > 0 then
          vim.api.nvim_set_current_win(state.group_win)
          local split_dir = window.get_split_direction(args.config.split)
          vim.cmd(window.create_split_command(split_dir, true))
        end
      else
        vim.api.nvim_set_current_win(original_win)
        window.create_group_split(state, args)
      end
      vim.api.nvim_set_current_buf(state.logs)
      vim.api.nvim_set_current_win(original_win)
      return
    end
  end

  -- Check if log file exists
  if not file_exists(log_path) then
    vim.notify("Log file not found: " .. log_path, vim.log.levels.WARN)
    return
  end

  -- Split and open the log file within group
  if window.ensure_group_visible(state, args) then
    local group_buffers = window.get_group_buffers(state)
    if #group_buffers > 0 then
      vim.api.nvim_set_current_win(state.group_win)
      local split_dir = window.get_split_direction(args.config.split)
      vim.cmd(window.create_split_command(split_dir, true))
    end
  else
    vim.api.nvim_set_current_win(original_win)
    window.create_group_split(state, args)
  end
  vim.cmd("edit " .. vim.fn.fnameescape(log_path))

  local buf = vim.api.nvim_get_current_buf()
  local win = vim.api.nvim_get_current_win()

  -- Track the logs buffer
  state.logs = buf

  -- Set the buffer to read-only and auto-refresh
  vim.api.nvim_buf_set_option(buf, "readonly", true)
  vim.api.nvim_buf_set_option(buf, "modifiable", false)
  vim.api.nvim_buf_set_option(buf, "autoread", true)

  -- Track if user has manually scrolled up from the bottom
  local user_scrolled_up = false
  local last_line_count = vim.api.nvim_buf_line_count(buf)

  -- Function to check if we're at the bottom of the buffer
  local function is_at_bottom()
    local current_line = vim.api.nvim_win_get_cursor(win)[1]
    local total_lines = vim.api.nvim_buf_line_count(buf)
    return current_line >= total_lines
  end

  -- Function to scroll to bottom
  local function scroll_to_bottom()
    if vim.api.nvim_win_is_valid(win) and vim.api.nvim_buf_is_valid(buf) then
      local total_lines = vim.api.nvim_buf_line_count(buf)
      vim.api.nvim_win_set_cursor(win, { total_lines, 0 })
    end
  end

  -- Track cursor movement to detect manual scrolling
  vim.api.nvim_create_autocmd({ "CursorMoved", "CursorMovedI" }, {
    buffer = buf,
    callback = function()
      -- If user moved cursor and we're not at bottom, they've scrolled up
      if not is_at_bottom() then
        user_scrolled_up = true
      else
        user_scrolled_up = false
      end
    end,
  })

  -- Auto-refresh and tail behavior
  vim.api.nvim_create_autocmd({ "FocusGained", "BufEnter", "CursorHold", "CursorHoldI", "FileChangedShellPost" }, {
    buffer = buf,
    callback = function()
      local current_line_count = vim.api.nvim_buf_line_count(buf)

      -- Reload the file
      vim.cmd("checktime")

      -- If file has grown and user hasn't scrolled up, follow the tail
      local new_line_count = vim.api.nvim_buf_line_count(buf)
      if new_line_count > current_line_count and not user_scrolled_up then
        vim.schedule(scroll_to_bottom)
      end

      last_line_count = new_line_count
    end,
  })

  -- Set up a timer for more frequent checking (like tail -f)
  local timer = vim.loop.new_timer()
  if timer then
    timer:start(
      500,
      500,
      vim.schedule_wrap(function()
        if not vim.api.nvim_buf_is_valid(buf) or not vim.api.nvim_win_is_valid(win) then
          timer:stop()
          timer:close()
          return
        end

        local current_line_count = vim.api.nvim_buf_line_count(buf)

        -- Only check for changes if buffer is still valid
        vim.cmd("silent! checktime")

        local new_line_count = vim.api.nvim_buf_line_count(buf)
        if new_line_count > current_line_count and not user_scrolled_up then
          scroll_to_bottom()
        end
      end)
    )

    -- Clean up timer when buffer is deleted
    vim.api.nvim_create_autocmd({ "BufDelete", "BufUnload" }, {
      buffer = buf,
      once = true,
      callback = function()
        if timer then
          timer:stop()
          timer:close()
        end
      end,
    })
  end

  -- Add keymap to manually go to bottom (useful if user scrolled up)
  vim.api.nvim_buf_set_keymap(buf, "n", "G", "", {
    noremap = true,
    silent = true,
    callback = function()
      scroll_to_bottom()
      user_scrolled_up = false
    end,
  })

  -- Go to end of file initially
  scroll_to_bottom()

  -- Return to original editor window
  vim.api.nvim_set_current_win(original_win)
end

local function show_backend_output(args)
  local original_win = vim.api.nvim_get_current_win()

  -- If backend buffer is already open, toggle its visibility
  if state.backend and vim.api.nvim_buf_is_valid(state.backend) then
    local wins = vim.fn.win_findbuf(state.backend)
    if #wins > 0 then
      -- Buffer is visible, hide it
      for _, win in ipairs(wins) do
        if vim.api.nvim_win_is_valid(win) then
          vim.api.nvim_win_close(win, false)
        end
      end
      return
    else
      -- Buffer exists but not visible, show it in group
      if window.ensure_group_visible(state, args) then
        local group_buffers = window.get_group_buffers(state)
        if #group_buffers > 0 then
          vim.api.nvim_set_current_win(state.group_win)
          local split_dir = window.get_split_direction(args.config.split)
          vim.cmd(window.create_split_command(split_dir, true))
        end
      else
        vim.api.nvim_set_current_win(original_win)
        window.create_group_split(state, args)
      end
      vim.api.nvim_set_current_buf(state.backend)
      vim.api.nvim_set_current_win(original_win)
      return
    end
  end

  -- Check if backend command is configured
  if not args.config.backend.command then
    vim.notify("No backend command configured", vim.log.levels.WARN)
    return
  end

  -- Create new buffer and split within group
  if window.ensure_group_visible(state, args) then
    local group_buffers = window.get_group_buffers(state)
    if #group_buffers > 0 then
      vim.api.nvim_set_current_win(state.group_win)
      local split_dir = window.get_split_direction(args.config.split)
      vim.cmd(window.create_split_command(split_dir, true))
    end
  else
    vim.api.nvim_set_current_win(original_win)
    window.create_group_split(state, args)
  end

  -- Create buffer for backend
  state.backend = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_set_current_buf(state.backend)

  -- Set buffer name and options
  vim.api.nvim_buf_set_name(state.backend, "Backend")
  vim.api.nvim_buf_set_option(state.backend, "buftype", "nofile")
  vim.api.nvim_buf_set_option(state.backend, "swapfile", false)
  vim.api.nvim_buf_set_option(state.backend, "bufhidden", "hide")

  -- Create or reuse backend process
  if not state.backend_process or not state.backend_process.is_running then
    state.backend_process = process_manager.new({
      name = "backend",
      command = args.config.backend.command,
      cwd = args.config.backend.cwd,
      env = args.config.backend.env or {},
      debug = args.config.debug,
      autoclose = args.config.autoclose,
      pidfile = args.config.backend.pidfile,
      on_exit = function(job_id, exit_code, event_type)
        state.backend = nil
        state.backend_process = nil
      end,
    })

    state.backend_process:start(state.backend)
  else
    -- Backend is already running, just attach the buffer for output
    if args.config.debug then
      print("Backend already running, attaching output buffer")
    end
    -- Note: We can't easily attach a new buffer to an existing process
    -- This is a limitation we'll document
    vim.notify("Backend is already running in background. Use BackendRestart to view output.", vim.log.levels.INFO)
  end

  -- Return to original editor window
  vim.api.nvim_set_current_win(original_win)
end

local function monitor_minipat_port(args)
  local original_win = vim.api.nvim_get_current_win()

  -- If monitor is already running, switch to its buffer
  if state.monitor and vim.api.nvim_buf_is_valid(state.monitor) then
    local ok = pcall(vim.api.nvim_set_current_buf, state.monitor)
    if ok then
      return
    end
  end

  -- Resolve source path and python binary if not already done
  if not state.resolved_python or not state.resolved_source_path then
    state.resolved_source_path = resolve_source_path(args.config.source_path)
    if args.config.debug then
      print("Resolved source_path: " .. state.resolved_source_path)
    end
    state.resolved_python = get_python_binary(state.resolved_source_path)
    if args.config.debug then
      print("Using python: " .. state.resolved_python)
    end
  end

  -- Create new buffer and split within group
  if window.ensure_group_visible(state, args) then
    local group_buffers = window.get_group_buffers(state)
    if #group_buffers > 0 then
      vim.api.nvim_set_current_win(state.group_win)
      local split_dir = window.get_split_direction(args.config.split)
      vim.cmd(window.create_split_command(split_dir, true))
    end
  else
    vim.api.nvim_set_current_win(original_win)
    window.create_group_split(state, args)
  end
  state.monitor = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_set_current_buf(state.monitor)

  -- Build the command to monitor the minipat port
  local port_name = args.config.minipat and args.config.minipat.port or "minipat"
  local full_cmd = state.resolved_python .. " -m minipat.mon -p " .. port_name

  -- Create process
  state.monitor_process = process_manager.new({
    name = "monitor",
    command = full_cmd,
    cwd = state.resolved_source_path,
    env = { PYTHON_GIL = "0" },
    debug = args.config.debug,
    autoclose = args.config.autoclose,
    on_exit = function(job_id, exit_code, event_type)
      state.monitor = nil
      state.monitor_process = nil
    end,
  })

  state.monitor_process:start(state.monitor)
  vim.api.nvim_set_current_win(original_win)
end

-- Updated helper functions using the unified subprocess management

-- Helper to quit all processes
local function quit_all_processes(config)
  local processes_quit = {}

  -- Quit all subprocesses
  for name, subprocess in pairs(state.subprocesses) do
    if subprocess.process then
      quit_subprocess(name)
      table.insert(processes_quit, name:upper())
    end
  end

  -- Hide group window
  window.hide_group(state, get_component_names())

  if #processes_quit > 0 then
    vim.notify("Quit: " .. table.concat(processes_quit, ", "), vim.log.levels.INFO)
  else
    vim.notify("No processes running", vim.log.levels.INFO)
  end
end

-- Helper to get overall status info
local function get_all_status()
  local status_lines = {
    "Minipat Status",
    "==============",
    "",
  }

  for name, _ in pairs(state.subprocesses) do
    local status = get_subprocess_status(name)
    table.insert(status_lines, name:upper() .. ":" .. string.rep(" ", 8 - #name) .. status)
  end

  return table.concat(status_lines, "\n")
end

local function open_config()
  local config_path = vim.fn.stdpath("config") .. "/lua/plugins/minipat.lua"
  local config_dir = vim.fn.fnamemodify(config_path, ":h")

  -- Create directory if it doesn't exist
  if not dir_exists(config_dir) then
    vim.fn.mkdir(config_dir, "p")
  end

  -- Create file with basic template if it doesn't exist
  if not file_exists(config_path) then
    local template = {
      "return {",
      "  {",
      '    dir = "path/to/minipat-nvim",',
      "    lazy = true,",
      '    ft = { "minipat" },',
      "    init = function()",
      '      vim.filetype.add({ extension = { minipat = "minipat" } })',
      "    end,",
      "    opts = {},",
      "  },",
      "}",
    }
    vim.fn.writefile(template, config_path)
  end

  vim.cmd("edit " .. vim.fn.fnameescape(config_path))
end


-- All window management functions are now accessible through wrapper functions above

-- ==============================================================================
-- Main Setup Function
-- ==============================================================================

function M.setup(user_args)
  -- Store initial user args for later config reloading
  state.initial_user_args = user_args or {}

  -- Get initial configuration (this will also cache it)
  local args = get_current_config(true)

  local boot_fn = function(fn_args)
    local extra_args = fn_args and fn_args["args"] or nil
    local current_config = get_current_config(true)
    boot_minipat_repl(current_config.config, extra_args)
  end

  local at_fn = function(fn_args)
    local repl = state.subprocesses.repl
    if not repl or not repl.process then
      local notify_id = vim.notify("Minipat is not booted. Booting minipat first...", vim.log.levels.INFO)
      local current_config = get_current_config(true)
      start_component("repl", current_config)
      -- Give minipat a moment to boot before sending code
      vim.defer_fn(function()
        vim.notify("", vim.log.levels.INFO, { replace = notify_id })
        local code = fn_args["args"]
        if code and code ~= "" then
          M.send(code)
        end
      end, 1000) -- Wait 1 second for boot
    else
      local code = fn_args["args"]
      if code and code ~= "" then
        M.send(code)
      end
    end
  end

  local toggle_fn = function()
    local repl = state.subprocesses.repl
    local current_config = get_current_config(false)
    if not repl or not repl.process then
      local notify_id = vim.notify("Minipat is not booted. Booting minipat first...", vim.log.levels.INFO)
      current_config = get_current_config(true)
      start_component("repl", current_config)
      -- Give minipat a moment to boot before toggling
      vim.defer_fn(function()
        vim.notify("", vim.log.levels.INFO, { replace = notify_id })
        M.send(current_config.config.minipat.nucleus_var .. ".playing = not " .. current_config.config.minipat.nucleus_var .. ".playing")
      end, 1000) -- Wait 1 second for boot
    else
      M.send(current_config.config.minipat.nucleus_var .. ".playing = not " .. current_config.config.minipat.nucleus_var .. ".playing")
    end
  end

  local help_fn = function()
    help_minipat()
  end


  -- Remove old backend functions since we're using the unified system

  -- These functions are no longer needed - using unified system

  local enter_fn = function()
    -- Invalidate config cache when entering a minipat buffer to pick up any changes on next use
    invalidate_config_cache()

    vim.cmd("set ft=python")
    vim.api.nvim_buf_set_option(0, "commentstring", "# %s")

    -- Disable LSP for minipat files
    vim.schedule(function()
      local clients = vim.lsp.get_clients({ bufnr = 0 })
      for _, client in ipairs(clients) do
        vim.lsp.buf_detach_client(0, client.id)
      end
    end)

    -- Use initial args for keymaps (they won't change during runtime)
    for key, value in pairs(args.keymaps) do
      key_map(key, value, args.config)
    end
  end

  local prefix = args.config.command_prefix

  -- Helper functions for creating common command patterns
  local function create_start_command(name, desc, start_fn, extra_args_support)
    local opts = { desc = desc }
    if extra_args_support then
      opts.nargs = "*"
    end

    vim.api.nvim_create_user_command(prefix .. name, function(fn_args)
      local current_config = get_current_config(true)
      local extra_args = extra_args_support and fn_args and fn_args["args"] or nil
      start_fn(current_config, extra_args)
    end, opts)
  end

  local function create_quit_command(name, desc, process_name)
    vim.api.nvim_create_user_command(prefix .. name, function()
      quit_subprocess(process_name)
      vim.notify(desc:gsub("Quit ", "") .. " quit", vim.log.levels.INFO)
    end, { desc = desc })
  end

  local function create_toggle_command(name, desc, process_name)
    vim.api.nvim_create_user_command(prefix .. name, function()
      local current_config = get_current_config(false)
      local result = window.toggle_subprocess_visibility(process_name, state, current_config, start_component)
      if result then
        vim.notify(desc:gsub("Toggle ", "") .. " " .. result, vim.log.levels.INFO)
      else
        vim.notify(desc:gsub("Toggle ", "") .. " not started", vim.log.levels.WARN)
      end
    end, { desc = desc })
  end

  local function create_status_command(name, desc, process_name)
    vim.api.nvim_create_user_command(prefix .. name, function()
      local status = get_subprocess_status(process_name)
      vim.notify(desc:gsub(" status", "") .. " status: " .. status, vim.log.levels.INFO)
    end, { desc = desc })
  end

  -- Component Commands
  for component, component_type in pairs(COMPONENTS) do
    local title_case = component:gsub("^%l", string.upper)
    local extra_args_support = component == "repl" -- Only REPL supports extra args

    if component_type == "process" then
      -- Process components get full command set
      create_start_command(title_case .. "Start", "Start " .. component, function(args, extra_args)
        return start_component(component, args, extra_args)
      end, extra_args_support)
      create_quit_command(title_case .. "Quit", "Quit " .. component, component)
      create_status_command(title_case .. "Status", title_case .. " status", component)
    end
    -- Buffer components only get hide/toggle (no start/quit/status commands)

    -- All components get hide/toggle command
    create_toggle_command(title_case .. "Hide", "Toggle " .. component, component)
  end

  -- Main Commands
  vim.api.nvim_create_user_command(prefix .. "Start", function()
    local current_config = get_current_config(true)
    -- Start backend if configured
    if current_config.config.backend and current_config.config.backend.command then
      start_component("backend", current_config)
    end
    -- Start REPL
    start_component("repl", current_config)
  end, { desc = "Start all", nargs = "*" })

  vim.api.nvim_create_user_command(prefix .. "Quit", function()
    local current_config = get_current_config(false)
    quit_all_processes(current_config.config)
  end, { desc = "Quit all" })

  vim.api.nvim_create_user_command(prefix .. "Info", function()
    vim.notify(get_all_status(), vim.log.levels.INFO)
  end, { desc = "Show status" })

  vim.api.nvim_create_user_command(prefix .. "Hide", function()
    local current_config = get_current_config(false)
    local result = window.toggle_all_buffers(state, current_config, start_component, get_component_names())
    if result then
      vim.notify("All buffers " .. result, vim.log.levels.INFO)
    end
  end, { desc = "Toggle all" })

  -- Other Commands
  vim.api.nvim_create_user_command(prefix .. "Panic", function()
    local current_config = get_current_config(false)
    panic_minipat(current_config.config)
  end, { desc = "Stop playback" })
  vim.api.nvim_create_user_command(prefix .. "Toggle", toggle_fn, { desc = "Toggle playback" })
  vim.api.nvim_create_user_command(prefix .. "At", at_fn, { desc = "Send code", nargs = "+" })
  vim.api.nvim_create_user_command(prefix .. "Help", help_fn, { desc = "Show help" })
  vim.api.nvim_create_user_command(prefix .. "ReloadConfig", function()
    invalidate_config_cache()
    local new_config = get_current_config(true)
    vim.notify("Configuration reloaded from minipat_config.lua", vim.log.levels.INFO)
  end, { desc = "Reload configuration from minipat_config.lua" })
  vim.api.nvim_create_user_command(prefix .. "Reflow", function()
    window.reflow_windows_in_group(state)
    vim.notify("Windows reflowed", vim.log.levels.INFO)
  end, { desc = "Reflow windows in group to equal sizes" })

  -- Legacy Commands (for compatibility)
  vim.api.nvim_create_user_command(prefix .. "Boot", function()
    vim.notify("MpBoot is deprecated, use MpStart instead", vim.log.levels.WARN)
    local current_config = get_current_config(true)
    if current_config.config.backend and current_config.config.backend.command then
      start_component("backend", current_config)
    end
    start_component("repl", current_config)
  end, { desc = "Deprecated", nargs = "*" })

  -- Set up global keymaps
  if args.global_keymaps and args.global_keymaps.leader_prefix then
    local leader_prefix = args.global_keymaps.leader_prefix
    -- Individual component keymaps are available through commands
    -- e.g., :MpReplStart, :MpMonitorStart, etc.

    -- Main keymaps
    if args.global_keymaps and args.global_keymaps.start then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.start, function()
        local current_config = get_current_config(true)
        -- Start backend if configured
        if current_config.config.backend and current_config.config.backend.command then
          local ok, err = pcall(start_component, "backend", current_config)
          if not ok then
            vim.notify("Error starting backend: " .. tostring(err), vim.log.levels.ERROR)
          end
        end
        -- Start REPL
        local ok, err = pcall(start_component, "repl", current_config)
        if not ok then
          vim.notify("Error starting REPL: " .. tostring(err), vim.log.levels.ERROR)
        end
      end, { desc = "Start all" })
    end

    if args.global_keymaps and args.global_keymaps.quit then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.quit, function()
        local current_config = get_current_config(false)
        quit_all_processes(current_config.config)
      end, { desc = "Quit all" })
    end

    if args.global_keymaps and args.global_keymaps.hide then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.hide, function()
        local current_config = get_current_config(false)
        local result = window.toggle_all_buffers(state, current_config, start_component, get_component_names())
        if result then
          vim.notify("All buffers " .. result, vim.log.levels.INFO)
        end
      end, { desc = "Toggle all" })
    end

    if args.global_keymaps and args.global_keymaps.all then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.all, function()
        local current_config = get_current_config(false)
        local shown_count = window.show_all_started(state, current_config, start_component, get_component_names())
        if shown_count > 0 then
          vim.notify("Showed " .. shown_count .. " started components", vim.log.levels.INFO)
        else
          vim.notify("All started components already visible", vim.log.levels.INFO)
        end
      end, { desc = "Show all started" })
    end

    if args.global_keymaps and args.global_keymaps.info then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.info, function()
        vim.notify(get_all_status(), vim.log.levels.INFO)
      end, { desc = "Show status" })
    end

    -- Other keymaps
    if args.global_keymaps and args.global_keymaps.panic then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.panic, function()
        local current_config = get_current_config(false)
        panic_minipat(current_config.config)
      end, { desc = "Panic minipat" })
    end
    if args.global_keymaps and args.global_keymaps.toggle then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.toggle, toggle_fn, { desc = "Toggle playback" })
    end
    if args.global_keymaps and args.global_keymaps.help then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.help, help_fn, { desc = "Show help" })
    end
    if args.global_keymaps and args.global_keymaps.at then
      vim.keymap.set("n", leader_prefix .. args.global_keymaps.at, function()
        local code = vim.fn.input("Minipat code: ")
        if code and code ~= "" then
          local repl = state.subprocesses.repl
          if not repl or not repl.process then
            vim.notify("Minipat is not booted. Booting minipat first...", vim.log.levels.INFO)
            local current_config = get_current_config(true)
            start_component("repl", current_config)
            -- Give minipat a moment to boot before sending code
            vim.defer_fn(function()
              M.send(code)
            end, 1000) -- Wait 1 second for boot
          else
            M.send(code)
          end
        end
      end, { desc = "Send code" })
    end

    -- Component keymaps
    for component, component_type in pairs(COMPONENTS) do
      local hide_key = component .. "_hide"
      local start_key = component .. "_start"
      local quit_key = component .. "_quit"
      local status_key = component .. "_status"
      local only_key = component .. "_only"

      -- All components get hide/toggle keymap
      if args.global_keymaps and args.global_keymaps[hide_key] then
        vim.keymap.set("n", leader_prefix .. args.global_keymaps[hide_key], function()
          local current_config = get_current_config(false)
          local result = window.toggle_subprocess_visibility(component, state, current_config, start_component)
          if result then
            vim.notify(component:gsub("^%l", string.upper) .. " " .. result, vim.log.levels.INFO)
          end
        end, { desc = "Toggle " .. component })
      end

      -- Only process components get start/quit/status keymaps
      if component_type == "process" then
        if args.global_keymaps and args.global_keymaps[start_key] then
          vim.keymap.set("n", leader_prefix .. args.global_keymaps[start_key], function()
            local current_config = get_current_config(true)
            start_component(component, current_config)
          end, { desc = "Start " .. component })
        end

        if args.global_keymaps and args.global_keymaps[quit_key] then
          vim.keymap.set("n", leader_prefix .. args.global_keymaps[quit_key], function()
            quit_subprocess(component)
            vim.notify(component:gsub("^%l", string.upper) .. " quit", vim.log.levels.INFO)
          end, { desc = "Quit " .. component })
        end

        if args.global_keymaps and args.global_keymaps[status_key] then
          vim.keymap.set("n", leader_prefix .. args.global_keymaps[status_key], function()
            local status = get_subprocess_status(component)
            vim.notify(component:gsub("^%l", string.upper) .. " status: " .. status, vim.log.levels.INFO)
          end, { desc = component:gsub("^%l", string.upper) .. " status" })
        end
      end
      -- Buffer components only get hide/toggle keymaps (no start/quit/status)

      -- All components get "only" keymap
      if args.global_keymaps and args.global_keymaps[only_key] then
        vim.keymap.set("n", leader_prefix .. args.global_keymaps[only_key], function()
          local current_config = get_current_config(false)
          local component_names = get_component_names()
          local hidden_count = window.hide_all_except(component, state, current_config, start_component, component_names)
          if hidden_count > 0 then
            vim.notify(
              component:gsub("^%l", string.upper) .. " only (hid " .. hidden_count .. " others)",
              vim.log.levels.INFO
            )
          else
            vim.notify(component:gsub("^%l", string.upper) .. " only (no others to hide)", vim.log.levels.INFO)
          end
        end, { desc = "Show only " .. component })
      end
    end
  end

  vim.api.nvim_create_autocmd({ "BufEnter", "BufWinEnter" }, {
    pattern = { "*." .. args.config.file_ext },
    callback = enter_fn,
  })
end

return M
