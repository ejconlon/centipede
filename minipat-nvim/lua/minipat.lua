local M = {}

local DEFAULTS = {
  config = {
    source_path = nil, -- Optional path to minipat project root
    file_ext = "minipat", -- File extension to trigger this plugin
    split = "v", -- Whether to split vertical (v) or horizontal (h)
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
  },
  keymaps = {
    send_line = "<C-L>",
    send_visual = "<C-L>",
    panic = "<C-H>",
  },
  global_keymaps = {
    leader_prefix = "<localleader>p",
    boot = "b",     -- <localleader>pb
    quit = "q",     -- <localleader>pq
    stop = "s",     -- <localleader>ps
    monitor = "m",  -- <localleader>pm (monitor minipat port)
    logs = "l",     -- <localleader>pl
    hide = "h",     -- <localleader>ph
    show = "w",     -- <localleader>pw (show)
    config = "c",   -- <localleader>pc
    help = "?",     -- <localleader>p?
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

local state = {
  booted = false,
  minipat = nil,
  minipat_process = nil,
  monitor = nil,
  monitor_process = nil,
  logs = nil,
  resolved_python = nil,
  resolved_source_path = nil,
  quit_callback = nil,
  -- Buffer group management
  group_hidden = false,
  group_split_dir = "v", -- Direction for the group split
  group_win = nil, -- Main group window
}

-- Helper function to check if a file exists
local function file_exists(path)
  local stat = vim.loop.fs_stat(path)
  return stat and stat.type == "file"
end

-- Helper function to check if a directory exists
local function dir_exists(path)
  local stat = vim.loop.fs_stat(path)
  return stat and stat.type == "directory"
end

-- Get the directory of this plugin
local function get_plugin_dir()
  local source = debug.getinfo(1, "S").source:sub(2)
  local plugin_lua_dir = vim.fn.fnamemodify(source, ":p:h")
  return vim.fn.fnamemodify(plugin_lua_dir, ":h")
end

-- Helper functions for buffer group management
local function get_opposite_split(split_dir)
  return split_dir == "v" and "h" or "v"
end

local function create_group_split(args)
  if state.group_hidden or (state.group_win and vim.api.nvim_win_is_valid(state.group_win)) then
    return state.group_win
  end

  -- Create the main group split in opposite direction from current window
  local opposite_dir = get_opposite_split(args.split)
  state.group_split_dir = opposite_dir

  -- Create the group split from the current window (editor)
  vim.cmd(opposite_dir == "v" and "vsplit" or "split")
  state.group_win = vim.api.nvim_get_current_win()

  return state.group_win
end

local function ensure_group_visible(args)
  if state.group_hidden then
    return false
  end

  if not state.group_win or not vim.api.nvim_win_is_valid(state.group_win) then
    create_group_split(args)
  end

  return true
end

local function get_group_buffers()
  local buffers = {}
  if state.minipat and vim.api.nvim_buf_is_valid(state.minipat) then
    table.insert(buffers, state.minipat)
  end
  if state.monitor and vim.api.nvim_buf_is_valid(state.monitor) then
    table.insert(buffers, state.monitor)
  end
  if state.logs and vim.api.nvim_buf_is_valid(state.logs) then
    table.insert(buffers, state.logs)
  end
  return buffers
end

-- Resolve the source path
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

local function boot_minipat(args, extra_args)
  -- Always create a fresh buffer for the terminal
  state.minipat = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_set_current_buf(state.minipat)

  -- Resolve source path and python binary if not already done
  if not state.resolved_python or not state.resolved_source_path then
    state.resolved_source_path = resolve_source_path(args.source_path)
    if args.debug then
      print("Resolved source_path: " .. state.resolved_source_path)
    end
    state.resolved_python = get_python_binary(state.resolved_source_path)
    if args.debug then
      print("Using python: " .. state.resolved_python)
    end
  end

  -- Build the command with resolved python binary
  local full_cmd = state.resolved_python .. " -i -m minipat.boot"
  if extra_args ~= nil then
    full_cmd = full_cmd .. " " .. extra_args
  end
  if args.debug then
    print("Executing command: " .. full_cmd .. " in " .. state.resolved_source_path)
  end

  -- Build environment variables
  local env = { PYTHON_GIL = "0" }
  if args.minipat then
    if args.minipat.port then
      env.MINIPAT_PORT = tostring(args.minipat.port)
    end
    if args.minipat.log_path then
      env.MINIPAT_LOG_PATH = tostring(args.minipat.log_path)
    end
    if args.minipat.log_level then
      env.MINIPAT_LOG_LEVEL = tostring(args.minipat.log_level)
    end
    if args.minipat.bpm then
      env.MINIPAT_BPM = tostring(args.minipat.bpm)
    end
    if args.minipat.bpc then
      env.MINIPAT_BPC = tostring(args.minipat.bpc)
    end
  end

  local job_id = vim.fn.termopen(full_cmd, {
    cwd = state.resolved_source_path,
    env = env,
    on_exit = function(job_id, exit_code, event_type)
      if args.debug then
        print("Minipat process exited with code: " .. exit_code)
      end

      -- Only close window and delete buffer if autoclose is enabled
      if args.autoclose then
        if state.minipat and vim.api.nvim_buf_is_valid(state.minipat) then
          if #vim.fn.win_findbuf(state.minipat) > 0 then
            vim.api.nvim_win_close(vim.fn.win_findbuf(state.minipat)[1], true)
          end
          vim.api.nvim_buf_delete(state.minipat, { unload = true })
        end
        state.minipat = nil
      end

      -- Clean up state
      state.minipat_process = nil
      state.resolved_python = nil
      state.resolved_source_path = nil
      state.booted = false

      -- Call quit callback if one was registered
      if state.quit_callback then
        local cb = state.quit_callback
        state.quit_callback = nil
        vim.schedule(cb) -- Schedule callback to avoid potential issues
      end
    end,
  })

  if job_id <= 0 then
    vim.notify("Failed to start minipat process (job_id: " .. job_id .. ")", vim.log.levels.ERROR)
    state.booted = false
    return
  end

  state.minipat_process = job_id
end

local function quit_minipat(config, callback)
  if not state.booted then
    if callback then
      callback()
    end
    return
  end

  -- Register callback to be called when process exits
  if callback then
    state.quit_callback = callback
  end

  if state.minipat_process then
    local process_to_quit = state.minipat_process -- Capture the process ID we want to quit
    -- Send exit command to gracefully quit the REPL
    M.send(config.minipat.nucleus_var .. ".exit()")

    -- Wait for process to exit gracefully
    local wait_time = config.exit_wait or 1000

    vim.defer_fn(function()
      -- Only kill the specific process we intended to quit, not whatever might be in state now
      if state.minipat_process == process_to_quit then
        vim.fn.jobstop(process_to_quit)
        -- Give it a moment to fully clean up
        vim.defer_fn(function()
          -- Force cleanup if callback wasn't called
          if state.quit_callback then
            local cb = state.quit_callback
            state.quit_callback = nil
            cb()
          end
        end, 100)
      end
    end, wait_time)
  else
    -- No process running, call callback immediately
    if callback then
      callback()
    end
  end
  state.booted = false
end

local function boot_minipat_repl(args, extra_args)
  local original_win = vim.api.nvim_get_current_win()

  local do_boot = function()
    -- Add small delay to ensure previous process is fully cleaned up
    vim.defer_fn(function()
      -- Ensure group window exists and is visible
      if ensure_group_visible(args) then
        -- If this is the first buffer in the group, we're already in the group window
        -- If there are already buffers, split within the group
        local group_buffers = get_group_buffers()
        if #group_buffers > 0 then
          vim.api.nvim_set_current_win(state.group_win)
          vim.cmd(args.split == "v" and "vsplit" or "split")
        end
      else
        -- Group is hidden, create the group split from original window
        vim.api.nvim_set_current_win(original_win)
        create_group_split(args)
      end
      boot_minipat(args, extra_args)
      -- Return to original editor window
      vim.api.nvim_set_current_win(original_win)
      state.booted = true
    end, 200) -- 200ms delay
  end

  -- If already booted, quit the existing process first
  if state.booted then
    if args.debug then
      print("Existing minipat process found, quitting it first...")
    end
    quit_minipat(args, do_boot)
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
  if not state.minipat_process then
    return
  end
  vim.api.nvim_chan_send(state.minipat_process, text .. "\n")
end

function M.send_reg(register)
  if not register then
    register = ""
  end
  local text = table.concat(vim.fn.getreg(register, 1, true), "\n")
  M.send(text)
end


local function help_minipat(args)
  local prefix = args.config.command_prefix
  local help_text = {
    "Minipat Neovim Plugin Help",
    "===========================",
    "",
    "Commands:",
    "  :" .. prefix .. "Boot    - Boot the minipat REPL",
    "  :" .. prefix .. "Quit    - Quit minipat (sends " .. args.config.minipat.nucleus_var .. ".exit())",
    "  :"
      .. prefix
      .. "Stop    - Stop minipat playback (sends "
      .. args.config.minipat.nucleus_var
      .. ".stop(immediate=True))",
    "  :" .. prefix .. "At <code> - Send Python code to minipat",
    "  :" .. prefix .. "Mon     - Toggle MIDI monitor",
    "  :" .. prefix .. "Mod     - Monitor minipat MIDI port",
    "  :" .. prefix .. "Logs    - Toggle minipat log file",
    "  :" .. prefix .. "Hide    - Hide minipat buffer group",
    "  :" .. prefix .. "Show    - Show minipat buffer group",
    "  :" .. prefix .. "Config  - Edit minipat config file",
    "  :" .. prefix .. "Help    - Show this help",
    "",
    "Keybindings (in *." .. args.config.file_ext .. " files):",
    "  " .. args.keymaps.send_line .. "  - Send current line to minipat",
    "  " .. args.keymaps.send_visual .. "  - (Visual mode) Send selection to minipat",
    "  "
      .. args.keymaps.panic
      .. "  - Panic (pause, reset cycle, clear patterns) - "
      .. args.config.minipat.nucleus_var
      .. ".panic()",
    "",
    "Global Keybindings:",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.boot .. "  - Boot minipat REPL",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.quit .. "  - Quit minipat",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.stop .. "  - Stop minipat playback",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.monitor .. "  - Monitor minipat port",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.logs .. "  - Open log file",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.hide .. "  - Hide buffer group",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.show .. "  - Show buffer group",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.config .. "  - Edit config file",
    "  " .. args.global_keymaps.leader_prefix .. args.global_keymaps.help .. "  - Show this help",
    "",
    "Configuration:",
    "  Source path: " .. (args.config.source_path or "(auto-detected)"),
    "  Split mode: " .. (args.config.split == "v" and "vertical" or "horizontal"),
    "  Autoclose: " .. tostring(args.config.autoclose),
    "  Debug mode: " .. tostring(args.config.debug),
    "",
    "Minipat Config:",
    "  Nucleus var: " .. (args.config.minipat and args.config.minipat.nucleus_var or "(not set)"),
    "  Port: " .. (args.config.minipat and args.config.minipat.port or "(not set)"),
    "  Log path: " .. (args.config.minipat and args.config.minipat.log_path or "(not set)"),
    "  Log level: " .. (args.config.minipat and args.config.minipat.log_level or "(not set)"),
    "  BPM: " .. (args.config.minipat and args.config.minipat.bpm and tostring(args.config.minipat.bpm) or "(not set)"),
    "  BPC: " .. (args.config.minipat and args.config.minipat.bpc and tostring(args.config.minipat.bpc) or "(not set)"),
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

local function stop_minipat(config)
  if not state.booted then
    return
  end
  if state.minipat_process then
    M.send(config.minipat.nucleus_var .. ".stop(immediate=True)")
  end
end

local function monitor_midi(args)
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
      if ensure_group_visible(args) then
        local group_buffers = get_group_buffers()
        if #group_buffers > 0 then
          vim.api.nvim_set_current_win(state.group_win)
          vim.cmd(args.split == "v" and "vsplit" or "split")
        end
      else
        vim.api.nvim_set_current_win(original_win)
        create_group_split(args)
      end
      vim.api.nvim_set_current_buf(state.monitor)
      vim.api.nvim_set_current_win(original_win)
      return
    end
  end

  -- Resolve source path and python binary if not already done
  if not state.resolved_python or not state.resolved_source_path then
    state.resolved_source_path = resolve_source_path(args.source_path)
    if args.debug then
      print("Resolved source_path: " .. state.resolved_source_path)
    end
    state.resolved_python = get_python_binary(state.resolved_source_path)
    if args.debug then
      print("Using python: " .. state.resolved_python)
    end
  end

  -- Create new buffer and split within group
  if ensure_group_visible(args) then
    local group_buffers = get_group_buffers()
    if #group_buffers > 0 then
      vim.api.nvim_set_current_win(state.group_win)
      vim.cmd(args.split == "v" and "vsplit" or "split")
    end
  else
    vim.api.nvim_set_current_win(original_win)
    create_group_split(args)
  end
  state.monitor = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_set_current_buf(state.monitor)

  -- Build the command with resolved python binary
  -- Default to monitoring minipat port if no args, otherwise use args for monitoring
  local full_cmd = state.resolved_python .. " -m minipat.mon"
  if extra_args ~= nil and extra_args ~= "" then
    full_cmd = full_cmd .. " " .. extra_args
  else
    local port_name = args.minipat and args.minipat.port or "minipat"
    full_cmd = full_cmd .. " -p " .. port_name
  end
  if args.debug then
    print("Executing MIDI monitor command: " .. full_cmd .. " in " .. state.resolved_source_path)
  end

  local job_id = vim.fn.termopen(full_cmd, {
    cwd = state.resolved_source_path,
    env = { PYTHON_GIL = "0" },
    on_exit = function(job_id, exit_code, event_type)
      if args.debug then
        print("MIDI monitor process exited with code: " .. exit_code)
      end

      -- Only close window and delete buffer if autoclose is enabled
      if args.autoclose then
        if state.monitor and vim.api.nvim_buf_is_valid(state.monitor) then
          if #vim.fn.win_findbuf(state.monitor) > 0 then
            vim.api.nvim_win_close(vim.fn.win_findbuf(state.monitor)[1], true)
          end
          vim.api.nvim_buf_delete(state.monitor, { unload = true })
        end
        state.monitor = nil
      end

      -- Clean up state
      state.monitor_process = nil
    end,
  })

  if job_id <= 0 then
    vim.notify("Failed to start MIDI monitor process (job_id: " .. job_id .. ")", vim.log.levels.ERROR)
    return
  end

  state.monitor_process = job_id
  vim.api.nvim_set_current_win(original_win)
end

local function open_logs(args)
  local original_win = vim.api.nvim_get_current_win()
  local log_path = args.minipat and args.minipat.log_path or "/tmp/minipat.log"

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
      if ensure_group_visible(args) then
        local group_buffers = get_group_buffers()
        if #group_buffers > 0 then
          vim.api.nvim_set_current_win(state.group_win)
          vim.cmd(args.split == "v" and "vsplit" or "split")
        end
      else
        vim.api.nvim_set_current_win(original_win)
        create_group_split(args)
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
  if ensure_group_visible(args) then
    local group_buffers = get_group_buffers()
    if #group_buffers > 0 then
      vim.api.nvim_set_current_win(state.group_win)
      vim.cmd(args.split == "v" and "vsplit" or "split")
    end
  else
    vim.api.nvim_set_current_win(original_win)
    create_group_split(args)
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
    state.resolved_source_path = resolve_source_path(args.source_path)
    if args.debug then
      print("Resolved source_path: " .. state.resolved_source_path)
    end
    state.resolved_python = get_python_binary(state.resolved_source_path)
    if args.debug then
      print("Using python: " .. state.resolved_python)
    end
  end

  -- Create new buffer and split within group
  if ensure_group_visible(args) then
    local group_buffers = get_group_buffers()
    if #group_buffers > 0 then
      vim.api.nvim_set_current_win(state.group_win)
      vim.cmd(args.split == "v" and "vsplit" or "split")
    end
  else
    vim.api.nvim_set_current_win(original_win)
    create_group_split(args)
  end
  state.monitor = vim.api.nvim_create_buf(false, false)
  vim.api.nvim_set_current_buf(state.monitor)

  -- Build the command to monitor the minipat port
  local port_name = args.minipat and args.minipat.port or "minipat"
  local full_cmd = state.resolved_python .. " -m minipat.mon -p " .. port_name
  if args.debug then
    print("Executing MIDI monitor command: " .. full_cmd .. " in " .. state.resolved_source_path)
  end

  local job_id = vim.fn.termopen(full_cmd, {
    cwd = state.resolved_source_path,
    env = { PYTHON_GIL = "0" },
    on_exit = function(job_id, exit_code, event_type)
      if args.debug then
        print("MIDI monitor process exited with code: " .. exit_code)
      end

      -- Only close window and delete buffer if autoclose is enabled
      if args.autoclose then
        if state.monitor and vim.api.nvim_buf_is_valid(state.monitor) then
          if #vim.fn.win_findbuf(state.monitor) > 0 then
            vim.api.nvim_win_close(vim.fn.win_findbuf(state.monitor)[1], true)
          end
          vim.api.nvim_buf_delete(state.monitor, { unload = true })
        end
        state.monitor = nil
      end

      -- Clean up state
      state.monitor_process = nil
    end,
  })

  if job_id <= 0 then
    vim.notify("Failed to start MIDI monitor process (job_id: " .. job_id .. ")", vim.log.levels.ERROR)
    return
  end

  state.monitor_process = job_id
  vim.api.nvim_set_current_win(original_win)
end

local function hide_group()
  if state.group_hidden then
    return
  end

  -- Find all windows displaying group buffers and close them
  local group_buffers = get_group_buffers()
  for _, buf in ipairs(group_buffers) do
    local wins = vim.fn.win_findbuf(buf)
    for _, win in ipairs(wins) do
      if vim.api.nvim_win_is_valid(win) then
        vim.api.nvim_win_close(win, false)
      end
    end
  end

  -- Close the main group window if it exists
  if state.group_win and vim.api.nvim_win_is_valid(state.group_win) then
    vim.api.nvim_win_close(state.group_win, false)
  end

  state.group_hidden = true
  state.group_win = nil
end

local function show_group(args)
  if not state.group_hidden then
    return
  end

  state.group_hidden = false

  -- Create the main group split
  create_group_split(args)

  -- Restore buffers that exist
  local group_buffers = get_group_buffers()
  if #group_buffers > 0 then
    vim.api.nvim_set_current_win(state.group_win)

    -- Show the first buffer in the group window
    vim.api.nvim_set_current_buf(group_buffers[1])

    -- Split and show additional buffers
    for i = 2, #group_buffers do
      vim.cmd(args.split == "v" and "vsplit" or "split")
      vim.api.nvim_set_current_buf(group_buffers[i])
    end
  end
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
      '    dir = "minipat-nvim",',
      "    lazy = true,",
      '    ft = { "minipat" },',
      "    init = function()",
      '      vim.filetype.add({ extension = { minipat = "minipat" } })',
      "    end,",
      "    opts = {",
      "      config = {",
      '        -- source_path = nil,  -- path to minipat source (nil = parent of plugin)',
      '        -- command_prefix = "Mp",  -- prefix for commands like :MpBoot',
      "        -- autoclose = true,  -- auto-close REPL buffer on exit",
      '        -- split = "v",  -- split direction: "v" (vertical) or "h" (horizontal)',
      "        -- exit_wait = 500,  -- ms to wait for graceful exit",
      "        -- debug = false,  -- show debug messages",
      "        minipat = {",
      '          nucleus_var = "n",  -- nucleus variable name',
      '          port = "minipat",  -- MIDI port name (MINIPAT_PORT)',
      '          log_path = "/tmp/minipat.log",  -- log file path (MINIPAT_LOG_PATH)',
      '          log_level = "INFO",  -- log level (MINIPAT_LOG_LEVEL)',
      "          bpm = 120,  -- initial BPM (MINIPAT_BPM)",
      "          bpc = 4,  -- beats per cycle (MINIPAT_BPC)",
      "        },",
      "      },",
      "      keymaps = {",
      '        send_line = "<C-L>",  -- send current line',
      '        send_visual = "<C-L>",  -- send selection (visual mode)',
      '        panic = "<C-H>",  -- panic: pause, reset cycle, clear patterns',
      "      },",
      "      global_keymaps = {",
      '        leader_prefix = "<localleader>p",',
      '        boot = "b",  -- <localleader>pb',
      '        quit = "q",  -- <localleader>pq',
      '        stop = "s",  -- <localleader>ps',
      '        monitor = "m",  -- <localleader>pm',
      '        logs = "l",  -- <localleader>pl',
      '        hide = "h",  -- <localleader>ph',
      '        show = "w",  -- <localleader>pw',
      '        config = "c",  -- <localleader>pc',
      '        help = "?",  -- <localleader>p?',
      "      },",
      "    },",
      "  },",
      "}",
    }
    vim.fn.writefile(template, config_path)
  end

  vim.cmd("edit " .. vim.fn.fnameescape(config_path))
end

function M.setup(args)
  args = vim.tbl_deep_extend("force", DEFAULTS, args)

  local boot_fn = function(fn_args)
    local extra_args = fn_args and fn_args["args"] or nil
    boot_minipat_repl(args.config, extra_args)
  end

  local at_fn = function(fn_args)
    if not state.booted then
      vim.notify("Minipat is not booted. Run :" .. args.config.command_prefix .. "Boot first", vim.log.levels.ERROR)
      return
    end
    local code = fn_args["args"]
    if code and code ~= "" then
      M.send(code)
    end
  end

  local help_fn = function()
    help_minipat(args)
  end

  local mon_fn = function()
    monitor_midi(args.config)
  end

  local mod_fn = function()
    monitor_minipat_port(args.config)
  end

  local logs_fn = function()
    open_logs(args.config)
  end

  local hide_fn = function()
    hide_group()
  end

  local show_fn = function()
    show_group(args.config)
  end

  local config_fn = function()
    open_config()
  end

  local enter_fn = function()
    vim.cmd("set ft=python")
    vim.api.nvim_buf_set_option(0, "commentstring", "# %s")

    -- Disable LSP for minipat files
    vim.schedule(function()
      local clients = vim.lsp.get_clients({ bufnr = 0 })
      for _, client in ipairs(clients) do
        vim.lsp.buf_detach_client(0, client.id)
      end
    end)

    for key, value in pairs(args.keymaps) do
      key_map(key, value, args.config)
    end
  end

  local prefix = args.config.command_prefix
  vim.api.nvim_create_user_command(
    prefix .. "Boot",
    boot_fn,
    { desc = "boots Minipat instance (can pass extra args)", nargs = "*" }
  )
  vim.api.nvim_create_user_command(prefix .. "Quit", function()
    quit_minipat(args.config)
  end, { desc = "quits Minipat instance" })
  vim.api.nvim_create_user_command(prefix .. "Stop", function()
    stop_minipat(args.config)
  end, { desc = "stops Minipat playback" })
  vim.api.nvim_create_user_command(prefix .. "At", at_fn, { desc = "send code to Minipat instance", nargs = "+" })
  vim.api.nvim_create_user_command(prefix .. "Mon", mon_fn, { desc = "toggle MIDI monitor" })
  vim.api.nvim_create_user_command(prefix .. "Mod", mod_fn, { desc = "monitor minipat MIDI port" })
  vim.api.nvim_create_user_command(prefix .. "Logs", logs_fn, { desc = "open minipat log file" })
  vim.api.nvim_create_user_command(prefix .. "Hide", hide_fn, { desc = "hide minipat buffer group" })
  vim.api.nvim_create_user_command(prefix .. "Show", show_fn, { desc = "show minipat buffer group" })
  vim.api.nvim_create_user_command(prefix .. "Config", config_fn, { desc = "edit minipat config file" })
  vim.api.nvim_create_user_command(prefix .. "Help", help_fn, { desc = "show Minipat help and keybindings" })

  -- Set up global keymaps
  local leader_prefix = args.global_keymaps.leader_prefix
  if leader_prefix then
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.boot, boot_fn, { desc = "Boot minipat REPL" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.quit, function() quit_minipat(args.config) end, { desc = "Quit minipat" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.stop, function() stop_minipat(args.config) end, { desc = "Stop minipat playback" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.monitor, function() monitor_minipat_port(args.config) end, { desc = "Monitor minipat port" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.logs, logs_fn, { desc = "Open minipat log file" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.hide, hide_fn, { desc = "Hide minipat buffer group" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.show, show_fn, { desc = "Show minipat buffer group" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.config, config_fn, { desc = "Edit minipat config" })
    vim.keymap.set("n", leader_prefix .. args.global_keymaps.help, help_fn, { desc = "Show minipat help" })
  end

  vim.api.nvim_create_autocmd({ "BufEnter", "BufWinEnter" }, {
    pattern = { "*." .. args.config.file_ext },
    callback = enter_fn,
  })
end

return M
