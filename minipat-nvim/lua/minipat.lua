local M = {}

local treesitter = require('vim.treesitter')

local DEFAULTS = {
  config = {
    source_path = nil, -- Optional path to minipat project root
    file_ext = 'minipat', -- File extension to trigger this plugin
    split = 'v', -- Whether to split vertical (v) or horizontal (h)
    command_prefix = 'Mp', -- prefix for the Boot, Quit commands, etc
    autoclose = false, -- Close the buffer on exit
    nucleus_var = 'n', -- name of the Nucleus variable in the REPL
    exit_wait = 1000, -- milliseconds to wait for process to exit gracefully
    debug = false, -- set to true to see debug messages
  },
  keymaps = {
    send_line = '<C-L>',
    send_node = '<Leader>s',
    send_visual = '<C-L>',
    stop = '<C-H>',
  },
}

local KEYMAPS = {
  send_line = {
    mode = 'n',
    action = 'Vy<cmd>lua require(\'minipat\').send_reg()<CR><ESC>',
    description = 'send line to Minipat',
  },
  send_node = {
    mode = 'n',
    action = function()
      M.send_node()
    end,
    description = 'send treesitter node to Minipat',
  },
  send_visual = {
    mode = 'v',
    action = 'y<cmd>lua require(\'minipat\').send_reg()<CR>',
    description = 'send selection to Minipat',
  },
  stop = {
    mode = 'n',
    action = nil, -- Will be set dynamically with config
    description = 'send stop command to Minipat',
  },
}

local state = {
  booted = false,
  minipat = nil,
  minipat_process = nil,
  resolved_python = nil,
  resolved_source_path = nil,
  quit_callback = nil,
}

-- Helper function to check if a file exists
local function file_exists(path)
  local stat = vim.loop.fs_stat(path)
  return stat and stat.type == 'file'
end

-- Helper function to check if a directory exists
local function dir_exists(path)
  local stat = vim.loop.fs_stat(path)
  return stat and stat.type == 'directory'
end

-- Get the directory of this plugin
local function get_plugin_dir()
  local source = debug.getinfo(1, "S").source:sub(2)
  local plugin_lua_dir = vim.fn.fnamemodify(source, ":p:h")
  return vim.fn.fnamemodify(plugin_lua_dir, ":h")
end

-- Resolve the source path
local function resolve_source_path(source_path)
  local plugin_dir = get_plugin_dir()

  if source_path == nil or source_path == '' then
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
    if file_exists(pyproject_file) and vim.fn.executable('uv') == 1 then
      print("Creating virtual environment with uv sync in " .. source_path .. "...")
      local result = vim.fn.system('cd "' .. source_path .. '" && uv sync')
      if vim.v.shell_error == 0 and dir_exists(venv_path) then
        print("Virtual environment created successfully")
      else
        vim.notify("Failed to create virtual environment: " .. result, vim.log.levels.ERROR)
        return 'python' -- Fallback to system python
      end
    else
      vim.notify("No virtual environment found at " .. venv_path, vim.log.levels.ERROR)
      return 'python' -- Fallback to system python
    end
  end

  local python_bin = venv_path .. "/bin/python"
  if file_exists(python_bin) then
    return python_bin
  end

  -- Fallback to system python
  return 'python'
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
  local job_id = vim.fn.termopen(full_cmd, {
    cwd = state.resolved_source_path,
    env = { PYTHON_GIL = "0" },
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
    M.send(config.nucleus_var .. '.exit()')

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
  local current_win = vim.api.nvim_get_current_win()

  local do_boot = function()
    -- Add small delay to ensure previous process is fully cleaned up
    vim.defer_fn(function()
      vim.cmd(args.split == 'v' and 'vsplit' or 'split')
      boot_minipat(args, extra_args)
      vim.api.nvim_set_current_win(current_win)
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
  -- Special handling for stop command to use nucleus_var
  if key == 'stop' then
    action = function()
      M.send(config.nucleus_var .. '.stop(immediate=True)')
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
  vim.api.nvim_chan_send(state.minipat_process, text .. '\n')
end

function M.send_reg(register)
  if not register then
    register = ''
  end
  local text = table.concat(vim.fn.getreg(register, 1, true), '\n')
  M.send(text)
end

function M.send_node()
  local node = treesitter.get_node_at_cursor(0)
  local root
  if node then
    root = treesitter.get_root_for_node(node)
  end
  if not root then
    return
  end
  local parent
  if node then
    parent = node:parent()
  end
  while node ~= nil and node ~= root do
    local t = node:type()
    if t == 'top_splice' then
      break
    end
    node = parent
    if node then
      parent = node:parent()
    end
  end
  if not node then
    return
  end
  local start_row, start_col, end_row, end_col = treesitter.get_node_range(node)
  local text = table.concat(vim.api.nvim_buf_get_text(0, start_row, start_col, end_row, end_col, {}), '\n')
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
    "  :" .. prefix .. "Quit    - Quit minipat (sends " .. args.config.nucleus_var .. ".exit())",
    "  :" .. prefix .. "Stop    - Stop minipat playback (sends " .. args.config.nucleus_var .. ".stop(immediate=True))",
    "  :" .. prefix .. "At <code> - Send Python code to minipat",
    "  :" .. prefix .. "Help    - Show this help",
    "",
    "Keybindings (in *." .. args.config.file_ext .. " files):",
    "  " .. args.keymaps.send_line .. "  - Send current line to minipat",
    "  " .. args.keymaps.send_node .. "  - Send treesitter node to minipat",
    "  " .. args.keymaps.send_visual .. "  - (Visual mode) Send selection to minipat",
    "  " .. args.keymaps.stop .. "  - Send stop command (" .. args.config.nucleus_var .. ".stop(immediate=True))",
    "",
    "Configuration:",
    "  Source path: " .. (args.config.source_path or "(auto-detected)"),
    "  Split mode: " .. (args.config.split == 'v' and "vertical" or "horizontal"),
    "  Autoclose: " .. tostring(args.config.autoclose),
    "  Nucleus var: " .. args.config.nucleus_var,
    "  Debug mode: " .. tostring(args.config.debug),
  }

  -- Create a floating window for the help text
  local buf = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_buf_set_lines(buf, 0, -1, false, help_text)
  vim.api.nvim_buf_set_option(buf, 'modifiable', false)

  local width = 60
  local height = #help_text
  local win = vim.api.nvim_open_win(buf, true, {
    relative = 'editor',
    width = width,
    height = height,
    col = (vim.o.columns - width) / 2,
    row = (vim.o.lines - height) / 2,
    style = 'minimal',
    border = 'rounded',
  })

  -- Set up key mapping to close the window
  vim.api.nvim_buf_set_keymap(buf, 'n', 'q', ':close<CR>', { noremap = true, silent = true })
  vim.api.nvim_buf_set_keymap(buf, 'n', '<Esc>', ':close<CR>', { noremap = true, silent = true })

  -- Auto-close when losing focus
  vim.api.nvim_create_autocmd({'WinLeave', 'BufLeave'}, {
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
    M.send(config.nucleus_var .. '.stop(immediate=True)')
  end
end

function M.setup(args)
  args = vim.tbl_deep_extend('force', DEFAULTS, args)

  local boot_fn = function(fn_args)
    boot_minipat_repl(args.config, fn_args['args'])
  end

  local at_fn = function(fn_args)
    if not state.booted then
      vim.notify("Minipat is not booted. Run :" .. args.config.command_prefix .. "Boot first", vim.log.levels.ERROR)
      return
    end
    local code = fn_args['args']
    if code and code ~= '' then
      M.send(code)
    end
  end

  local help_fn = function()
    help_minipat(args)
  end

  local enter_fn = function()
    vim.cmd('set ft=python')
    vim.api.nvim_buf_set_option(0, 'commentstring', '# %s')
    for key, value in pairs(args.keymaps) do
      key_map(key, value, args.config)
    end
  end

  local prefix = args.config.command_prefix
  vim.api.nvim_create_user_command(prefix .. 'Boot', boot_fn, { desc = 'boots Minipat instance (can pass extra args)', nargs = '*' })
  vim.api.nvim_create_user_command(prefix .. 'Quit', function() quit_minipat(args.config) end, { desc = 'quits Minipat instance' })
  vim.api.nvim_create_user_command(prefix .. 'Stop', function() stop_minipat(args.config) end, { desc = 'stops Minipat playback' })
  vim.api.nvim_create_user_command(prefix .. 'At', at_fn, { desc = 'send code to Minipat instance', nargs = '+' })
  vim.api.nvim_create_user_command(prefix .. 'Help', help_fn, { desc = 'show Minipat help and keybindings' })
  vim.api.nvim_create_autocmd({ 'BufEnter', 'BufWinEnter' }, {
    pattern = { '*.' .. args.config.file_ext },
    callback = enter_fn,
  })
end

return M
