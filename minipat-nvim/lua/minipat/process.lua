local M = {}

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

-- Read PID from pidfile
function M.read_pidfile(pidfile_path)
  if not file_exists(pidfile_path) then
    return nil
  end

  local f = io.open(pidfile_path, "r")
  if not f then
    return nil
  end

  local pid_str = f:read("*all")
  f:close()

  if not pid_str or pid_str == "" then
    return nil
  end

  local pid = tonumber(pid_str:match("(%d+)"))
  return pid
end

-- Write PID to pidfile
function M.write_pidfile(pidfile_path, pid)
  local pidfile_dir = vim.fn.fnamemodify(pidfile_path, ":h")
  if not dir_exists(pidfile_dir) then
    vim.fn.mkdir(pidfile_dir, "p")
  end

  local f = io.open(pidfile_path, "w")
  if not f then
    return false
  end

  f:write(tostring(pid) .. "\n")
  f:close()
  return true
end

-- Remove pidfile
function M.remove_pidfile(pidfile_path)
  if file_exists(pidfile_path) then
    os.remove(pidfile_path)
  end
end

-- Check if process is running
function M.is_process_running(pid)
  if not pid then
    return false
  end

  -- Use kill -0 to check if process exists
  local result = vim.fn.system("kill -0 " .. pid .. " 2>/dev/null")
  return vim.v.shell_error == 0
end

-- Process management class
local Process = {}
Process.__index = Process

function Process:new(opts)
  local instance = {
    name = opts.name or "process",
    command = opts.command,
    cwd = opts.cwd,
    env = opts.env or {},
    debug = opts.debug or false,
    pidfile = opts.pidfile,
    autoclose = opts.autoclose ~= false, -- default true
    exit_wait = opts.exit_wait or 500,
    on_exit = opts.on_exit,
    on_stdout = opts.on_stdout,
    on_stderr = opts.on_stderr,

    -- Internal state
    job_id = nil,
    buffer = nil,
    pid = nil,
    is_running = false,
  }
  setmetatable(instance, self)
  return instance
end

function Process:start(buffer)
  if self.is_running then
    if self.debug then
      print(self.name .. " is already running")
    end
    return false
  end

  if not self.command then
    vim.notify("No command specified for " .. self.name, vim.log.levels.ERROR)
    return false
  end

  -- Set up buffer if provided
  self.buffer = buffer

  if self.debug then
    print("Starting " .. self.name .. ": " .. self.command .. (self.cwd and (" in " .. self.cwd) or ""))
  end

  -- Build environment
  local env_vars = vim.tbl_extend("force", vim.fn.environ(), self.env)

  local opts = {
    cwd = self.cwd,
    env = env_vars,
    on_exit = function(job_id, exit_code, event_type)
      if self.debug then
        print(self.name .. " exited with code: " .. exit_code)
      end

      -- Clean up pidfile if we're tracking one
      if self.pidfile then
        M.remove_pidfile(self.pidfile)
      end

      -- Call custom exit handler if provided
      if self.on_exit then
        self.on_exit(job_id, exit_code, event_type)
      end

      -- Handle buffer cleanup if autoclose is enabled
      if self.autoclose and self.buffer and vim.api.nvim_buf_is_valid(self.buffer) then
        local wins = vim.fn.win_findbuf(self.buffer)
        for _, win in ipairs(wins) do
          if vim.api.nvim_win_is_valid(win) then
            vim.api.nvim_win_close(win, true)
          end
        end
        vim.api.nvim_buf_delete(self.buffer, { unload = true })
        self.buffer = nil
      end

      -- Reset state
      self.job_id = nil
      self.pid = nil
      self.is_running = false
    end,
  }

  -- Add stdout/stderr handlers if provided
  if self.on_stdout then
    opts.on_stdout = self.on_stdout
  end
  if self.on_stderr then
    opts.on_stderr = self.on_stderr
  end

  -- Start the process
  if self.buffer then
    -- Use termopen for terminal-based processes
    self.job_id = vim.fn.termopen(self.command, opts)
  else
    -- Use jobstart for background processes
    self.job_id = vim.fn.jobstart(self.command, opts)
  end

  if self.job_id <= 0 then
    vim.notify("Failed to start " .. self.name .. " (job_id: " .. self.job_id .. ")", vim.log.levels.ERROR)
    return false
  end

  -- Get PID if possible
  self.pid = vim.fn.jobpid(self.job_id)
  if self.pid > 0 then
    self.is_running = true

    -- Write pidfile if configured
    if self.pidfile then
      M.write_pidfile(self.pidfile, self.pid)
    end
  end

  return true
end

function Process:stop(force)
  if not self.is_running or not self.job_id then
    return true
  end

  if self.debug then
    print("Stopping " .. self.name .. (force and " (force)" or ""))
  end

  if force then
    -- Force stop immediately
    vim.fn.jobstop(self.job_id)
    return true
  end

  -- Graceful stop - send termination signal and wait
  vim.fn.jobstop(self.job_id)

  -- Wait for process to exit gracefully
  vim.defer_fn(function()
    if self.is_running and self.job_id then
      if self.debug then
        print("Force stopping " .. self.name .. " after timeout")
      end
      vim.fn.jobstop(self.job_id)
    end
  end, self.exit_wait)

  return true
end

function Process:restart(buffer)
  if self.is_running then
    self:stop()
    -- Wait a bit before restarting
    vim.defer_fn(function()
      self:start(buffer)
    end, 200)
  else
    self:start(buffer)
  end
end

function Process:send(data)
  if not self.is_running or not self.job_id then
    return false
  end

  vim.api.nvim_chan_send(self.job_id, data .. "\n")
  return true
end

function Process:status()
  return {
    name = self.name,
    is_running = self.is_running,
    job_id = self.job_id,
    pid = self.pid,
    command = self.command,
    cwd = self.cwd,
  }
end

-- Factory function to create a new process
function M.new(opts)
  return Process:new(opts)
end

-- Utility function to create a terminal process with buffer management
function M.create_terminal_process(opts)
  local buffer = opts.buffer
  if not buffer then
    buffer = vim.api.nvim_create_buf(false, false)
    if opts.buffer_name then
      vim.api.nvim_buf_set_name(buffer, opts.buffer_name)
    end
  end

  local process = M.new(opts)
  process:start(buffer)

  return process, buffer
end

return M
