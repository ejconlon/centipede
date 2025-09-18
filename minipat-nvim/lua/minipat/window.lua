-- ==============================================================================
-- Minipat Window Management Module
-- ==============================================================================

local M = {}

-- ==============================================================================
-- Window and Split Utilities
-- ==============================================================================

function M.get_opposite_split(split_dir)
  return split_dir == "v" and "h" or "v"
end

function M.get_optimal_split_direction()
  local win = vim.api.nvim_get_current_win()
  local width = vim.api.nvim_win_get_width(win)
  local height = vim.api.nvim_win_get_height(win)
  return width > height and "v" or "h"
end

function M.get_split_direction(config_split)
  if config_split == nil then
    return M.get_optimal_split_direction()
  end
  return config_split
end

function M.create_split_command(split_dir, equal_size)
  local base_cmd = split_dir == "v" and "vsplit" or "split"
  if equal_size then
    local win = vim.api.nvim_get_current_win()
    local size
    if split_dir == "v" then
      size = math.floor(vim.api.nvim_win_get_width(win) / 2)
      return size .. "vsplit"
    else
      size = math.floor(vim.api.nvim_win_get_height(win) / 2)
      return size .. "split"
    end
  end
  return base_cmd
end

-- ==============================================================================
-- Group Window Management
-- ==============================================================================

function M.get_group_buffers(state)
  local buffers = {}
  for _, subprocess in pairs(state.subprocesses) do
    if subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      table.insert(buffers, subprocess.buffer)
    end
  end
  return buffers
end

function M.create_group_split(state, args)
  local win_valid = false
  if state.group_win then
    local ok, valid = pcall(vim.api.nvim_win_is_valid, state.group_win)
    win_valid = ok and valid
  end

  if state.group_hidden or (state.group_win and win_valid) then
    return state.group_win
  end

  local subprocess_split_dir = M.get_split_direction(args.config.split)
  local opposite_dir = M.get_opposite_split(subprocess_split_dir)
  state.group_split_dir = opposite_dir

  local split_cmd = M.create_split_command(opposite_dir, true)
  pcall(vim.cmd, split_cmd)
  state.group_win = vim.api.nvim_get_current_win()

  return state.group_win
end

function M.ensure_group_visible(state, args)
  if state.group_hidden then
    return false
  end

  local win_valid = false
  if state.group_win then
    local ok, valid = pcall(vim.api.nvim_win_is_valid, state.group_win)
    win_valid = ok and valid
  end

  if not state.group_win or not win_valid then
    M.create_group_split(state, args)
  end

  return true
end

-- ==============================================================================
-- Subprocess Visibility Management
-- ==============================================================================

function M.toggle_subprocess_visibility(name, state, args, start_component_fn)
  local subprocess = state.subprocesses[name]
  if not subprocess or not subprocess.buffer or not vim.api.nvim_buf_is_valid(subprocess.buffer) then
    start_component_fn(name, args)
    return "started"
  end

  local wins = vim.fn.win_findbuf(subprocess.buffer)
  if #wins > 0 then
    for _, win in ipairs(wins) do
      if vim.api.nvim_win_is_valid(win) then
        vim.api.nvim_win_close(win, false)
      end
    end
    return "hidden"
  else
    if state.group_hidden then
      state.group_hidden = false
    end
    local original_win = vim.api.nvim_get_current_win()
    if M.ensure_group_visible(state, args) then
      local group_buffers = M.get_group_buffers(state)
      local visible_buffers = {}
      for _, buf in ipairs(group_buffers) do
        local wins = vim.fn.win_findbuf(buf)
        if #wins > 0 then
          table.insert(visible_buffers, buf)
        end
      end

      vim.api.nvim_set_current_win(state.group_win)
      if #visible_buffers > 0 then
        local split_dir = M.get_split_direction(args.config.split)
        vim.cmd(M.create_split_command(split_dir, true))
      end
      vim.api.nvim_set_current_buf(subprocess.buffer)
    end
    vim.api.nvim_set_current_win(original_win)
    return "shown"
  end
end

function M.hide_all_except(keep_component, state, args, start_component_fn, components)
  local hidden_count = 0
  for _, component in ipairs(components) do
    if component ~= keep_component then
      local subprocess = state.subprocesses[component]
      if subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
        local wins = vim.fn.win_findbuf(subprocess.buffer)
        if #wins > 0 then
          for _, win in ipairs(wins) do
            if vim.api.nvim_win_is_valid(win) then
              vim.api.nvim_win_close(win, false)
            end
          end
          hidden_count = hidden_count + 1
        end
      end
    end
  end

  local kept_subprocess = state.subprocesses[keep_component]
  if kept_subprocess and kept_subprocess.buffer and vim.api.nvim_buf_is_valid(kept_subprocess.buffer) then
    local wins = vim.fn.win_findbuf(kept_subprocess.buffer)
    if #wins == 0 then
      M.toggle_subprocess_visibility(keep_component, state, args, start_component_fn)
    end
  else
    start_component_fn(keep_component, args)
  end

  return hidden_count
end

function M.show_all_started(state, args, start_component_fn, components)
  local shown_count = 0
  for _, component in ipairs(components) do
    local subprocess = state.subprocesses[component]
    if subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      local wins = vim.fn.win_findbuf(subprocess.buffer)
      if #wins == 0 then
        local result = M.toggle_subprocess_visibility(component, state, args, start_component_fn)
        if result == "shown" then
          shown_count = shown_count + 1
        end
      end
    end
  end
  return shown_count
end

function M.show_group(state, args, start_component_fn)
  if not state.group_hidden then
    return
  end

  for _, component in ipairs(state.previously_visible_components) do
    local subprocess = state.subprocesses[component]
    if subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      local wins = vim.fn.win_findbuf(subprocess.buffer)
      if #wins == 0 then
        M.toggle_subprocess_visibility(component, state, args, start_component_fn)
      end
    end
  end
end

function M.hide_group(state, components)
  if state.group_hidden then
    return
  end

  -- Remember which components are currently visible
  state.previously_visible_components = {}
  for _, component in ipairs(components) do
    local subprocess = state.subprocesses[component]
    if subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      local wins = vim.fn.win_findbuf(subprocess.buffer)
      if #wins > 0 then
        table.insert(state.previously_visible_components, component)
      end
    end
  end

  -- Find all windows displaying group buffers and close them
  local group_buffers = M.get_group_buffers(state)
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

function M.toggle_all_buffers(state, args, start_component_fn, components)
  local any_visible = false
  local buffers = M.get_group_buffers(state)

  for _, buf in ipairs(buffers) do
    local wins = vim.fn.win_findbuf(buf)
    if #wins > 0 then
      any_visible = true
      break
    end
  end

  if any_visible or (state.group_win and vim.api.nvim_win_is_valid(state.group_win)) then
    M.hide_group(state, components)
    return "hidden"
  else
    M.show_group(state, args, start_component_fn)
    return "shown"
  end
end

return M
