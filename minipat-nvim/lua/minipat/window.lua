-- ==============================================================================
-- Minipat Window Management Module
-- ==============================================================================

local M = {}

-- Safe window navigation
local function safe_set_current_win(win)
  if win and vim.api.nvim_win_is_valid(win) then
    vim.api.nvim_set_current_win(win)
    return true
  end
  return false
end

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
  -- If width > height (wide screen), use horizontal split to stack windows vertically
  -- If height > width (tall screen), use vertical split for side-by-side windows
  return width > height and "h" or "v"
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
-- Window Reflow and Equalization
-- ==============================================================================

function M.reflow_windows_in_group(subprocesses)
  -- Get all windows displaying group buffers
  local group_buffers = M.get_group_buffers(subprocesses)
  if #group_buffers == 0 then
    return
  end

  local group_windows = {}
  for _, buf in ipairs(group_buffers) do
    local wins = vim.fn.win_findbuf(buf)
    for _, win in ipairs(wins) do
      if vim.api.nvim_win_is_valid(win) then
        table.insert(group_windows, win)
      end
    end
  end

  if #group_windows <= 1 then
    return -- No need to reflow with 0 or 1 window
  end

  -- Save current window to restore later
  local current_win = vim.api.nvim_get_current_win()

  -- Find a valid window in the group to work from
  local valid_group_win = nil
  for _, win in ipairs(group_windows) do
    if vim.api.nvim_win_is_valid(win) then
      valid_group_win = win
      break
    end
  end

  if not valid_group_win then
    return
  end

  -- Use Neovim's built-in window equalization
  -- This handles complex nested layouts better than manual resizing
  vim.api.nvim_set_current_win(valid_group_win)

  -- First, use wincmd = to equalize all windows
  vim.cmd("wincmd =")

  -- Then explicitly set equal widths for vertical splits
  -- This ensures windows are properly equalized even in complex layouts
  vim.schedule(function()
    if #group_windows > 1 then
      local wins_still_valid = {}
      for _, win in ipairs(group_windows) do
        if vim.api.nvim_win_is_valid(win) then
          table.insert(wins_still_valid, win)
        end
      end

      if #wins_still_valid > 1 then
        -- Get the total width available
        local total_width = 0
        for _, win in ipairs(wins_still_valid) do
          total_width = total_width + vim.api.nvim_win_get_width(win)
        end

        -- Set equal width for all windows
        local target_width = math.floor(total_width / #wins_still_valid)
        for i, win in ipairs(wins_still_valid) do
          if i < #wins_still_valid then  -- Don't resize the last one to avoid rounding issues
            pcall(vim.api.nvim_win_set_width, win, target_width)
          end
        end
      end
    end
  end)

  -- Restore original window if still valid
  if vim.api.nvim_win_is_valid(current_win) then
    vim.api.nvim_set_current_win(current_win)
  end
end

function M.equalize_group_windows(win_state)
  -- Use vim's built-in equalize functionality as a fallback
  if win_state.group_win and vim.api.nvim_win_is_valid(win_state.group_win) then
    local current_win = vim.api.nvim_get_current_win()
    vim.api.nvim_set_current_win(win_state.group_win)
    vim.cmd("wincmd =")
    if vim.api.nvim_win_is_valid(current_win) then
      vim.api.nvim_set_current_win(current_win)
    end
  end
end

-- ==============================================================================
-- Group Window Management
-- ==============================================================================

function M.get_group_buffers(subprocesses)
  local buffers = {}
  for _, subprocess in pairs(subprocesses) do
    if subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      table.insert(buffers, subprocess.buffer)
    end
  end
  return buffers
end

function M.create_group_split(win_state, args)
  local win_valid = false
  if win_state.group_win then
    local ok, valid = pcall(vim.api.nvim_win_is_valid, win_state.group_win)
    win_valid = ok and valid
  end

  if win_state.group_hidden or (win_state.group_win and win_valid) then
    return win_state.group_win
  end

  local subprocess_split_dir = M.get_split_direction(args.config.split)
  local opposite_dir = M.get_opposite_split(subprocess_split_dir)
  win_state.group_split_dir = opposite_dir

  local split_cmd = M.create_split_command(opposite_dir, true)
  pcall(vim.cmd, split_cmd)
  win_state.group_win = vim.api.nvim_get_current_win()

  return win_state.group_win
end

function M.ensure_group_visible(win_state, args)
  if win_state.group_hidden then
    return false
  end

  local win_valid = false
  if win_state.group_win then
    local ok, valid = pcall(vim.api.nvim_win_is_valid, win_state.group_win)
    win_valid = ok and valid
  end

  if not win_state.group_win or not win_valid then
    M.create_group_split(win_state, args)
  end

  return true
end

-- ==============================================================================
-- Subprocess Visibility Management
-- ==============================================================================

function M.toggle_subprocess_visibility(name, subprocesses, win_state, args, start_component_fn)
  local subprocess = subprocesses[name]
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
    -- Reflow remaining windows after hiding
    vim.defer_fn(function()
      M.reflow_windows_in_group(subprocesses)
    end, 100)
    return "hidden"
  else
    if win_state.group_hidden then
      win_state.group_hidden = false
    end
    local original_win = vim.api.nvim_get_current_win()
    if M.ensure_group_visible(win_state, args) then
      local group_buffers = M.get_group_buffers(subprocesses)
      local visible_buffers = {}
      for _, buf in ipairs(group_buffers) do
        local wins = vim.fn.win_findbuf(buf)
        if #wins > 0 then
          table.insert(visible_buffers, buf)
        end
      end

      -- Find any visible window to split from
      local split_from_win = nil
      for _, buf in ipairs(visible_buffers) do
        local wins = vim.fn.win_findbuf(buf)
        if #wins > 0 and vim.api.nvim_win_is_valid(wins[1]) then
          split_from_win = wins[1]
          break
        end
      end

      if split_from_win then
        vim.api.nvim_set_current_win(split_from_win)
      elseif win_state.group_win and vim.api.nvim_win_is_valid(win_state.group_win) then
        vim.api.nvim_set_current_win(win_state.group_win)
      end

      if #visible_buffers > 0 then
        local split_dir = M.get_split_direction(args.config.split)
        vim.cmd(M.create_split_command(split_dir, true))
      end
      vim.api.nvim_set_current_buf(subprocess.buffer)
      -- Reflow windows after showing
      vim.defer_fn(function()
        M.reflow_windows_in_group(subprocesses)
      end, 100)
    end
    safe_set_current_win(original_win)
    return "shown"
  end
end

function M.hide_all_except(keep_component, subprocesses, win_state, args, start_component_fn, components)
  local hidden_count = 0
  for _, component in ipairs(components) do
    if component ~= keep_component then
      local subprocess = subprocesses[component]
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

  local kept_subprocess = subprocesses[keep_component]
  if kept_subprocess and kept_subprocess.buffer and vim.api.nvim_buf_is_valid(kept_subprocess.buffer) then
    local wins = vim.fn.win_findbuf(kept_subprocess.buffer)
    if #wins == 0 then
      M.toggle_subprocess_visibility(keep_component, subprocesses, win_state, args, start_component_fn)
    end
  else
    start_component_fn(keep_component, args)
  end

  -- Reflow windows after hiding all except the kept component
  if hidden_count > 0 then
    vim.defer_fn(function()
      M.reflow_windows_in_group(subprocesses)
    end, 50)
  end

  return hidden_count
end

function M.show_all_started(subprocesses, win_state, args, start_component_fn, components)
  local shown_count = 0
  for _, component in ipairs(components) do
    local subprocess = subprocesses[component]
    if subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      local wins = vim.fn.win_findbuf(subprocess.buffer)
      if #wins == 0 then
        local result = M.toggle_subprocess_visibility(component, subprocesses, win_state, args, start_component_fn)
        if result == "shown" then
          shown_count = shown_count + 1
        end
      end
    end
  end

  -- Reflow windows after showing all started components
  if shown_count > 0 then
    vim.defer_fn(function()
      M.reflow_windows_in_group(subprocesses)
    end, 100) -- Slightly longer delay since multiple windows may be created
  end

  return shown_count
end

function M.show_group(subprocesses, win_state, args, start_component_fn)
  if not win_state.group_hidden then
    return
  end

  local shown_count = 0
  for _, component in ipairs(win_state.previously_visible_components) do
    local subprocess = subprocesses[component]
    if subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      local wins = vim.fn.win_findbuf(subprocess.buffer)
      if #wins == 0 then
        local result = M.toggle_subprocess_visibility(component, subprocesses, win_state, args, start_component_fn)
        if result == "shown" then
          shown_count = shown_count + 1
        end
      end
    end
  end

  -- Reflow windows after showing the group
  if shown_count > 0 then
    vim.defer_fn(function()
      M.reflow_windows_in_group(subprocesses)
    end, 100)
  end
end

function M.hide_group(subprocesses, win_state, components)
  if win_state.group_hidden then
    return
  end

  -- Remember which components are currently visible
  win_state.previously_visible_components = {}
  for _, component in ipairs(components) do
    local subprocess = subprocesses[component]
    if subprocess and subprocess.buffer and vim.api.nvim_buf_is_valid(subprocess.buffer) then
      local wins = vim.fn.win_findbuf(subprocess.buffer)
      if #wins > 0 then
        table.insert(win_state.previously_visible_components, component)
      end
    end
  end

  -- Find all windows displaying group buffers and close them
  local group_buffers = M.get_group_buffers(subprocesses)
  for _, buf in ipairs(group_buffers) do
    local wins = vim.fn.win_findbuf(buf)
    for _, win in ipairs(wins) do
      if vim.api.nvim_win_is_valid(win) then
        vim.api.nvim_win_close(win, false)
      end
    end
  end

  -- Close the main group window if it exists
  if win_state.group_win and vim.api.nvim_win_is_valid(win_state.group_win) then
    vim.api.nvim_win_close(win_state.group_win, false)
  end

  win_state.group_hidden = true
  win_state.group_win = nil
end

function M.toggle_all_buffers(subprocesses, win_state, args, start_component_fn, components)
  local any_visible = false
  local buffers = M.get_group_buffers(subprocesses)

  for _, buf in ipairs(buffers) do
    local wins = vim.fn.win_findbuf(buf)
    if #wins > 0 then
      any_visible = true
      break
    end
  end

  if any_visible or (win_state.group_win and vim.api.nvim_win_is_valid(win_state.group_win)) then
    M.hide_group(subprocesses, win_state, components)
    return "hidden"
  else
    M.show_group(subprocesses, win_state, args, start_component_fn)
    return "shown"
  end
end

return M
