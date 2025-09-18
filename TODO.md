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

Next available task ID: {#15}

### Active

- {#13} Reflow windows when hiding/showing {nvim}

### Review


### Upcoming

- {#14} Add cheatsheet param to config for custom cheatsheet path (default nil) {nvim}

### Done

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


