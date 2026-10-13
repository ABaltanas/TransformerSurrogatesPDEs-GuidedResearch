---
name: auto-commit
description: Automated Git commit workflow using Conventional Commits. Use whenever the user asks to "make a commit", "commit changes", or "haz commit".
---

# Auto Commit Instructions

You act as a Git workflow automator. When this skill is active, you MUST execute the following sequence:

1. **Reset Staging Area**:
   Ensure no files are currently staged by executing:
   `git reset`

2. **Inspect Changes**:
   Inspect all modified, untracked, and deleted files by executing:
   `git status -s`
   `git diff`

3. **Logical Grouping & Rules**:
   Group unstaged files by intent. Every commit message MUST strictly adhere to Conventional Commits: `<type>(<scope>): <description>`.

   Allowed types:
   - `feat`: New feature or architecture.
   - `fix`: Bug fix, tensor dimension error, numerical instability.
   - `docs`: Documentation only (README, docstrings, markdown notes).
   - `style`: Formatting, whitespace (no code behavior change).
   - `refactor`: Restructuring code without fixing a bug or adding features.
   - `perf`: Performance or memory optimization.
   - `test`: Adding or updating test cases.
   - `chore`: Build scripts, dependencies, submodule pointers.

   Safety constraints:
   - Never commit files inside a submodule directory. Only stage submodule pointer changes when explicitly requested.
   - If multiple unrelated changes exist, separate them into individual commits.

4. **Stage and Commit**:
   For each logical group:
   1. Stage the files: `git add <file1> <file2>`
   2. Execute the commit: `git commit -m "<type>(<scope>): <description>"`