# PR Worker

Create a pull request from a worker's branch: $ARGUMENTS

## Process

1. Identify the worker session or branch from $ARGUMENTS
   - Can be worker name (e.g., "Groucho") or branch name (e.g., "cic-abc-feature")

2. If worker name provided:
   - Get worker info via `list_workers` or `examine_worker`
   - Find the worktree/branch from worktree_path

3. Detect the parent branch (branch the worktree diverged from):
   ```bash
   git merge-base --fork-point main <branch> || \
   git merge-base --fork-point $(git branch --show-current) <branch>
   ```
   Or check git reflog for the branch point. If unclear, ask user.

4. Gather PR information:
   - Get commits on branch: `git log <parent>..<branch> --oneline`
   - Get changed files: `git diff <parent>..<branch> --stat`
   - Extract issue ID from branch name if present

5. Push branch if not already pushed:
   ```bash
   git push -u origin <branch>
   ```

6. Generate an architecture diagram:
   - Create a self-contained HTML file that visually documents the PR's architecture
   - Include: module structure, data flow, key types/interfaces, integration points
   - Use a dark theme (`#0d1117` background), styled for readability
   - Tailor the content to the PR — show what's most useful for reviewers:
     - For new modules: schema, module layout, consumer map
     - For refactors: before/after structure, migration paths
     - For features: data flow, integration points, config/API surface
   - Write to `/tmp/pr-architecture-<branch>.html`
   - Upload as a public gist:
     ```bash
     gh gist create /tmp/pr-architecture-<branch>.html --public -d "<PR title> — Architecture"
     ```
   - Build the preview URL: `https://gist.githack.com/<user>/<gist-id>/raw/<filename>.html`

7. Create PR using gh CLI (targeting parent branch), including the architecture link:
   ```bash
   gh pr create --base <parent-branch> --title "<issue-id>: <summary>" --body "$(cat <<'EOF'
   ## Summary
   <bullet points from commits>

   📐 **[Architecture Diagram](<githack-preview-url>)** — visual overview of this PR's design.

   ## Changes
   <list of changed files>

   ## Testing
   - [ ] Tests pass
   - [ ] Manual verification

   ---
   Related: <issue-id>
   EOF
   )"
   ```

8. Report the PR URL

## Output Format

```
## Pull Request Created

**Branch:** cic-abc-feature-name
**PR:** https://github.com/org/repo/pull/42
**Title:** cic-abc: Implement feature X

### Commits
- <sha> Add new endpoint
- <sha> Update tests

### Files Changed
- src/api/endpoint.py
- tests/test_endpoint.py

**Next steps:**
- Review PR at <url>
- After merge, run `/cleanup-worktrees` to remove worktree
```

## Notes

- Requires `gh` CLI to be authenticated
- Branch must have commits ahead of parent
- Does not close the worker session
- **Note:** `close_workers` removes the worktree directory but keeps the branch (commits are preserved). You can safely close workers after pushing — the branch remains for the PR

## If Small Change

**Stop and notify the user** if the change is trivial (single file, minor fix). Let them know they may want to use `/merge-worker` instead for a direct merge without PR overhead.

Allow the user flexibility to proceed with PR if they prefer — this is a suggestion, not a hard block.
