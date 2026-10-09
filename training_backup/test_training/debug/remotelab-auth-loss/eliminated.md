# Eliminated hypotheses

## Git pull or `git worktree` directly removed the credential

- The main repository's last reflog update was 2026-07-13, not the failure day.
- The new worktree was created at 12:03 and committed at 12:10.
- The recorded RemoteLab run contains 246 tool-call records; none mention `auth.json`, `codex logout`, `.codex/auth`, `unlink`, or a destructive command targeting `.codex`.
- The resulting commit changes only Feishu connector/session and adapter files; it contains no Codex authentication/storage change.
- Git worktree metadata and checkout paths are outside `/home/ubuntu/.codex`.

Conclusion: the Git/worktree operation is temporally adjacent but not a direct file-deletion path.

## RemoteLab smoke tests explicitly removed the personal credential

- The invoked smoke suite uses temporary homes for runtime/auth tests and fake Codex binaries where a Codex process is required.
- `test-chat-instance-sync` uses a `mkdtemp` source and target; its cleanup is confined to that temporary root.
- `test-runtime-policy` sets a temporary `HOME` before dynamically importing the runtime policy and removes only that temporary tree.
- Repository-wide search found no `codex logout` implementation or test command.

Conclusion: no invoked test has a direct path to `/home/ubuntu/.codex/auth.json`.

## RemoteLab started reading a different credential path

- The systemd service has `HOME=/home/ubuntu` and no `CODEX_HOME`, `REMOTELAB_MACHINE_CODEX_HOME`, or credential-store override.
- User config has no `cli_auth_credentials_store` override, so file storage remains the default.
- The managed RemoteLab path is still the same symlink to `/home/ubuntu/.codex/auth.json` and its symlink mtime remains 2026-06-16.
- No system-level forced login/workspace configuration exists.

Conclusion: this was not a worktree-dependent path switch.

## The 2026-07-20 formula release restart directly removed Codex auth

- The transient unit ran `restart-formula-release.sh`; that script waits for the active runner, sleeps, restarts the two RemoteLab services, checks health, and writes a restart log.
- The release run that created and launched the script is preserved under RemoteLab session `5aa249cd93245b81b169564341411a80` / run `run_5a8e160bc139e5b31b06c56e`.
- Every recorded RemoteLab tool call between 04:30 and 05:18 UTC was searched for `auth.json`, `codex logout`, `.codex/auth`, `CODEX_HOME`, instance sync, and destructive `.codex` operations; there were no matches.
- The restart and cutover scripts, installed unit files, and runtime auth route contain no credential deletion operation.
- A new Codex subprocess started by the restarted service failed with 401 at 05:17:50, roughly two seconds after the service came up.

Conclusion: the restart exposed an already absent or unusable on-disk credential to a fresh process; it did not directly delete it.
