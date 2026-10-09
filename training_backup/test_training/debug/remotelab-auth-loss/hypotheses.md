# Active hypotheses

## H-1: A RemoteLab update or test script deletes the personal Codex auth file

The pull/worktree workflow may execute cleanup logic against a wrongly resolved `CODEX_HOME`, causing `/home/ubuntu/.codex/auth.json` to be removed.

- Test: search all active/recent RemoteLab worktrees, diffs, scripts, and tests for writes/unlinks/removals involving `auth.json`, `CODEX_HOME`, or `codex logout`.
- Falsifier: no shipped or invoked path targets the personal auth file, and execution history contains no such cleanup command.
- Status: eliminated for the observed worktree/update run
- Confidence: very low

## H-2: A Codex client or agent explicitly logged out while sharing the same credential store

Another CLI, app, extension, or automated task may have run `codex logout`, which removes the shared cached credential used by both normal Codex and RemoteLab.

- Test: inspect Codex login diagnostics, local session tool calls, shell history, and process/service logs for logout or credential-clear operations around the failure boundary.
- Falsifier: no logout/clear event exists and filesystem evidence identifies another writer.
- Evidence for: OpenAI's Codex manual confirms CLI and IDE share cached login state and logout clears the shared credentials; several Codex surfaces/processes use this host account.
- Evidence against: shell history and all recorded RemoteLab tool calls in the failure window contain no `codex logout` or auth-file removal command.
- Status: survives, but the exact initiating surface is unproven
- Confidence: medium-low

## H-3: The credential path changed rather than the file being deleted

A worktree or service update may have changed `CODEX_HOME` or credential-store mode, making the existing credential invisible.

- Test: compare current/past environment, config history, service environment, and worktree-specific config for `CODEX_HOME`, `CODEX_SQLITE_HOME`, and `cli_auth_credentials_store` changes.
- Falsifier: the failed process used the same default `~/.codex` path and that path was proven absent.
- Status: eliminated
- Confidence: very low

## H-4: Login invalidation is caused by an external credential lifecycle event

The ChatGPT session/token may have been revoked or invalidated, after which Codex removed or stopped using the cached auth file independently of Git operations.

- Test: inspect login diagnostics and failure shape for refresh/revocation errors preceding file loss; compare with the observed `Missing bearer` error.
- Falsifier: a local deletion command or code path is positively identified.
- Evidence against: the first fresh login worked for multiple RemoteLab calls, then the error became `Missing bearer`, and the current fresh access token has a ten-day lifetime. This does not resemble normal expiry.
- Status: weakened; no refresh/revocation evidence was found
- Confidence: low

## H-5: The recurrent outage is enabled by RemoteLab and interactive Codex sharing one mutable credential file

RemoteLab defaults to the personal machine Codex home and its managed auth path is a symlink to `/home/ubuntu/.codex/auth.json`. Therefore any logout, credential cleanup, or file loss in any Codex surface immediately removes authentication from RemoteLab as well.

- Test: inspect service environment, runtime policy, resolved symlink, and Codex credential-store documentation.
- Falsifier: RemoteLab uses an independently authenticated `CODEX_HOME` with its own non-symlink credential store.
- Status: confirmed architectural root cause for recurrence and blast radius
- Confidence: high

## H-6: Concurrent Codex activity triggered an internal credential-store race or cleanup

Between the successful 11:51 login and the first 12:11 401, RemoteLab launched several overlapping Codex runs plus background summarizers, all against the same personal credential store.

- Test: find a Codex internal log, filesystem audit record, or reproducible multi-process sequence that deletes the file.
- Falsifier: an explicit logout/delete actor is identified instead.
- Status: survives only as a trigger hypothesis; no positive confirmation
- Confidence: low

## H-7: The 2026-07-20 formula release restart directly deleted Codex auth

The production cutover restarted `remotelab.service` and `remotelab-feishu.service` at the same boundary where the next Codex subprocess first failed.

- Test: inspect the transient systemd unit, restart script, cutover script, release-run tool history, unit files, and every RemoteLab tool command from 04:30 through 05:18 UTC.
- Falsifier: all invoked release paths only restart services and none target `~/.codex`, `auth.json`, `CODEX_HOME`, or `codex logout`.
- Evidence against: no release script, systemd unit action, or recorded tool command in the window touches the credential; the first new process after restart merely observed unauthenticated state.
- Status: eliminated as a direct deletion path; the restart is an observation boundary
- Confidence: very low
