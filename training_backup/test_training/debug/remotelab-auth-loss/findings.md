# Repeated Codex auth loss after RemoteLab update/worktree activity

## Symptom

`/home/ubuntu/.codex/auth.json` disappeared again after recent RemoteLab code-update and worktree operations. `codex login status` returned `Not logged in`, RemoteLab Codex runs failed with HTTP 401, and the user had to authenticate again.

## Observed conditions

- The file was absent during the first investigation around 11:13 UTC on 2026-07-17.
- Codex login diagnostics show successful device logins at `11:51:13` and again at `13:20:00` UTC. This proves the credential became unusable a second time after the first successful re-login.
- The first re-login was followed by successful RemoteLab Codex/summarizer calls at 11:52 and 11:56.
- The first post-login failure was at `12:11:55`: new RemoteLab Codex processes received HTTP 401 `Missing bearer or basic authentication in header`.
- The user reauthenticated again; the recreated file now has mtime `2026-07-17 13:20:00 UTC`, mode `600`, owner `ubuntu`.
- The managed RemoteLab auth path remains a symlink to `/home/ubuntu/.codex/auth.json`.
- Current `codex login status` reports `Logged in using ChatGPT`.
- Main RemoteLab worktree reflog shows its last checkout/merge activity on 2026-07-13, while another active worktree exists under `/home/ubuntu/.remotelab/workspace/remotelab-feishu-v2-merge`.
- Investigation budget: 15 iterations.
- A third re-login was required on 2026-07-20; OAuth completed at `10:41:58 UTC` and recreated `/home/ubuntu/.codex/auth.json`.

## Iteration log

### Iteration 0 - initial state capture

- Confirmed this is a real file-loss/recreation event, not merely a currently broken login.
- No repository or service mutations performed.

### Iteration 1 - authentication timeline

- `~/.codex/log/codex-login.log` records successful OAuth exchanges at 11:51:13 and 13:20:00.
- The second login was not merely refreshing an expired token: a normal freshly issued token on this host has a ten-day access-token lifetime.
- The 12:11 error explicitly says the request carried no bearer/basic authentication, which is consistent with an absent/unavailable file rather than normal token expiry.

### Iteration 2 - worktree and command provenance

- The Feishu migration run created `/home/ubuntu/.remotelab/workspace/remotelab-feishu-v2-merge` at 12:03, ran `npm ci --ignore-scripts`, targeted tests and smoke tests, and committed `721a9fd` at 12:10:51.
- All 246 recorded tool-call entries for that run were checked. There was no auth-file operation, `codex logout`, unlink, or destructive `.codex` command.
- The migration commit changes 16 Feishu/session/adapter/test files and does not modify Codex auth/storage code.

### Iteration 3 - source path analysis

- `chat/runtime-policy.mjs` defaults the machine Codex home to the personal `~/.codex` home.
- Even the RemoteLab-managed Codex home links its `auth.json` back to the personal auth source for the owner instance.
- The unlink in `ensureSymlinkOrCopy` targets only the managed target path, not the personal source path.
- Guest redaction, migration, and instance-sync helpers copy/write guest or target credentials; none were invoked against the personal auth path in the failure window.

### Iteration 4 - system evidence limits

- Shell history contains the re-login command but no logout.
- RemoteLab API logs contain no request to the new Codex device-login/auth routes.
- No relevant cron timer or system-level forced-login configuration exists.
- Linux audit/process accounting was not installed/enabled, so the historical unlinking PID cannot be reconstructed after the fact.

### Iteration 5 - recurrence on 2026-07-20

- The next observed failure began at `05:17:50 UTC`, again as HTTP 401 from newly spawned Codex processes.
- Login diagnostics record another successful device login at `10:41:58 UTC`; the recreated file has that same mtime, mode `600`, and owner `ubuntu`.
- The new access token was issued at 10:41:58 and expires ten days later. The prior credential was only about three days old, so ordinary access-token expiry does not explain the event.
- After the 10:41 re-login, `codex login status` reports `Logged in using ChatGPT` and no further RemoteLab 401s appear in the service journal through the verification time.

### Iteration 6 - release restart boundary

- `remotelab.service` and `remotelab-feishu.service` were deliberately restarted by transient unit `remotelab-formula-release-restart.service` at 05:17:48-05:17:50; the host itself did not reboot.
- The restart script and earlier cutover script only install/restart services and perform health checks. They do not read, write, unlink, copy, or move Codex auth.
- The release run and all RemoteLab tool commands in the surrounding window contain no `codex logout`, `auth.json`, `CODEX_HOME`, instance-sync, or destructive `.codex` command.
- A fresh summarizer process started immediately after restart and failed at 05:17:50. Therefore the restart is the first reliable observation point for the missing/unusable disk credential, not a proven deletion event.

### Iteration 7 - last known working process and masking behavior

- RemoteLab successfully launched fresh Codex summarizer activity at 05:05 and continued an interactive Codex-backed run from 05:14 through 05:17.
- Codex auth is cached per process. A process that loaded credentials before the file disappeared can continue until it needs a reload or is restarted, which explains why service operation can look healthy before a restart exposes the missing disk state.
- The loss window is narrower than the service uptime: after the 05:14 run loaded auth and before the restarted service's 05:17:50 request.

### Iteration 8 - deletion-path and product-behavior review

- RemoteLab's `/api/codex-auth/status` route only runs `codex login status`; its device-login route only starts `codex login --device-auth`. Neither route logs out or deletes credentials.
- Current production runtime defaults `CODEX_HOME` to `/home/ubuntu/.codex`, and the managed RemoteLab auth path resolves to the same personal file. No auth lock or independent credential home is configured.
- Recorded shell history, local Codex tool calls, RemoteLab tool calls, and the system journal contain no explicit logout or policy-enforced logout in the failure interval.
- Codex documentation confirms CLI and IDE surfaces share cached credentials, and a logout from either requires reauthentication. Codex source shows explicit logout and forced login/workspace restriction paths delete persisted auth; no forced restriction is configured here.

## Conclusion

### Confirmed root cause of recurrence

RemoteLab and interactive Codex are not credential-isolated. They share one mutable file, `/home/ubuntu/.codex/auth.json`; the managed RemoteLab path is only a symlink to it. Any logout or credential cleanup by any Codex surface immediately breaks RemoteLab. This is the confirmed structural cause and explains why the same outage can recur.

### Trigger attribution

The Git pull/worktree operation and the 2026-07-20 release restart did not directly delete the file. The restart instead forced new processes to reload the shared on-disk credential and revealed the broken state immediately. The exact actor that removed it is not recoverable from the current logs because filesystem auditing was absent. The remaining plausible triggers are a logout/credential-clear action from another Codex surface or an internal Codex credential-store cleanup/race while multiple Codex processes shared the same store. Neither is positively proven.

## Recommended remediation

1. Give RemoteLab a dedicated `CODEX_HOME` and authenticate that home directly; do not symlink its `auth.json` to the user's interactive `~/.codex/auth.json`.
2. Add filesystem auditing/inotify monitoring for both auth paths before the next incident so the deleting PID and executable are captured.
3. Avoid treating a Git/worktree switch as the repair. It does not address the shared credential single point of failure.
