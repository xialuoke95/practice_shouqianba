# Feishu connector to RemoteLab failure investigation

## Symptom

At 19:13 China Standard Time on 2026-07-17, a Feishu group message asked the bot to inspect prior memory. At 19:14, the bot returned: `我收到了你的消息，但这次生成回复失败了。你可以稍后再发一次。`

The connector journal reports `Error: reply publication failed` for messages received after its 11:12 UTC restart.

## Observed conditions

- Occurs: inbound Feishu group messages are accepted after the connector restart, but generated-reply publication fails.
- Does not occur: the fallback error message is visibly delivered to Feishu, so inbound delivery and at least one outbound reply path work.
- First observed in current evidence: 2026-07-17 11:13 UTC (19:13 CST).
- Last known good evidence: earlier on 2026-07-17, the connector logged inbound events and successful replies.
- Investigation budget: 15 iterations (skill default).

## Iteration log

### Iteration 0 - initial observation

- Screenshot time aligns with the first `reply publication failed` event after the Feishu connector restarted at 11:12 UTC.
- The RemoteLab Web service and local build-info endpoint were healthy during the same period.

### Iteration 1 - correlate Feishu message to RemoteLab run

- Inbound message: `om_x100b6a94aa8e18a0c1b4908078ae3b6` at `2026-07-17T11:13:08Z`.
- RemoteLab session: `48c730cda5a3cfe4bd8e8828db87bdcd`.
- RemoteLab run: `run_8972f45109204d859b2f2b20`.
- The run ended `failed`, exit code 1, at `2026-07-17T11:13:28.514Z`.
- Its reply publication state became `failed`; the connector then threw `reply publication failed` at line 2284.

### Iteration 2 - inspect provider events

- The Codex thread started successfully locally.
- WebSocket attempts to `wss://api.openai.com/v1/responses` returned HTTP 401 with `Missing bearer or basic authentication in header`.
- Codex fell back to HTTPS; all HTTPS attempts returned the same HTTP 401.
- RemoteLab classified the terminal failure as `Provider requires interactive login before RemoteLab can use it`.

### Iteration 3 - verify authentication sources

- `codex login status` returned `Not logged in`.
- `env CODEX_HOME=/home/ubuntu/.config/remotelab/provider-runtime-homes/codex codex login status` also returned `Not logged in`.
- `/home/ubuntu/.codex/auth.json` does not exist.
- The managed runtime `auth.json` is a symlink to `/home/ubuntu/.codex/auth.json`, so it is currently broken.
- The live RemoteLab service environment has neither `OPENAI_API_KEY` nor `CHATGPT_API_KEY`; therefore there is no alternate credential source.
- Last known good Feishu Codex run started at `11:12:22Z`; the first failing run started at `11:13:11Z`. Authentication became unavailable across that boundary.

## Root Cause (Confirmed)

RemoteLab receives and submits Feishu messages correctly, but its spawned Codex CLI has no usable OpenAI credential. The personal Codex auth file is missing, the managed runtime auth link points to that missing file, and no API-key environment variable is present. Codex therefore sends no Authorization header, receives HTTP 401, exits 1, and causes reply publication to become `failed`.

## Evidence

- Four consecutive Feishu runs failed with the same provider-login classification.
- The exact screenshot run contains repeated `401 Unauthorized: Missing bearer or basic authentication in header` events over both WebSocket and HTTPS.
- Both relevant `codex login status` checks report `Not logged in`.
- The fallback Feishu notice succeeds, disproving a general Feishu outbound failure.

## Reproduction case

```bash
codex login status
env CODEX_HOME=/home/ubuntu/.config/remotelab/provider-runtime-homes/codex codex login status
```

Both commands deterministically return `Not logged in` in the failing environment.

## Proposed fix

Log Codex in again as user `ubuntu` so `/home/ubuntu/.codex/auth.json` is recreated. The existing managed-runtime symlink will then resolve to the restored credential. Verify both login-status commands and send a new Feishu message; the services do not inherently need a restart.

The present evidence does not identify which process removed or invalidated the personal auth file. No `codex logout` entry was found in the checked shell/Codex histories.

## Next step

If authorized, perform the interactive Codex login checkpoint, verify both auth contexts, and run one new Feishu end-to-end message.
