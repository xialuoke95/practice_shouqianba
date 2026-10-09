# RemoteLab dual Feishu bot investigation

## Symptom

Two Feishu bot connector instances exist for RemoteLab; one works, while the other does not appear to receive and process messages normally.

## Observed conditions

- Occurs: one of the two Feishu bot instances; exact instance not established yet.
- Does NOT occur: the other bot instance operates normally.
- First observed: reported by the user on 2026-09-07.
- Last known good state: not established.
- Investigation budget: 15 iterations (skill default).

## Iteration log

### Iteration 0 - initial observation

- Both connector units are active, have stable PIDs, and report `NRestarts=0` since `12:29:28 UTC`.
- Both reached `persistent connection ready (feishu-cn)` and `ws client ready`.
- Default bot uses app `cli_a97c...`; bot-2 uses a distinct app `cli_aae9...`. Both point to the healthy RemoteLab origin at `127.0.0.1:7690` and both use `allow_all` intake.
- Bot-2 has an additional per-chat rule: chat `oc_73e8...` is `mention_only`.
- Since recovery, bot-2 received four new allowed events at `12:40:03`, `12:58:42`, `13:01:04`, and `13:03:58 UTC`; default bot only replayed two pre-recovery events at startup and its event file has not advanced since `12:29:30 UTC`.
- Operational status reports eight recent silent bot-2 group messages, while mentioned messages and P2P messages are visible as inbound events.

### Iteration 1 - live message-to-run correlation

- Default bot (`cli_a97c...`, display mention `Lv.6上升气流`) received a new mentioned group message at `13:06:39 UTC` and completed run `run_ae4365933950fc0f72890014` at `13:06:47 UTC` with assistant text persisted.
- Bot-2 (`cli_aae9...`, display mention `Lv.0 茵蒂克丝`) received a P2P `hello` at `13:03:58 UTC` and completed run `run_6e00229ae54f6f82d0e303ed` at `13:04:05 UTC` with assistant text persisted.
- Therefore neither bot has a Feishu long-connection failure or a RemoteLab execution failure.

### Iteration 2 - bot-specific intake behavior

- Bot-2 config sets global group mode to `all`, but overrides chat `oc_73e827386be3d4ec48206b092f0dae05` to `mention_only`.
- The focused test `node tests/test-feishu-group-routing.mjs` passes and logs the expected skip for unmentioned messages.
- This accounts for the bot-specific observation that ordinary group messages appear not to be received or processed. P2P messages and messages that mention bot-2 are admitted.

### Iteration 3 - shared reply-publication regression

- Both latest successful normal requests have `result.state=completed` but `deliveries=[]`, so the model result exists while no Feishu outbox item is created.
- The connector's durable inbox copy proves it submitted a valid `sourceDelivery` object containing the correct `chatId`, `messageId`, and `sourceRouteId`.
- The running `remotelab.service` started at `12:28:24 UTC`. The code fix `27bcf1a4` (`fix(http): preserve connector reply destinations on message admission`) was committed later at `12:56:18 UTC`, so the live Node process has not loaded it.
- An idempotent replay of bot-2's existing message returned `duplicate=true` against the old fingerprint, confirming the live process still discards `sourceDelivery` during JSON request parsing.
- The isolated regression test `node tests/test-http-session-source-delivery.mjs` passes on the current checkout and proves the committed code preserves the destination through request admission and creates a pending outbox delivery.

## Root cause

There are two independent effects:

1. Bot-2 intentionally ignores unmentioned messages in one configured group because of its `mention_only` override.
2. For admitted messages on both bots, the live RemoteLab server is still running pre-fix code that drops `sourceDelivery`; runs finish, but replies are not queued for Feishu. The fix is already present and tested in the checkout, but the server has not been restarted since that commit.

## Reproduction and verification

- Reproduction: inspect recent connector inbox submission (contains `sourceDelivery`) and matching archived RemoteLab request (does not contain it and has `deliveries=[]`).
- Live confirmation: idempotently replay the same request id; the server reports `duplicate=true`, matching the pre-fix fingerprint.
- Fixed-code verification: `node tests/test-http-session-source-delivery.mjs` passes and verifies JSON plus multipart admission and outbox creation.
- Routing verification: `node tests/test-feishu-group-routing.mjs` passes and verifies `mention_only` filtering.

## Corrective action not performed

The user requested inspection/diagnosis. No service restart, configuration change, message backfill, or external Feishu send was performed. To activate the already committed fix, restart `remotelab.service` and verify one new mentioned/P2P message for each bot produces a delivered source-delivery receipt. Decide separately whether bot-2's per-chat `mention_only` rule should remain.
