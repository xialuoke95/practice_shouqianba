# FriendLab 502 investigation

## Symptom

Public URL `https://friendlab.jiujianian-dev-world.win/?session=2081db746980c0414f7a558ed244cb07` returns HTTP 502 on 2026-09-07.

## Observed conditions

- Occurs: public request through Cloudflare.
- Does NOT occur: after the converted state cutover completed at 12:28 UTC; the same URL and public build-info endpoint are healthy.
- First observed: reported by the user on 2026-09-07.
- Last known good state: previously deployed; exact time not established.
- Investigation budget: 15 iterations (skill default).

## Iteration log

### Iteration 0 - initial observation

- At `2026-09-07 12:35:49 UTC`, the exact public URL returned HTTP 302 with `Location: /login`, `server: cloudflare`, and Ray ID `a375c1686825b298-PDX`.
- At the same time, `http://127.0.0.1:7690/` returned the same 302 redirect and the local build-info endpoint returned HTTP 200.
- RemoteLab, cloudflared, and the Feishu connector are currently active.
- RemoteLab entered the current active state at `12:28:24 UTC`, immediately after an active exit at `12:28:23 UTC`; cloudflared has stayed active since `2026-08-12 06:58:14 UTC`.
- Port 7690 is currently listening on loopback.

### Iteration 1 - correlate public 502 with origin availability

- Cloudflared recorded the exact reported session URL repeatedly between `08:46:02` and `09:58:21 UTC`.
- Every failing request has the same direct cause: `dial tcp 127.0.0.1:7690: connect: connection refused`.
- The tunnel remained active and routed ingress rule 0 to `http://localhost:7690`.
- This confirms H-1 and eliminates an edge/tunnel-connectivity failure.

### Iteration 2 - identify why RemoteLab had no listener

- RemoteLab was deliberately stopped at `08:15:09 UTC`; the stop timed out at `08:16:39 UTC`.
- It was started again at `08:34:57 UTC`, but the new process immediately threw `Legacy runtime state requires offline conversion: scripts/convert-request-state.mjs`.
- The startup guard found existing legacy run/delivery state without the new `requests/schema.json` marker and refused to start.
- With `Restart=always`, systemd retried every five seconds. There were 1,737 failed exits and 3,474 matching exception lines between `08:34:58` and `11:27:06 UTC`.

### Iteration 3 - verify recovery and non-regression

- An offline conversion/recovery staged a converted config and cut it over at `12:28:12 UTC`.
- Live state now contains `requests/schema.json` with `{"version":1}` and 3,758 request-state files.
- The recovery started RemoteLab, verified `/login` returned 200, performed a clean test restart, and recorded zero automatic restarts.
- The current service has stayed active since `12:28:24 UTC`, `NRestarts=0`, and no post-recovery schema/startup failure.
- The local build-info endpoint returns 200; the exact public URL returned 302 five consecutive times through both Cloudflare IPs.
- Public `/api/build-info` returns 200 with a response body identical to the local endpoint, and cloudflared has logged zero post-recovery origin-connection errors.

## Root Cause (Confirmed)

RemoteLab was restarted onto code that enforces request-state schema version 1 while its live configuration still contained legacy run/delivery state and no schema marker. The startup guard intentionally aborted, and `Restart=always` created a five-second crash loop. Cloudflared stayed healthy but had no process listening on `127.0.0.1:7690`, so it returned 502 for the reported URL.

## Evidence

- The exact URL appears in cloudflared logs with `connect: connection refused` to port 7690.
- At the same time, every RemoteLab startup exits at `lib/request-schema.mjs:19` with the explicit offline-conversion error.
- After converted state was cut over, RemoteLab bound port 7690 and the same public URL stopped returning 502.

## Reproduction case

Start this build with a config directory that has `chat-runs/run_*` or `chat-source-deliveries.json` but lacks `requests/schema.json`. `ensureRequestSchema()` throws before the chat server can remain available; requests through the still-running tunnel receive 502 because port 7690 refuses the origin connection.

## Proposed fix

Run the provided offline state converter against a stopped snapshot, validate the staged output, then atomically cut the converted config into place before starting the upgraded service. This recovery has already been completed on the host.

## Next step

No immediate repair remains. Add deployment gating so a schema-requiring build cannot restart against unconverted live state, and change the restart policy or start-limit behavior to avoid an unbounded crash loop on deterministic migration errors.
