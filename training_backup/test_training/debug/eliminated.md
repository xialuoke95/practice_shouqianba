# Eliminated hypotheses

## H-2: Connector-to-RemoteLab API/auth transport fails

Eliminated. The connector successfully created/submitted the RemoteLab turn, and every publication poll returned HTTP 200. Session `48c730cda5a3cfe4bd8e8828db87bdcd` and run `run_8972f45109204d859b2f2b20` were persisted. The failure happened inside the Codex provider run, after local API transport succeeded.

## H-3: Feishu outbound publication is generally broken

Eliminated. For the same inbound message, the connector successfully delivered the Chinese fallback notice and persisted its outbound Feishu message ID. The generated-reply path failed before it reached normal Feishu delivery.
