# Active hypotheses

## H-3: Bot-2 receives messages but intentionally suppresses unmentioned messages in one group

- Test: correlate bot-2's `mention_only` chat rule with silent/unhandled events from that chat, and contrast mentioned/P2P messages that produce RemoteLab runs. Matching behavior confirms this for affected group messages.
- Result: confirmed. Bot-2 is configured with a `mention_only` override for chat `oc_73e827386be3d4ec48206b092f0dae05`. The focused routing test confirms unmentioned group messages are skipped before RemoteLab submission, while mentioned and P2P messages are admitted.
- Confidence: confirmed.

## H-6: The running RemoteLab server predates the committed HTTP source-delivery fix

- Test: compare the service start time with commit `27bcf1a4`, replay an existing request idempotently, and inspect archived request delivery plans. If the live process still drops `sourceDelivery` while the checked-in focused test passes, this is confirmed.
- Result: confirmed. `remotelab.service` started at `12:28:24 UTC`; commit `27bcf1a4` was created at `12:56:18 UTC`. The live idempotent replay matched the old fingerprint with `sourceDelivery` absent, while the current isolated HTTP test preserves it and creates the outbox entry.
- Confidence: confirmed.
