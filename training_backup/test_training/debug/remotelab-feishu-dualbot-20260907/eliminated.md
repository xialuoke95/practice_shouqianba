# Eliminated hypotheses

## H-1: One connector process is down or crash-looping

- Both connector units are `active/running`, have stable PIDs, and have `NRestarts=0` since `12:29:28 UTC`.

## H-2: One connector cannot authenticate or establish its Feishu persistent connection

- Both connectors logged `persistent connection ready (feishu-cn)` and `ws client ready`, retain established network connections, and both accepted new Feishu events.

## H-4: The default bot's Feishu event delivery stopped after the recovery restart

- A new default-bot group event arrived at `13:06:39 UTC`, was admitted as request `feishu:om_x100b66df2321f8a0dd8545277161dd2`, and completed as run `run_ae4365933950fc0f72890014` at `13:06:47 UTC`.

## H-5a: Recent admitted messages fail during RemoteLab model processing

- The default-bot run `run_ae4365933950fc0f72890014` and bot-2 run `run_6e00229ae54f6f82d0e303ed` both completed with exit code 0 and persisted assistant results.
