# Eliminated hypotheses

## H-2: Cloudflare Tunnel connectivity failed while the local origin remained healthy

Eliminated. `cloudflared.service` stayed active continuously from 2026-08-12 and its request logs explicitly report `dial tcp 127.0.0.1:7690: connect: connection refused`. The edge-to-tunnel path handled the failing requests; the missing listener was behind the tunnel.

## H-3: The public hostname is routed to the wrong origin or tunnel ingress rule

Eliminated. The cloudflared log identifies ingress rule 0 as `http://localhost:7690`, and current public responses match the local service's 302 `/login` behavior. The tunnel target is the expected RemoteLab listener.

## H-4: The supplied session query triggers an application-specific 502

Eliminated in its deterministic form. The exact query generated connection-refused errors only while port 7690 was absent and now returns 302 consistently after the origin recovered. Five consecutive public requests and three local requests succeeded.
