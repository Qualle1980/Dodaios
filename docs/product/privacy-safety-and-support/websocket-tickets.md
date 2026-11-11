---
title: "WebSocket tickets"
slug: websocket-tickets
summary: Why Dodaios mints short-lived WebSocket tickets, when HTTPS is required, and how that relates to the Control Key.
section: "Privacy, safety, and support"
section_order: 6
order: 7
diataxis: explanation
status: active
owner: maintainers
last_reviewed: 2026-11-11
phase: "-"
audience: end-user
tags: [websocket, tickets, control-key, https, browser, authentication]
related: [control-key, control-api-reference, troubleshooting, architecture]
---

Some Dodaios browser connections use a **WebSocket** for live updates (chat,
voice, progress, terminals). The browser does not always send the normal
HttpOnly session cookie on that handshake, so Dodaios mints a short-lived
**WebSocket ticket** over an authenticated HTTP request first. The socket then
presents that ticket once; Dodaios consumes it and opens the live channel.

## Why tickets exist

Desktop WebKit (and similar browsers) can drop the session cookie from a
WebSocket upgrade even when ordinary HTTP requests carry it. Without a separate
ticket, the live channel would stay offline after a successful unlock. The
ticket is the deliberate workaround: mint over HTTP that already proved who you
are, then spend the ticket on the first WebSocket connect.

A ticket lasts about one minute and is single-use. It is not a long-lived API
key and it is not the Control Key.

## When HTTPS matters

On the same computer, a direct loopback connection (for example
`http://127.0.0.1`) can mint and use tickets without TLS.

From another device, over the LAN, or through a reverse proxy, the browser
surface must use **HTTPS**. Plain HTTP from outside loopback is refused for the
ticket path so a ticket cannot be stolen in transit. That is why a remote
server that only serves HTTP appears to "issue no WebSocket tickets": the
authenticated mint step never succeeds until TLS is in place.

Remote and non-loopback callers must also present the **Control Key** (or an
already unlocked browser session) as described in the Control Key guide. The
ticket does not replace the Control Key; it only bridges the cookie gap for the
socket handshake.

## What to check when live updates stay offline

1. Open Dodaios in the browser and confirm the main UI unlocks.
2. Prefer HTTPS for any address that is not loopback.
3. Confirm the Control Key (or an existing session) is accepted before expecting
   live channels.
4. Retry once after unlock; a ticket that already expired or was spent will not
   open a second socket.

If the HTTP API answers but every live view stays disconnected, the usual cause
is missing HTTPS on a non-loopback host, not a missing WebSocket feature.
