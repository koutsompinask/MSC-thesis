# Phone Remote Control

This deck includes a local WebSocket remote for phone control.

## Run It

Use two terminals from `presentation/`:

```bash
npm run remote
```

```bash
npm run dev:host
```

Open the presentation on the laptop:

```text
http://localhost:5173
```

Open the remote on the phone:

```text
http://<laptop-ip>:5173/remote
```

The phone and laptop must be on the same network. A phone hotspot is usually more reliable than university Wi-Fi because campus networks may block device-to-device traffic.

## Controls

- `Next` and `Previous` change slides.
- `First Slide` and `Last Slide` jump to boundaries.
- The pointer pad controls a red laser pointer overlay on the presentation.

## Troubleshooting

- If the remote says `disconnected`, confirm `npm run remote` is running.
- If the phone cannot open the URL, confirm Vite is running with `npm run dev:host`.
- If university Wi-Fi blocks access, connect the laptop to the phone hotspot and use the laptop IP shown by your OS.
- If the pointer works but slides do not change, refresh both laptop presentation and phone remote.
