# ChatGPT Live über bestehenden ChatGPT-Web-Login: Machbarkeitsprüfung

**Stand:** 25.10.2026. **Status:** Read-only-Codeanalyse und PC-Control-Verbindung bestätigt. **Kein Live-/Browser-Audiotest durchgeführt.**

## Ziel und Nichtziele
- **Ziel:** sichtbaren ChatGPT-Live-Sprachmodus im bereits angemeldeten ChatGPT-Webbrowser für Dodaios prüfen, mit den im Chat verfügbaren Stimmen.
- **Nicht:** Codex-OAuth, `openai_subscription_live.py`, OpenAI-API, kostenpflichtige Realtime-API, Web-Access-Token-Export oder undokumentierte WebRTC-Endpunkte.
- **Keine Änderungen** am produktiven Jarvis-Audio-/Brain-Routing; keine zusätzlichen Browserprofile oder Programme ohne Freigabe.

## Verifizierter Ist-Stand

### Lokaler PC und Branch
- PC Control über `worktree-ai/pc-control` Issue #63: `pc_file_list` auf `D:\Projects\Dodaios` erfolgreich, also PC-Worker erreichbar.
- `D:\Projects\Dodaios\.git\HEAD` = `refs/heads/pr61`.
- `.git/refs/heads/pr61` = `c1ab9d5ee2102ed9aebea1507a633c461d890219`. Entspricht GitHub-PR #61 `fix/worktree-add-timeout-chatgpt-20261025` (Stand 25.10.2026).
- Drei lokale Quelltexte wurden über PC Control gelesen und gegen den PR-61-GitHub-Branch mit normalisierten Zeilenenden verglichen: **alle identisch**.
- Keine aktiven Browserprozesse, Login-Status, Mikrofonberechtigungen, lokalen uncommitteten Änderungen oder Web-Voice-Sitzungen geprüft. PC Control bietet dafür in der aktuellen Allowlist keine UI-Steuerung und kein freies Terminal.

### Relevanter Dodaios-Code
| Datei | Befund |
| --- | --- |
| `jarvis/plugins/brain/chatgpt_web.py` | Text/SSE-Bridge, `supports_tools=False`, keine Web-Voice-Audioverbindung |
| `jarvis/plugins/brain/chatgpt_web_bridge.py` | Verwaltet `chatgpt-web-provider` und eigenes persistentes Browserprofil; Start bei Bedarf |
| `jarvis/browser_voice/route.py` | Eigenes `/ws/audio`-Audio-/Realtime-Routing; keine ChatGPT-Web-Voice-Kopplung |
| `jarvis/browser_voice/session.py` | Klassische PCM → STT → Brain → TTS-Kette |
| `jarvis/plugins/realtime/openai_subscription_live.py` | Separater **Codex**-Abonnementpfad, ausdrücklich nicht verwenden |
| `jarvis/ui/web/voice_call_routes.py` | Start/Hangup des **Dodaios**-Sprachsystems, nicht des ChatGPT-Web-Sprachbuttons |
| `jarvis/ui/web/browser_profile_routes.py` | Separates Dodaios-Browserprofil-/Chrome-Connector-System, nicht automatisch dasselbe Profil wie die ChatGPT-Web-Bridge |

### Upstream und Referenzen
1. [guberm/chatgpt-web-provider](https://github.com/guberm/chatgpt-web-provider): Browser-Backend ruft `playwright.chromium.launch_persistent_context(profile_dir)` auf. Es öffnet ein **eigenes** persistentes Chromium-Profil, keine vorhandene CDP-Verbindung. Es verarbeitet Textantworten, **keinen** Sprachmodus. **Das Profil nicht parallel in einem zweiten Browser öffnen.**
2. [jwoglom/chatgpt-voice-bridge](https://github.com/jwoglom/chatgpt-voice-bridge): startet/beendet ChatGPT Voice in bestehendem Chrome per Playwright/CDP; Linux-Audiobrücke mit PulseAudio/SIP. **Nicht** direkt Windows-kompatibel und nicht nachweislich mit GPT-Live-1.1 getestet.
3. [dyhhhhhh/chatgpt-web-voice](https://github.com/dyhhhhhh/chatgpt-web-voice) und [Space3044/chatgpt-web-voice](https://github.com/Space3044/chatgpt-web-voice): inoffizielle WebRTC-/Token-Gateways. Keine bestätigte aktuelle Kompatibilität und kein geeigneter Weg für „bestehendes Browserprofil ohne Tokenexport“.
4. [Offizielle ChatGPT-Voice-Hilfe](https://help.openai.com/en/articles/20001274-chatgpt-voice): Voice im Web ist ein ChatGPT-Endnutzerfeature. Es ist **keine** dokumentierte Drittanbieter-Voice-API. Automatisierung/Extraktion muss auf Zulässigkeit geprüft werden.

## Technische Schlussfolgerung
**Text-Bridge ≠ ChatGPT-Live-Voice.** Das vorhandene Browserprofil ist zwar für ChatGPT-Webtext vorhanden, aber `chatgpt-web-provider` besitzt den Playwright-Browserkontext exklusiv. Das Linux-Referenzprojekt setzt einen CDP-verwalteten Chrome-Browser voraus. Einfach den Linux-Bot auf Windows zu starten oder ein zweites Playwright-Fenster auf das gleiche Profil zu setzen, wäre falsch.

Für einen minimalen **manuellen** Machbarkeitstest genügt zunächst der sichtbare ChatGPT-Webbrowser mit echtem Mikrofon und Lautsprecher. Eine vollständige Jarvis-Integration (Wake-Word, Tool-Ausführung, TTS/Audioübergabe) ist damit noch **nicht** nachgewiesen.

## Kleine, voneinander getrennte Folgeaufgaben

**A. Sichere Laufzeitinspektion (nur lesen)**
- Den konfigurierten Profilpfad, Browserprozess und Login-Status **nur über redigierte Statusfelder** prüfen. Niemals `jarvis.toml`, Browser-Cookies, Session-Storage oder Tokens in GitHub-Kommentare/Logs kopieren.
- Prüfen, ob der Text-Bridge-Prozess läuft und wie er den Browserkontext besitzt. Kein Browser-Neustart.

**B. Sichtbarer UI-Smoke-Test (separate Freigabe, kein automatisches Auslesen)**
- Den bereits angemeldeten Browser mit bestehendem Profil verwenden. Profil-Exklusivität beachten, keine zweite Browserinstanz auf demselben Profil.
- Manuell ChatGPT-Voice-Button und auswählbare Stimmen prüfen; Mikrofonfreigabe nur durch den Nutzer.
- Kurze deutsche und arabische Unterhaltung, Unterbrechung, Stoppen, erneuter Start. Audioausgabe hörbar? Profil danach unverändert? Keine Kosten/API-Aufrufe?
- Exakte Fehler und Zeitstempel dokumentieren. Bei Login-/Cloudflare-Problemen stoppen, nichts umgehen.

**C. Nur Architektur-Skizze**
- Besitzübergang des Mikrofons: Jarvis-Wake-Word pausieren, solange sichtbarer Browser-Voice-Dialog läuft; nach Hangup wieder aufnehmen.
- Kein doppelter STT/TTS- oder Realtime-Fallback, kein ungewollter Codex-/API-Aufruf.
- Fehlerszenarien: Nutzerabbruch, Browserfenster weg, Mic verweigert, Stimme fehlt, Quota, Timeout, Session abgelaufen.
- Lokale Jarvis-Werkzeuge sind **nicht automatisch** durch ChatGPT Voice verfügbar. Dafür gesonderten sicheren Mechanismus definieren.

**D. Erst nach Freigabe und positivem Test**
- Eine kleinste Implementierung mit Tests auf separatem Branch, nicht in `main` und nicht in `pr61` schreiben.
- Testen: Start/Stop/Abbruch, Mikrofon-Exklusivität, kein Tokenexport, kein zweiter Browser, keine kostenpflichtigen Fallbacks, unveränderte vorhandene Voice-Funktion.

## Stop-Kriterien
Keine parallele Öffnung desselben Browserprofils; keine Web-Token-Extraktion; keine inoffiziellen WebRTC-Aufrufe; keine automatisierte ChatGPT-Ausgabeextraktion; keine Installation von PulseAudio/Docker/virtuellen Audiogeräten; kein produktiver Commit oder Merge ohne Freigabe.

## Ergebnis dieses Audits
**Bestätigt:** PC erreichbar, Code auf `pr61` identisch zu GitHub, Web-Textbridge vorhanden, Browser-Voice noch nicht integriert. **Nicht bestätigt:** ChatGPT-Live-Button und ausgewählte Stimmen im vorhandenen lokalen Browserprofil, aktive Sitzung, Audio-/Unterbrechungsfunktion, Zulässigkeit einer Drittanbieter-Steuerung.

*Dieses Dokument wurde ausschließlich auf einem separaten Dokumentationsbranch erstellt.*
