# Tradingbot (låtsaspengar)

Webbappen ligger i `docs/` och uppdateras automatiskt varje vardag 22:40 (svensk vintertid) av GitHub Actions.

## Kom igång (en gång)
1. Skapa ett nytt repo på github.com, t.ex. `tradingbot` (Public, så GitHub Pages är gratis).
2. Ladda upp alla filer från zip-filen (dra in hela innehållet på "uploading an existing file").
   Viktigt: mappen `.github/workflows` måste följa med.
3. Settings → Pages → Source: "Deploy from a branch", branch `main`, mapp `/docs` → Save.
4. Settings → Actions → General → Workflow permissions: "Read and write permissions" → Save.
5. (Valfritt) Settings → Secrets and variables → Actions → New secret
   `ANTHROPIC_API_KEY` = din nyckel, så läser Claude nyheterna.
6. Actions → "Daglig papperstrading" → "Run workflow" för att köra direkt.

Appen finns sedan på https://DITTNAMN.github.io/tradingbot/
Öppna den i mobilen → Dela → "Lägg till på hemskärmen".
