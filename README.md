# Agentic Browser Booking

An experiment combining [Vercel's agent-browser](https://github.com/vercel-labs/agent-browser) with [Moonshot's Kimi K2.5](https://platform.moonshot.ai/) to automate restaurant bookings via a real browser.

The idea: give an LLM a browser it can control, point it at a restaurant website, and watch it fill in the reservation form autonomously. Kimi K2.5's multimodal capabilities (particularly strong at vision) make it a good fit — when the text-based accessibility snapshot can't see a modal or overlay, the agent falls back to taking a screenshot and sending it to Kimi's vision API to figure out what's on screen.

## Demo

<video src="demos/demo.mp4" controls width="100%"></video>

## How it works

```
You: "Book Kronenhalle for 2 at 3:30pm on March 27th"
                    │
                    ▼
        ┌───────────────────────┐
        │   Kimi K2.5 (brain)   │  Decides what to do next
        │   via Vercel AI SDK   │  based on page state
        └───────┬───────────────┘
                │  tool calls
                ▼
        ┌───────────────────────┐
        │   agent-browser (hands)│  Executes browser commands:
        │   Playwright under    │  open, click, fill, snapshot
        │   the hood            │
        └───────┬───────────────┘
                │
                ▼
        ┌───────────────────────┐
        │   Headed Chromium     │  You watch it happen
        │   browser window      │  in real-time
        └───────────────────────┘
```

The agent loop works like this:

1. Kimi reads the page state via `snapshot -i` (compact accessibility tree with `@ref` handles)
2. Decides the next action (click a button, fill a field, navigate)
3. Calls the `browser` tool which shells out to `agent-browser` CLI
4. Reads the result, takes another snapshot, repeats

When `snapshot -i` can't see elements (common with iframe-based booking widgets), the agent:
- Uses `eval` to extract the iframe URL and navigates directly to it
- Falls back to `screenshot_analyze` which takes a screenshot and sends it to Kimi K2.5's vision API

## Why agent-browser over raw Playwright?

agent-browser is a CLI wrapper around Playwright designed for AI agents. The key difference is the **snapshot + @refs system**: `snapshot -i` returns a compact accessibility tree with ref handles that the LLM can reference directly. This is 82-93% smaller than raw DOM output, which matters when you're feeding it into an LLM context window.

```
# agent-browser
snapshot -i  →  button "Date" [ref=e3]
click @e3

# equivalent Playwright
await page.locator('button[aria-label="Date"]').click();
```

With raw Playwright you'd need to manage browser lifecycle, build your own accessibility tree parser, and design a JSON response format for tool results. agent-browser gives you all of that as a stateful CLI.

## Setup

```bash
# Clone and install
git clone <this-repo>
cd agentic-browser-booking
npm install

# Download Chromium for agent-browser
npx agent-browser install
# On Linux, if the browser fails to launch:
# npx agent-browser install --with-deps

# Add your Moonshot API key
# Get one from https://platform.moonshot.ai/
echo "MOONSHOT_API_KEY=your_key_here" > .env
```

## Usage

```bash
# Pass any booking request as a CLI argument
npx tsx index.ts "Book Kronenhalle Zurich for 2 at 3:30pm on March 27th"

npx tsx index.ts "Book Lux Zurich for 4 8pm on March 27th" --name "Jane Doe" --phone "+41 79 987 6543" --email "janedoe@gmail.com"

# Or use the npm script (uses a default query)
npm start
```

A headed Chromium window will open and you'll see the agent navigate the site in real-time. Terminal output shows each action and result:

```
[User]: Book Kronenhalle Zurich for 2 at 3:30pm on March 27th

[Browser Action]: open https://www.kronenhalle.com/en/restaurant/
[Browser Result]: {"success":true,"data":{"title":"Restaurant | Kronenhalle..."}}
[Browser Action]: snapshot -i
[Browser Action]: click @e15
[Browser Action]: eval "document.querySelector('iframe[src]')?.src"
[Browser Action]: open https://guest.foratable.com/restaurants/01993d8a...
[Browser Action]: snapshot -i
[Browser Result]: textbox "Guests", button "Date", button "Time", button "Find table"
[Browser Action]: click @e2
[Browser Action]: click @e13  (selects "2" guests)
[Browser Action]: click @e3   (opens date picker)
...

[Kimi]: The reservation details have been submitted. The system requires
SMS verification to complete the booking...
[Steps taken: 45]
```

## Architecture

**`index.ts`** — single file, ~230 lines:

| Section | What it does |
|---|---|
| `moonshot` provider (L17-21) | Connects to Moonshot API via `@ai-sdk/openai-compatible` — needed because `@ai-sdk/openai` defaults to OpenAI's Responses API which Moonshot doesn't support |
| `analyzeScreenshot()` (L24-76) | Direct HTTP call to Kimi K2.5 vision API with base64 image — used as a fallback when the text snapshot can't see modal/overlay content |
| `browserTool` (L79-128) | Shells out to `agent-browser` CLI. No AI here — just `exec("agent-browser --headed --json <command>")` |
| `screenshotTool` (L131-180) | Takes a screenshot via agent-browser, sends it to Kimi vision, returns text description |
| `startBooking()` (L183-226) | The main agent loop via Vercel AI SDK's `generateText` with tool calling and `stopWhen: stepCountIs(50)` |

## Lessons learned

**iframe booking widgets are the main challenge.** Most restaurant sites (Kronenhalle, Baur's, etc.) embed third-party booking systems (Foratable, Aleno, OpenTable) in iframes. `snapshot -i` is completely blind to iframe content, and agent-browser's `frame` command doesn't reliably switch context. The workaround: use `eval` to grab the iframe `src` URL and navigate directly to it.

**`@ai-sdk/openai` vs `@ai-sdk/openai-compatible`.** The standard OpenAI provider in AI SDK v6 defaults to the Responses API (`/v1/responses`). Moonshot only supports `/v1/chat/completions`, so you need `@ai-sdk/openai-compatible` with `createOpenAICompatible()` and `.chatModel('kimi-k2.5')`.

**AI SDK v6 API changes.** `tool()` uses `inputSchema` (not `parameters`), and multi-step is `stopWhen: stepCountIs(N)` (not `maxSteps`).

**Kimi K2.5 supports vision via the standard chat completions endpoint.** Send images as base64 data URLs in the `image_url` content type — same format as OpenAI's vision API. Works well for describing booking form layouts.

## Limitations

- SMS/email verification steps can't be completed by the agent
- CAPTCHAs will block the agent
- Payment forms are intentionally skipped (the agent stops and asks the user)
- Some booking widgets may use shadow DOM or other rendering that the snapshot can't see
- The agent occasionally picks the wrong time slot if the ref numbering is ambiguous

## Stack

- [Vercel AI SDK](https://sdk.vercel.ai/) v6 — `generateText` with tool calling loop
- [agent-browser](https://github.com/vercel-labs/agent-browser) v0.9.1 — AI-first browser automation CLI (Playwright under the hood)
- [Kimi K2.5](https://platform.moonshot.ai/) — Moonshot's multimodal LLM (1T MoE, 32B active params)
- TypeScript, run with [tsx](https://github.com/privatenumber/tsx)
