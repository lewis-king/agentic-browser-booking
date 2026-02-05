import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { generateText, tool, stepCountIs } from 'ai';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import * as fs from 'node:fs';
import * as https from 'node:https';
import { z } from 'zod';
import 'dotenv/config';

const execAsync = promisify(exec);

// Resolve agent-browser path: prefer local node_modules, fall back to global
const AGENT_BROWSER = './node_modules/.bin/agent-browser';
const SCREENSHOT_PATH = '/tmp/booking-screenshot.png';

// 1. Initialize Kimi k2.5 via Moonshot's OpenAI-compatible API
const moonshot = createOpenAICompatible({
  name: 'moonshot',
  baseURL: 'https://api.moonshot.ai/v1',
  apiKey: process.env.MOONSHOT_API_KEY,
});

// Helper: call Kimi k2.5 vision directly (for screenshot analysis)
function analyzeScreenshot(base64Image: string, prompt: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      model: 'kimi-k2.5',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            {
              type: 'image_url',
              image_url: { url: `data:image/png;base64,${base64Image}` },
            },
          ],
        },
      ],
      max_tokens: 1024,
    });

    const req = https.request(
      {
        hostname: 'api.moonshot.ai',
        port: 443,
        path: '/v1/chat/completions',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${process.env.MOONSHOT_API_KEY}`,
          'Content-Length': Buffer.byteLength(data),
        },
      },
      (res) => {
        let body = '';
        res.on('data', (chunk) => (body += chunk));
        res.on('end', () => {
          try {
            const json = JSON.parse(body);
            const content =
              json.choices?.[0]?.message?.content ||
              json.choices?.[0]?.message?.reasoning_content ||
              'Could not analyze screenshot';
            resolve(content);
          } catch {
            reject(new Error(`Failed to parse vision response: ${body.slice(0, 200)}`));
          }
        });
      }
    );
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

// 2. Define the Browser Tool
const browserTool = tool({
  description:
    'Execute browser automation commands. Available commands include:\n' +
    '- open <url>, click <ref>, fill <ref> <text>, type <ref> <text>\n' +
    '- snapshot -i (get interactive elements with @refs)\n' +
    '- press <key>, select <ref> <value>, scroll up/down\n' +
    '- back, forward, reload, close, wait <ms>\n' +
    '- get text <ref>, get url, get title\n' +
    '- tab (list tabs), tab <n> (switch tab)\n' +
    '- frame <selector> (enter iframe), frame main (back to main)\n' +
    '- SEMANTIC FIND (use when @refs are not visible, e.g. in modals/overlays):\n' +
    '  find label <label> click — click element by its aria-label\n' +
    '  find label <label> fill <value> — fill input by its aria-label\n' +
    '  find text <text> click — click element containing text\n' +
    '  find role button click — click first button\n' +
    '  find placeholder <text> fill <value> — fill by placeholder\n' +
    '- wait --text <text> — wait for specific text to appear on page',
  inputSchema: z.object({
    command: z
      .string()
      .describe(
        'The agent-browser command to run, e.g. "open https://example.com", ' +
        '"snapshot -i", "click @e3", "fill @e5 John Smith", ' +
        '"find label Guests click", "find text Find table click", ' +
        '"find label Date click", "wait --text Pick a time"'
      ),
  }),
  execute: async ({ command }) => {
    const fullCommand = `${AGENT_BROWSER} --headed --json ${command}`;
    console.log(`\x1b[34m[Browser Action]:\x1b[0m ${command}`);

    try {
      const { stdout, stderr } = await execAsync(fullCommand, {
        timeout: 30_000,
      });
      const output = stdout.trim() || stderr.trim();
      console.log(
        `\x1b[90m[Browser Result]:\x1b[0m ${output.slice(0, 500)}${output.length > 500 ? '...' : ''}`
      );
      return output;
    } catch (error: any) {
      const msg =
        error.stdout?.toString().trim() ||
        error.stderr?.toString().trim() ||
        error.message;
      console.log(`\x1b[31m[Browser Error]:\x1b[0m ${msg}`);
      return `Error: ${msg}`;
    }
  },
});

// 3. Define the Screenshot Analysis Tool
const screenshotTool = tool({
  description:
    'Take a screenshot of the current browser page and analyze it with vision AI. ' +
    'Use this when "snapshot -i" returns no useful form elements — the page may have ' +
    'overlays, popups, iframes, or dynamic content that the text snapshot cannot see. ' +
    'Returns a text description of what is visible on screen, including any forms, buttons, or inputs.',
  inputSchema: z.object({
    question: z
      .string()
      .describe(
        'What you want to know about the screenshot, e.g. ' +
        '"What form fields and buttons are visible? Describe the booking widget." or ' +
        '"Is there a date picker or time selector visible?"'
      ),
  }),
  execute: async ({ question }) => {
    console.log(`\x1b[35m[Screenshot]:\x1b[0m Taking screenshot for analysis...`);

    try {
      // Take screenshot
      await execAsync(`${AGENT_BROWSER} --headed screenshot ${SCREENSHOT_PATH}`, {
        timeout: 10_000,
      });

      // Read as base64
      const imageBuffer = fs.readFileSync(SCREENSHOT_PATH);
      const base64 = imageBuffer.toString('base64');
      console.log(
        `\x1b[35m[Screenshot]:\x1b[0m Captured (${(imageBuffer.length / 1024).toFixed(0)} KB), sending to Kimi vision...`
      );

      // Analyze with Kimi vision
      const analysis = await analyzeScreenshot(
        base64,
        `You are looking at a browser screenshot of a restaurant booking page. ${question}\n\n` +
        'Be specific: describe any visible form fields, dropdowns, date pickers, time slots, ' +
        'buttons, overlays, modals, or popups. If you see a booking widget, describe its current state ' +
        'and what fields need to be filled in. Mention any text labels you can read.'
      );

      console.log(
        `\x1b[35m[Screenshot Analysis]:\x1b[0m ${analysis.slice(0, 500)}${analysis.length > 500 ? '...' : ''}`
      );
      return analysis;
    } catch (error: any) {
      console.log(`\x1b[31m[Screenshot Error]:\x1b[0m ${error.message}`);
      return `Error taking/analyzing screenshot: ${error.message}`;
    }
  },
});

// 4. Main Agent Function
async function startBooking(query: string) {
  console.log(`\x1b[32m[User]:\x1b[0m ${query}\n`);

  const { text, steps } = await generateText({
    model: moonshot.chatModel('kimi-k2.5'),
    system: `You are a specialized booking agent that automates restaurant and venue reservations using a browser.

Workflow:
1. Open the given URL.
2. Run "snapshot -i" to see all interactive elements.
3. If the page has a "Book a table", "Reserve online" or similar link/button, click it.
4. Wait for the booking form to load: "wait 3000", then "snapshot -i".
5. If "snapshot -i" still shows only navigation links and NO form fields (textbox, combobox, button for Date/Time/Guests), the booking widget is inside an IFRAME that the snapshot cannot see.
6. IFRAME WORKAROUND (critical — this is the most common issue):
   a. Run: eval "document.querySelector('iframe[src]')?.src" to get the iframe URL.
   b. If the URL contains "embed" remove that query parameter.
   c. Navigate directly to the iframe URL: open <url>
   d. Wait for it to load: "wait 3000", then "snapshot -i"
   e. You should now see form fields like textbox "Guests", button "Date", button "Time", button "Find table".
7. Fill in the form fields using @refs:
   - For the Guests dropdown: click the ref, then select the value.
   - For the Date picker: click the ref, navigate months if needed, click the target day number.
   - For the Time picker: click the ref, select from available slots.
   - Click "Find table" or equivalent submit button.
8. Fill in personal details (name, email, phone) on the next page when prompted.

Rules:
- After EVERY action, run "snapshot -i" to see what changed.
- Use @refs (e.g. @e1, @e2) to interact with elements.
- If "snapshot -i" shows no form fields after clicking a booking link, check for iframes IMMEDIATELY using eval. Do NOT scroll blindly.
- If clicking a link opens a new tab, run "tab" to list tabs and "tab 1" to switch.
- If a specific time slot is unavailable, STOP and report alternatives to the user.
- Never assume a booking succeeded — only confirm when you see "Success" or "Confirmed".
- If the site requires payment info, STOP and ask the user.
- If you encounter a CAPTCHA or login wall, STOP and inform the user.
- Use "screenshot_analyze" tool if you are confused about what is on screen.`,
    prompt: query,
    tools: { browser: browserTool, screenshot_analyze: screenshotTool },
    stopWhen: stepCountIs(50),
  });

  console.log(`\n\x1b[32m[Kimi]:\x1b[0m ${text}`);
  console.log(`\x1b[90m[Steps taken: ${steps.length}]\x1b[0m`);
}

// 5. Run
const query = process.argv[2] || 'Book me Goodman City at 7pm on 13th March';
startBooking(query).catch((err) => {
  console.error(`\x1b[31m[Fatal Error]:\x1b[0m`, err.message);
  process.exit(1);
});
