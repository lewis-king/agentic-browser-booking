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

// Restaurant URL mapping
const RESTAURANT_MAP: Record<string, string> = {
  'kronenhalle': 'https://www.kronenhalle.com/en/restaurant/',
  'tschuggen grand hotel': 'https://tschuggencollection.ch/en/hotel/tschuggen-grand-hotel/restaurants',
  'tschuggen': 'https://tschuggencollection.ch/en/hotel/tschuggen-grand-hotel/restaurants',
  'baur\'s': 'https://www.baurs-zurich.ch/en/reservation.html',
  'baurs': 'https://www.baurs-zurich.ch/en/reservation.html',
  'spices kitchen': 'https://burgenstockresort.com/en/dining/spices-kitchen-and-terrace#popup',
  'spices kitchen & terrace': 'https://burgenstockresort.com/en/dining/spices-kitchen-and-terrace#popup',
  'marguita': 'https://www.marguita.ch/#bookatable',
  'la rotisserie': 'https://mytools.aleno.me/reservations/v2.0/reservations.html?k=eyJrIjoid2l2dTVrM2lsNm15cnBiOWlwdzZ4bmViajhycnVkaWRpZ280bGZwODBsbzlhNGlweTEiLCJyIjoiUlp6WXlvRVBRTXlMTUQ0YXMiLCJzIjoiaHR0cHM6Ly9teXRvb2xzLmFsZW5vLm1lLyJ9',
  'lux zurich': 'https://www.lux-zurich.ch/de/',
  'the restaurant at the dolder grand': 'https://www.exploretock.com/saltzatthedoldergrandhotel/?_gl=1*13zno47*_gcl_aw*R0NMLjE3Njk3OTI5NzIuQ2owS0NRaUF5dkhMQmhEbEFSSXNBSHhsNnhxRC14al9sZ3Q5TlFNbXpHQ2lLeXJ1TUx4OTRsaEFfak1ORHFKNEpiQmZ2aTByOWlTOHNCb2FBalFrRUFMd193Y0I.*_gcl_au*MjQxNDYzMjEuMTc2OTc5Mjk2NA..*_ga*MTIxMTExOTMwOS4xNzY5NzkyOTY0*_ga_17SXM4315G*czE3Njk3OTI5NjMkbzEkZzEkdDE3Njk3OTI5NzIkajU4JGwwJGgxNjA5ODcwNTc4',
  'dolder grand': 'https://www.exploretock.com/saltzatthedoldergrandhotel/?_gl=1*13zno47*_gcl_aw*R0NMLjE3Njk3OTI5NzIuQ2owS0NRaUF5dkhMQmhEbEFSSXNBSHhsNnhxRC14al9sZ3Q5TlFNbXpHQ2lLeXJ1TUx4OTRsaEFfak1ORHFKNEpiQmZ2aTByOWlTOHNCb2FBalFrRUFMd193Y0I.*_gcl_au*MjQxNDYzMjEuMTc2OTc5Mjk2NA..*_ga*MTIxMTExOTMwOS4xNzY5NzkyOTY0*_ga_17SXM4315G*czE3Njk3OTI5NjMkbzEkZzEkdDE3Njk3OTI5NzIkajU4JGwwJGgxNjA5ODcwNTc4',
  'alex restaurant': 'https://alexlakezurich.com/en/restaurants/alex-restaurant/',
  'alex': 'https://alexlakezurich.com/en/restaurants/alex-restaurant/',
  'bindella zurich': 'https://www.bindella.ch/restaurants/bindella-zuerich',
  'bindella': 'https://www.bindella.ch/restaurants/bindella-zuerich',
};

// Helper: resolve restaurant name from query to URL
function resolveRestaurantUrl(query: string): { url: string | null; restaurantName: string | null } {
  const lowerQuery = query.toLowerCase();
  
  // Try exact matches first, then partial matches
  for (const [name, url] of Object.entries(RESTAURANT_MAP)) {
    if (lowerQuery.includes(name.toLowerCase())) {
      return { url, restaurantName: name };
    }
  }
  
  return { url: null, restaurantName: null };
}

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
    '- wait --text <text> — wait for specific text to appear on page\n' +
    '- EVAL COMMAND (IMPORTANT: wrap JavaScript in single quotes):\n' +
    "  eval 'document.querySelector(\"iframe\").src' — get iframe URL (use single quotes)",
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
      
      // If browser not launched, try to close stale instance and retry
      if (msg.includes('Browser not launched') && command.startsWith('open')) {
        console.log(`\x1b[33m[Browser]:\x1b[0m Cleaning up stale browser instance...`);
        try {
          await execAsync(`${AGENT_BROWSER} --json close`, { timeout: 5_000 });
          console.log(`\x1b[33m[Browser]:\x1b[0m Retrying command...`);
          const { stdout, stderr } = await execAsync(fullCommand, { timeout: 30_000 });
          const output = stdout.trim() || stderr.trim();
          console.log(
            `\x1b[90m[Browser Result]:\x1b[0m ${output.slice(0, 500)}${output.length > 500 ? '...' : ''}`
          );
          return output;
        } catch (retryError: any) {
          const retryMsg = retryError.stdout?.toString().trim() || retryError.stderr?.toString().trim() || retryError.message;
          console.log(`\x1b[31m[Browser Error]:\x1b[0m ${retryMsg}`);
          return `Error: ${retryMsg}`;
        }
      }
      
      console.log(`\x1b[31m[Browser Error]:\x1b[0m ${msg}`);
      return `Error: ${msg}`;
    }
  },
});

// 3. Define the Modal Interaction Tool
const modalInteractionTool = tool({
  description:
    'INTERACT WITH MODALS AND POPUPS - Use this when a booking form appears in a modal overlay that is not showing up in the regular snapshot. When you see a modal in screenshots but cannot interact with it via @refs, use this tool to click elements in the modal. Common selectors: input fields with placeholder, dropdown buttons, or submit buttons.',
  inputSchema: z.object({
    action: z.enum(['click', 'fill']).describe('The action to perform: click or fill'),
    selector: z.string().describe('CSS selector to find the element, e.g., "button:contains(\'Buchen\')", "input[placeholder*=\"Datum\"]", ".modal button"'),
    value: z.string().optional().describe('Value to fill (only for fill action)'),
  }),
  execute: async ({ action, selector, value }) => {
    console.log(`\x1b[33m[Modal]:\x1b[0m ${action} on "${selector}"`);
    
    try {
      // Try various methods to interact with the modal
      let cmd = '';
      
      if (action === 'click') {
        // Try multiple strategies for clicking
        const strategies = [
          `eval 'document.querySelector("${selector}")?.click()'`,
          `eval 'document.querySelectorAll("${selector}")[0]?.click()'`,
          `eval 'Array.from(document.querySelectorAll("button")).find(b => b.textContent.includes("${selector.replace(/"/g, '\\"')}"))?.click()'`,
        ];
        
        for (const strategy of strategies) {
          try {
            const { stdout } = await execAsync(`${AGENT_BROWSER} --headed --json '${strategy}'`, { timeout: 10_000 });
            const result = stdout.trim();
            if (result.includes('"data":{}') || result.includes('success')) {
              return `Successfully clicked using: ${strategy}`;
            }
          } catch {
            // Try next strategy
          }
        }
        
        // If all strategies fail, try clicking at modal center
        cmd = `eval 'const modal = document.querySelector("[role=dialog], .modal, [class*=modal], [class*=popup]"); if(modal) modal.click(); else document.elementFromPoint(window.innerWidth/2, window.innerHeight/2).click()'`;
      } else if (action === 'fill') {
        cmd = `eval 'const el = document.querySelector("${selector}"); if(el) { el.value = "${value || ''}"; el.dispatchEvent(new Event("input", {bubbles: true})); el.dispatchEvent(new Event("change", {bubbles: true})); }'`;
      }
      
      const { stdout } = await execAsync(`${AGENT_BROWSER} --headed --json '${cmd}'`, { timeout: 10_000 });
      return stdout.trim();
    } catch (error: any) {
      const msg = error.stdout?.toString().trim() || error.stderr?.toString().trim() || error.message;
      return `Modal interaction error: ${msg}`;
    }
  },
});

// 4. Define the Iframe Detection Tool
const iframeDetectionTool = tool({
  description:
    'Detect iframes on the current page and extract their URLs. Use this when you suspect a booking widget is inside an iframe but the eval command is failing or returning null. This tool checks for iframes with various selectors and returns the most likely booking iframe URL.',
  inputSchema: z.object({
    dummy: z.string().optional().describe('Optional parameter, not used'),
  }),
  execute: async () => {
    console.log(`\x1b[36m[Iframe Detection]:\x1b[0m Searching for booking iframes...`);
    
    const selectors = [
      'iframe[src*="booking"]',
      'iframe[src*="reservation"]',
      'iframe[src*="reservierung"]',
      'iframe[src*="aleno"]',
      'iframe[src*="foratable"]',
      'iframe[src*="opentable"]',
      'iframe[src*="tock"]',
      'iframe[src]'
    ];
    
    for (const selector of selectors) {
      try {
        const cmd = `eval 'Array.from(document.querySelectorAll("${selector}")).map(f => f.src).filter(s => s && s.length > 0).join("|")'`;
        const { stdout } = await execAsync(`${AGENT_BROWSER} --headed --json '${cmd}'`, { timeout: 10_000 });
        const result = stdout.trim();
        
        // Parse the JSON response
        const match = result.match(/\{[^}]+\}/);
        if (match) {
          const json = JSON.parse(match[0]);
          if (json.data && json.data.result) {
            const urls = json.data.result.split('|').filter((u: string) => u.length > 0);
            if (urls.length > 0) {
              console.log(`\x1b[36m[Iframe Detection]:\x1b[0m Found ${urls.length} iframe(s)`);
              return urls.join('\n');
            }
          }
        }
      } catch {
        // Try next selector
      }
    }
    
    return 'No iframes found with common booking selectors. Try using screenshot_analyze to visually inspect the page.';
  },
});

// 5. Define the Screenshot Analysis Tool
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

// 6. Main Agent Function
async function startBooking(query: string, details: BookingDetails) {
  console.log(`\x1b[32m[User]:\x1b[0m ${query}\n`);

  // Resolve restaurant from query
  const { url, restaurantName } = resolveRestaurantUrl(query);
  
  if (!url) {
    console.log(`\x1b[31m[Error]:\x1b[0m Could not find restaurant in query. Available restaurants:`);
    Object.keys(RESTAURANT_MAP).forEach(name => console.log(`  - ${name}`));
    process.exit(1);
  }

  console.log(`\x1b[36m[Resolved]:\x1b[0m ${restaurantName} → ${url}\n`);
  
  // Extract booking parameters
  const { guests, time, date, normalizedQuery } = extractBookingParams(query);
  
  console.log(`\x1b[36m[Booking Details]:\x1b[0m`);
  if (guests) console.log(`  Guests: ${guests}`);
  if (time) console.log(`  Time: ${time}`);
  if (date) console.log(`  Date: ${date}`);
  if (details.name) console.log(`  Name: ${details.name}`);
  if (details.email) console.log(`  Email: ${details.email}`);
  if (details.phone) console.log(`  Phone: ${details.phone}`);
  console.log();
  
  // Warn if email is missing
  if (!details.email) {
    console.log(`\x1b[33m[Warning]:\x1b[0m No email provided. Most booking systems require email.`);
    console.log(`\x1b[33m[Tip]:\x1b[0m Add --email "your@email.com" to complete the booking automatically.\n`);
  }
  
  // Build personal details section for system prompt
  const personalDetailsSection = [];
  if (details.name) personalDetailsSection.push(`Name: ${details.name}`);
  if (details.email) personalDetailsSection.push(`Email: ${details.email}`);
  if (details.phone) personalDetailsSection.push(`Phone: ${details.phone}`);
  if (details.notes) personalDetailsSection.push(`Special requests: ${details.notes}`);
  
  const personalDetailsPrompt = personalDetailsSection.length > 0
    ? `\n\nPERSONAL DETAILS TO USE (fill these in when you see the corresponding form fields):\n${personalDetailsSection.join('\n')}\n\nIf the form asks for Salutation/Title (Herr/Frau/Mr/Ms), select the appropriate option based on the name provided.`
    : '';

  const bookingParamsPrompt = [];
  if (guests) bookingParamsPrompt.push(`Guests: ${guests}`);
  if (time) bookingParamsPrompt.push(`Time: ${time}`);
  if (date) bookingParamsPrompt.push(`Date: ${date}`);

  const { text, steps } = await generateText({
    model: moonshot.chatModel('kimi-k2.5'),
    system: `You are a specialized booking agent that automates restaurant and venue reservations using a browser.

BOOKING PARAMETERS FROM USER QUERY:
${bookingParamsPrompt.length > 0 ? bookingParamsPrompt.join('\n') : 'Extract date, time, and guests from the query above.'}${personalDetailsPrompt}

Workflow:
1. Open the given URL: ${url}
2. Run "snapshot -i" to see all interactive elements.
3. Look for booking-related elements: "Book a table", "Reserve online", "Reservation", "Book now", "Tisch reservieren", "Reservierung" buttons/links. Click the most appropriate one.
4. After clicking, run "get url" to check if the page navigated. Then wait 3000ms, then "snapshot -i" to see the new page state.
5. If "snapshot -i" still shows only navigation links and NO form fields (textbox, combobox, button for Date/Time/Guests):
   - Take a screenshot to check for a MODAL/POPUP overlay (common with modern booking widgets)
   - If you see a modal with booking form fields in the screenshot but not in snapshot: Use modal_interact tool to click buttons in the modal
   - Otherwise, the booking widget is likely inside an IFRAME
6. IFRAME WORKAROUND (critical — this is the most common issue):
   a. First, use the "detect_iframes" tool to automatically find iframe URLs - this is more reliable than manual eval commands.
   b. If detect_iframes doesn't find anything or eval is needed: Run: eval 'document.querySelector("iframe").src' to get the iframe URL. (Use single quotes around the eval command)
   c. If eval fails with a shell error, try: eval 'document.querySelectorAll("iframe")[0].src' or try eval 'Array.from(document.querySelectorAll("iframe")).map(f => f.src).filter(s => s && s.length > 0)[0]'
   d. If the URL contains "embed" remove that query parameter.
   e. Navigate directly to the iframe URL: open <url>
   f. Wait for it to load: "wait 3000", then "snapshot -i"
   g. You should now see form fields like textbox "Guests", button "Date", button "Time", button "Find table".
7. Fill in the booking form step by step:
   - For the Guests dropdown/combobox: click the ref, wait for options, select the matching number (${guests || 'as specified'}).
   - For the Date picker: 
     * Click the date field/ref to open the calendar
     * Look at the calendar header to see current month/year (e.g., "März 2026" means March 2026)
     * Navigate using arrows (< >) to reach the target month (${date ? date.split('-')[1] : 'as specified'})
     * In the calendar grid, look for buttons showing day numbers (1-31). The calendar may also show full dates like "Freitag, 27. März 2026" or just the number "27"
     * Find and click the button with the number ${date ? date.split('-')[2] : 'as specified'}
     * CRITICAL: After selecting, immediately verify the displayed date shows "${date ? date.split('-')[2] : 'the correct day'}" and not a different day
     * If the date is WRONG (e.g., shows 26th instead of 27th), click the date field again and re-select the correct day - fix it immediately before proceeding
   - For the Time picker:
     * Only proceed after confirming the DATE is correct
     * Click the time field and select ${time || 'the requested time'} or closest available slot
     * Verify the displayed time matches before moving on
   - For the Guests dropdown:
     * Only proceed after confirming DATE and TIME are correct
     * Select ${guests || 'the requested number'} guests
     * Verify the displayed guest count matches
   - For the Time picker: click the time field, select ${time || 'the requested time'} or the closest available slot.
   - Look for and click "Find table", "Check availability", or "Next" button.
   - After filling all fields, ALWAYS verify the summary shows: Date=${date || 'correct date'}, Time=${time || 'correct time'}, Guests=${guests || 'correct number'}
8. On the personal details page:
   - Fill in ALL required fields with the personal details provided above.
   - Look for fields like: First name, Last name, Email, Phone, Comments/Notes.
   - If a field is required but no value was provided above, STOP and report to the user.
9. CRITICAL - VERIFY AT EVERY STEP:
   - After setting date, time, and guests, ALWAYS check the displayed summary before proceeding
   - Date must show ${date ? date.split('-')[2] : 'correct day'} (e.g., "27.03.2026" for March 27th)
   - Time must show ${time || 'correct time'}
   - Guests must show ${guests || 'correct number'}
   - If ANY detail is wrong, fix it immediately - don't proceed with incorrect information
10. Click the final "Book", "Reserve", or "Confirm" button to complete the booking.

CRITICAL RULES:
- After EVERY action, run "snapshot -i" to see what changed.
- Use @refs (e.g. @e1, @e2) to interact with elements.
- BE EFFICIENT: Don't repeat actions. If you already clicked a date/time/guests field and it's correct, don't click it again.
- If you click a button and nothing seems to change (no form appears), try clicking different buttons with booking-related text.
- If "snapshot -i" shows no form fields after clicking a booking link, check for iframes IMMEDIATELY using eval. Do NOT scroll blindly.
- If the site is in German (e.g., Bindella), look for "Tisch reservieren", "Reservierung", "Jetzt reservieren", "Tisch buchen", or "Online reservieren" buttons.
- MODAL HANDLING: When you see a booking modal in screenshots but cannot find form elements in snapshot, use modal_interact tool to click elements (e.g., modal_interact with action="click" and selector="Buchen" or "button:contains('Buchen')")
- If clicking a link opens a new tab, run "tab" to list tabs and "tab 1" to switch.
- If the requested time slot is unavailable, look for alternative times close to the requested time and report them to the user. Do NOT book a different time without confirmation.
- Never assume a booking succeeded — only confirm when you see explicit success text like "Reservation confirmed", "Booking complete", "Success", or a confirmation number.
- If the site requires payment info, STOP and ask the user before proceeding.
- If you encounter a CAPTCHA, login wall, or SMS verification, STOP and inform the user.
- If you see any error messages ("Time not available", "Fully booked", etc.), report them clearly to the user.
- Use "screenshot_analyze" tool if you are confused about what is on screen or if form fields aren't visible in the snapshot.
- Be patient: wait for pages to load fully before proceeding (use "wait 2000" or "wait 3000").`,
    prompt: normalizedQuery,
    tools: { browser: browserTool, screenshot_analyze: screenshotTool, detect_iframes: iframeDetectionTool, modal_interact: modalInteractionTool },
    stopWhen: stepCountIs(100),
  });

  console.log(`\n\x1b[32m[Kimi]:\x1b[0m ${text}`);
  console.log(`\x1b[90m[Steps taken: ${steps.length}]\x1b[0m`);
}

// 7. Parse CLI arguments
interface BookingDetails {
  name: string | null;
  email: string | null;
  phone: string | null;
  notes: string | null;
}

function parseCliArgs(): { query: string; details: BookingDetails } {
  const args = process.argv.slice(2);
  
  const details: BookingDetails = {
    name: null,
    email: null,
    phone: null,
    notes: null,
  };
  
  // Extract flags
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--name' && i + 1 < args.length) {
      details.name = args[i + 1];
      args.splice(i, 2);
      i--;
    } else if (args[i] === '--email' && i + 1 < args.length) {
      details.email = args[i + 1];
      args.splice(i, 2);
      i--;
    } else if (args[i] === '--phone' && i + 1 < args.length) {
      details.phone = args[i + 1];
      args.splice(i, 2);
      i--;
    } else if (args[i] === '--notes' && i + 1 < args.length) {
      details.notes = args[i + 1];
      args.splice(i, 2);
      i--;
    }
  }
  
  // Remaining args form the query
  const query = args.join(' ') || 'Book me Goodman City at 7pm on 13th March';
  
  return { query, details };
}

// Helper: normalize time format (e.g., "8pm" -> "20:00")
function normalizeTime(query: string): string {
  return query
    .replace(/(\d{1,2}):?(\d{2})?\s*(am|pm)/gi, (match, hour, min, meridian) => {
      let h = parseInt(hour, 10);
      const m = min || '00';
      if (meridian.toLowerCase() === 'pm' && h !== 12) h += 12;
      if (meridian.toLowerCase() === 'am' && h === 12) h = 0;
      return `${h.toString().padStart(2, '0')}:${m}`;
    });
}

// Helper: extract booking parameters from query
function extractBookingParams(query: string): { 
  guests: number | null; 
  time: string | null; 
  date: string | null;
  normalizedQuery: string;
} {
  let normalizedQuery = normalizeTime(query);
  
  // Extract guests (number before "for" or after digits followed by space)
  const guestsMatch = query.match(/for\s+(\d+)\s+(people|person|guests?)?/i) || 
                      query.match(/(\d+)\s+(people|person|guests?)/i);
  const guests = guestsMatch ? parseInt(guestsMatch[1], 10) : null;
  
  // Extract time (24h format after normalization)
  const timeMatch = normalizedQuery.match(/(\d{2}:\d{2})/);
  const time = timeMatch ? timeMatch[1] : null;
  
  // Extract date - look for patterns like "March 27th", "27th March", "27/03", etc.
  const dateMatch = query.match(/(?:on\s+)?(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?/i) ||
                    query.match(/(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(\w+)/i);
  
  let date = null;
  if (dateMatch) {
    const monthNames = ['january', 'february', 'march', 'april', 'may', 'june', 
                        'july', 'august', 'september', 'october', 'november', 'december'];
    const monthIndex = monthNames.findIndex(m => 
      dateMatch[1].toLowerCase().startsWith(m) || dateMatch[2].toLowerCase().startsWith(m)
    );
    if (monthIndex !== -1) {
      const day = parseInt(dateMatch[1].match(/^\d+$/) ? dateMatch[1] : dateMatch[2], 10);
      const year = new Date().getFullYear();
      date = `${year}-${(monthIndex + 1).toString().padStart(2, '0')}-${day.toString().padStart(2, '0')}`;
    }
  }
  
  return { guests, time, date, normalizedQuery };
}

// 8. Run
const { query, details } = parseCliArgs();
startBooking(query, details)
  .then(async () => {
    // Cleanup: close browser
    try {
      await execAsync(`${AGENT_BROWSER} --json close`, { timeout: 5_000 });
    } catch {
      // Ignore cleanup errors
    }
  })
  .catch((err) => {
    console.error(`\x1b[31m[Fatal Error]:\x1b[0m`, err.message);
    process.exit(1);
  });
