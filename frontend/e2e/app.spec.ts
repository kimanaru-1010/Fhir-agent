import { test, expect } from "@playwright/test";
import type { APIRequestContext, Page } from "@playwright/test";

const BASE_URL = process.env.FRONTEND_URL || "http://localhost:3000";
const API_URL = process.env.API_URL || "http://localhost:8000";

// Timeout for LLM responses — these can be slow
const CHAT_TIMEOUT = 120_000;
const TEST_PASSWORD = "password123";

async function authenticatePage(page: Page, request: APIRequestContext, testTitle: string) {
  const username = `e2e_${testTitle.replace(/[^a-z0-9]+/gi, "_").slice(0, 40)}_${Date.now()}`;
  await request.post(`${API_URL}/api/auth/register`, {
    data: { username, password: TEST_PASSWORD },
  });
  const res = await request.post(`${API_URL}/api/auth/login`, {
    data: { username, password: TEST_PASSWORD },
  });
  expect(res.ok()).toBeTruthy();
  const body = await res.json();
  await page.addInitScript(({ token, user }) => {
    localStorage.setItem("fhir-agent-access-token", token);
    localStorage.setItem("fhir-agent-user", JSON.stringify(user));
  }, {
    token: body.access_token,
    user: body.user,
  });
}

test.describe("Healthcare Context Graph", () => {
  test.beforeEach(async ({ page, request }, testInfo) => {
    await authenticatePage(page, request, testInfo.title);
    await page.goto(BASE_URL);
  });

  // --------------------------------------------------------------------------
  // Basic page load
  // --------------------------------------------------------------------------

  test("page loads with header and chat panel", async ({ page }) => {
    // Header with domain name
    await expect(page.getByRole("heading", { name: /Healthcare/i })).toBeVisible();

    // Chat heading
    await expect(page.getByRole("heading", { name: /chat/i })).toBeVisible();

    // Chat input
    await expect(page.getByPlaceholder(/ask about/i)).toBeVisible();
  });

  test("demo scenario badges are visible", async ({ page }) => {
    // Should show demo scenario section
    await expect(page.getByText(/try these/i)).toBeVisible();

    // Should have clickable badges
    const badges = page.locator("[role='group'] span[data-scope='badge'], .chakra-badge").filter({ hasText: /.{10,}/ });
    const count = await badges.count();
    expect(count).toBeGreaterThan(0);
  });

  // --------------------------------------------------------------------------
  // Backend health
  // --------------------------------------------------------------------------

  test("backend health check returns ok or degraded", async ({ request }) => {
    const res = await request.get(`${API_URL}/health`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();
    expect(["ok", "degraded"]).toContain(body.status);
    expect(body.domain).toBe("healthcare");
  });

  test("connection status indicator visible", async ({ page }) => {
    // The header contains a colored status dot
    const dot = page.locator("[title*='Backend']");
    await expect(dot).toBeVisible({ timeout: 10_000 });
  });

  test("knowledge graph and trace panels are not rendered", async ({ page }) => {
    await expect(page.getByPlaceholder(/ask about/i)).toBeVisible();
    await expect(page.getByText(/knowledge graph/i)).toHaveCount(0);
    await expect(page.getByText(/decision traces/i)).toHaveCount(0);
    await expect(page.getByRole("button", { name: /graph panel/i })).toHaveCount(0);
    await expect(page.getByRole("button", { name: /details panel/i })).toHaveCount(0);
  });

  // --------------------------------------------------------------------------
  // Chat interaction with demo prompts
  // --------------------------------------------------------------------------

  test("demo prompt: Patient Lookup — sends and gets response", async ({ page }) => {
    test.setTimeout(CHAT_TIMEOUT);

    // Type the prompt
    const input = page.getByPlaceholder(/ask about/i);
    await input.fill("Show me all patients with a chronic diagnosis");
    await page.getByRole("button", { name: /send/i }).click();

    // Should show user message
    await expect(page.getByText("Show me all patients with a chronic diagnosis").first()).toBeVisible();

    // Should show loading state (thinking or tool calls)
    await expect(
      page.getByText(/thinking|running|generating/i).first()
    ).toBeVisible({ timeout: 10_000 });

    // Wait for assistant response (not an error)
    const assistantResponse = page.locator(".markdown-content").last();
    await expect(assistantResponse).toBeVisible({ timeout: CHAT_TIMEOUT });

    // Response should have meaningful content (not empty, not just an error)
    const text = await assistantResponse.textContent();
    expect(text).toBeTruthy();
    expect(text!.length).toBeGreaterThan(20);

    // Should NOT be an error message
    expect(text!.toLowerCase()).not.toContain("cannot reach the backend");
  });

  test("demo prompt: Clinical Decision Support — sends and gets response", async ({ page }) => {
    test.setTimeout(CHAT_TIMEOUT);

    // Type the prompt
    const input = page.getByPlaceholder(/ask about/i);
    await input.fill("Are there any potential drug interactions in current prescriptions?");
    await page.getByRole("button", { name: /send/i }).click();

    // Should show user message
    await expect(page.getByText("Are there any potential drug interactions in current prescriptions?").first()).toBeVisible();

    // Should show loading state (thinking or tool calls)
    await expect(
      page.getByText(/thinking|running|generating/i).first()
    ).toBeVisible({ timeout: 10_000 });

    // Wait for assistant response (not an error)
    const assistantResponse = page.locator(".markdown-content").last();
    await expect(assistantResponse).toBeVisible({ timeout: CHAT_TIMEOUT });

    // Response should have meaningful content (not empty, not just an error)
    const text = await assistantResponse.textContent();
    expect(text).toBeTruthy();
    expect(text!.length).toBeGreaterThan(20);

    // Should NOT be an error message
    expect(text!.toLowerCase()).not.toContain("cannot reach the backend");
  });

  test("demo prompt: Provider Network — sends and gets response", async ({ page }) => {
    test.setTimeout(CHAT_TIMEOUT);

    // Type the prompt
    const input = page.getByPlaceholder(/ask about/i);
    await input.fill("Which providers are affiliated with the largest hospital in the network?");
    await page.getByRole("button", { name: /send/i }).click();

    // Should show user message
    await expect(page.getByText("Which providers are affiliated with the largest hospital in the network?").first()).toBeVisible();

    // Should show loading state (thinking or tool calls)
    await expect(
      page.getByText(/thinking|running|generating/i).first()
    ).toBeVisible({ timeout: 10_000 });

    // Wait for assistant response (not an error)
    const assistantResponse = page.locator(".markdown-content").last();
    await expect(assistantResponse).toBeVisible({ timeout: CHAT_TIMEOUT });

    // Response should have meaningful content (not empty, not just an error)
    const text = await assistantResponse.textContent();
    expect(text).toBeTruthy();
    expect(text!.length).toBeGreaterThan(20);

    // Should NOT be an error message
    expect(text!.toLowerCase()).not.toContain("cannot reach the backend");
  });

  // --------------------------------------------------------------------------
  // Demo badge click flow
  // --------------------------------------------------------------------------

  test("clicking a demo badge sends the prompt", async ({ page }) => {
    test.setTimeout(CHAT_TIMEOUT);

    // Find and click the first demo badge
    const badge = page.locator(".chakra-badge[title]").first();
    const promptText = await badge.getAttribute("title");
    expect(promptText).toBeTruthy();

    await badge.click();

    // Should show user message with the badge prompt
    await expect(page.getByText(promptText!).first()).toBeVisible({ timeout: 5_000 });

    // Should eventually get an assistant response
    const assistantResponse = page.locator(".markdown-content").last();
    await expect(assistantResponse).toBeVisible({ timeout: CHAT_TIMEOUT });
  });

  // --------------------------------------------------------------------------
  // Tool call visualization
  // --------------------------------------------------------------------------

  test("tool calls show timeline with status indicators", async ({ page }) => {
    test.setTimeout(CHAT_TIMEOUT);

    // Send a prompt that should trigger tool calls
    const input = page.getByPlaceholder(/ask about/i);
    await input.fill("Show me all patients with a chronic diagnosis");
    await page.getByRole("button", { name: /send/i }).click();

    // Wait for at least one tool call badge to appear
    const toolBadge = page.locator("[data-scope='badge']").filter({ hasText: /execute_cypher|get_schema|search_patient/ });
    await expect(toolBadge.first()).toBeVisible({ timeout: 30_000 });
  });

  // --------------------------------------------------------------------------
  // New conversation
  // --------------------------------------------------------------------------

  test("new conversation button resets chat", async ({ page }) => {
    test.setTimeout(CHAT_TIMEOUT);

    // Send a message first
    const input = page.getByPlaceholder(/ask about/i);
    await input.fill("Hello");
    await page.getByRole("button", { name: /send/i }).click();

    // Wait for response
    await expect(page.locator(".markdown-content").last()).toBeVisible({ timeout: CHAT_TIMEOUT });

    // Click "New" button
    await page.getByRole("button", { name: /new/i }).click();

    // Demo scenarios should be visible again
    await expect(page.getByText(/try these/i)).toBeVisible();
  });

  // --------------------------------------------------------------------------
  // Mobile navigation (viewport 375px)
  // --------------------------------------------------------------------------

  test("mobile: chat remains the only main panel", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto(BASE_URL);

    await expect(page.getByPlaceholder(/ask about/i)).toBeVisible();
    await expect(page.getByRole("button", { name: /graph panel/i })).toHaveCount(0);
    await expect(page.getByRole("button", { name: /details panel/i })).toHaveCount(0);
    await expect(page.getByText(/traces/i)).toHaveCount(0);
  });

  // --------------------------------------------------------------------------
  // Regression tests for v0.12.0 / v0.13.0 frontend bug fixes
  // --------------------------------------------------------------------------

  test("composite keys do not trigger React duplicate-key warnings across renders", async ({ page }) => {
    // Send a few prompts in sequence. React logs a console.error if entity,
    // preference, or tool-call badges collide on a non-unique key — pre-v0.13.0
    // they used `key={i}` (the array index), which would fire here.
    const keyWarnings: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "warning" || msg.type() === "error") {
        const text = msg.text();
        if (
          text.includes("Encountered two children with the same key") ||
          text.includes("Each child in a list should have a unique") ||
          text.includes("key prop")
        ) {
          keyWarnings.push(text);
        }
      }
    });

    const input = page.locator("textarea, input[type='text']").first();
    if (await input.count() === 0) test.skip(true, "no chat input found");

    for (const prompt of [
      "Tell me about the data",
      "What entities exist here?",
      "Show me a summary of recent activity",
    ]) {
      await input.fill(prompt);
      await page.keyboard.press("Enter");
      // Allow streaming + tool-call badges to flush before sending the next.
      await page.waitForTimeout(3000);
    }
    expect(keyWarnings).toEqual([]);
  });

  // --------------------------------------------------------------------------
  // API-level prompt quality checks
  // --------------------------------------------------------------------------

  test("API: Patient Lookup prompt 1 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "Show me all patients with a chronic diagnosis" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });

  test("API: Patient Lookup prompt 2 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "What medications are currently prescribed to patients in the cardiology department?" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });

  test("API: Patient Lookup prompt 3 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "Find all recent patient encounters in the last 6 months" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });

  test("API: Clinical Decision Support prompt 1 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "Are there any potential drug interactions in current prescriptions?" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });

  test("API: Clinical Decision Support prompt 2 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "What treatments have been most effective for patients with heart failure?" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });

  test("API: Provider Network prompt 1 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "Which providers are affiliated with the largest hospital in the network?" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });

  test("API: Provider Network prompt 2 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "Show the referral patterns between primary care and specialists" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });

  test("API: Provider Network prompt 3 returns quality response", async ({ request }) => {
    test.setTimeout(CHAT_TIMEOUT);

    const res = await request.post(`${API_URL}/api/chat`, {
      data: { message: "Which providers have the most patient encounters this quarter?" },
    });
    expect(res.ok()).toBeTruthy();

    const body = await res.json();

    // Should have a response string
    expect(body.response).toBeTruthy();
    expect(typeof body.response).toBe("string");
    expect(body.response.length).toBeGreaterThan(50);

    // Should have a session_id
    expect(body.session_id).toBeTruthy();

    // Response should not be a generic error
    expect(body.response.toLowerCase()).not.toContain("i apologize");
    expect(body.response.toLowerCase()).not.toContain("i don't have access");
  });
});
