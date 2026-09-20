// Optional browser check; Playwright is a development tool, not an app dependency.
// BASE_URL=http://127.0.0.1:8001 PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node web/tests/smoke.mjs
import assert from "node:assert/strict";
import {mkdir} from "node:fs/promises";
import path from "node:path";

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.BASE_URL || "http://127.0.0.1:8000";
const artifacts = process.env.ARTIFACT_DIR || "/tmp/tensor2silicon-browser";
await mkdir(artifacts, {recursive: true});
const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_PATH ? {executablePath: process.env.BROWSER_PATH} : {})});
let assertions = 0;
const check = (condition, message) => {assert.ok(condition, message); assertions++;};

try {
  const context = await browser.newContext({viewport: {width: 1440, height: 1100}, permissions: ["clipboard-read", "clipboard-write"]});
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const done = async () => {
    await page.waitForFunction(() => document.querySelector("#run-status").textContent.includes("✓ 실행 완료"));
    check(await page.locator("#error-message").isHidden(), "successful run has no error banner");
  };
  const step = async (index) => page.locator(`[data-step="${index}"]`).click();
  const run = async () => {
    const response = page.waitForResponse((r) => r.url().endsWith("/api/run"));
    await page.locator("#run-button").click();
    const result = await response;
    check(result.ok(), "real compiler API succeeds");
    await done();
    return result.json();
  };

  await page.goto(url);
  await done();
  check((await page.locator("#lesson-title").textContent()).includes("함수"), "source lesson appears");
  check(await page.locator(".tensor-tile").count() === 3, "three actual inputs displayed");
  await page.screenshot({path: path.join(artifacts, "01-source-desktop.png"), fullPage: true});

  await page.locator("#next").click();
  check(await page.locator("[data-node]").count() === 8, "raw graph includes inputs, operations, output");
  await page.locator(".graph-node.matmul").click();
  check((await page.locator("#inspector").textContent()).includes("matmul"), "node inspection updates");
  const broadcast = page.locator(".graph-node.broadcast_in_dim");
  await broadcast.focus();
  await page.keyboard.press("Enter");
  check(await broadcast.getAttribute("aria-pressed") === "true", "graph selection works by keyboard");
  await page.locator(".ir-details summary").click();
  check((await page.locator(".ir-details pre").textContent()).includes("broadcast_in_dim"), "raw IR is inspectable");
  await page.screenshot({path: path.join(artifacts, "02-graph-desktop.png"), fullPage: true});

  await step(2);
  check(await page.locator('[data-graph-panel="optimized"] .graph-node.fusion').count() === 1, "full fusion rendered");
  check(await page.locator(".inner-node").count() > 4, "fusion body can be inspected");
  await page.screenshot({path: path.join(artifacts, "03-fusion-desktop.png"), fullPage: true});
  await page.locator("#fusion").selectOption("epilogue");
  check((await page.locator("#run-status").textContent()).includes("이전 실행"), "changed settings mark existing results stale");
  const epilogue = await run();
  check(epilogue.optimized.operation_count === 2, "epilogue keeps matmul separate");

  await page.locator("#example").selectOption("shared");
  await page.locator("#fusion").selectOption("full");
  await page.locator("#m").fill("7");
  await page.locator("#k").fill("5");
  await page.locator("#n").fill("3");
  await page.locator("#tiled").check();
  const shared = await run();
  check(shared.output.shape.join(",") === "7,3", "shape controls affect real output");
  check(await page.locator('[data-graph-panel="optimized"] .graph-node.matmul').count() === 1, "shared matmul remains visible");
  check(Object.values(shared.checks).every((c) => c.status === "pass"), "all real shared example execution paths agree");

  await step(3);
  await page.locator('[data-tab="python"]').focus();
  await page.keyboard.press("ArrowRight");
  check(await page.locator('[data-tab="c"]').getAttribute("aria-selected") === "true", "code tabs support keyboard arrows");
  check((await page.locator("#code-panel pre").textContent()).includes("ii"), "tiling loops appear in actual C source");
  await page.locator('[data-copy="c_code"]').click();
  check((await page.evaluate(() => navigator.clipboard.readText())).includes("void toy_run"), "copy control copies original source");

  await step(4);
  check(await page.locator(".check-status.pass").count() === 4, "four correctness checks displayed");
  await page.locator(".branch-example summary").click();
  check((await page.locator(".branch-error").textContent()).includes("cannot branch"), "real trace failure explained");
  await page.screenshot({path: path.join(artifacts, "05-results-desktop.png"), fullPage: true});

  await page.locator("#m").fill("");
  check(await page.locator("#run-button").isDisabled(), "empty dimension cannot run");
  check(await page.locator("#m").getAttribute("aria-invalid") === "true", "invalid dimension is accessible");
  await page.locator("#m").fill("129");
  check(await page.locator("#run-button").isDisabled(), "dimension bound enforced");
  await page.locator("#m").fill("7");

  await page.route("**/api/run", (route) => route.fulfill({status: 504, contentType: "application/json", body: JSON.stringify({error: "테스트: 실행 제한 시간 초과"})}));
  await page.locator("#run-button").click();
  await page.waitForFunction(() => !document.querySelector("#error-message").hidden);
  check((await page.locator("#error-message").textContent()).includes("제한 시간"), "timeout is visible");
  await page.unroute("**/api/run");
  await run();

  await page.route("**/api/run", async (route) => {
    const response = await route.fetch();
    const data = await response.json();
    data.checks.c = {status: "unavailable", message: "테스트: GCC 없음"};
    await route.fulfill({response, json: data});
  });
  await run();
  check((await page.locator(".results-header h2").textContent()).includes("사용할 수 없습니다"), "C unavailable does not claim full success");
  check(await page.locator(".check-status.pass").count() === 3, "Python results remain available");
  await page.unroute("**/api/run");
  await run();

  await page.emulateMedia({reducedMotion: "reduce"});
  await page.setViewportSize({width: 390, height: 844});
  for (const index of [0, 1, 2, 3, 4]) {
    await step(index);
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `stage ${index + 1} fits narrow viewport`);
  }
  await page.screenshot({path: path.join(artifacts, "05-results-narrow.png"), fullPage: true});
  await step(1);
  await page.screenshot({path: path.join(artifacts, "02-graph-narrow.png"), fullPage: true});
  check(errors.length === 0, `no browser script errors: ${errors.join("; ")}`);
  console.log(JSON.stringify({status: "PASS", assertions, screenshots: artifacts, browserErrors: errors}, null, 2));
} finally {
  await browser.close();
}
