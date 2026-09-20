// Use the same BASE_URL / PLAYWRIGHT_MODULE configuration as smoke.mjs.
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
  const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
  page.setDefaultTimeout(10000);
  const errors = [], apiRequests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {if (request.url().includes("/api/")) apiRequests.push(request.url());});
  const part = (family, key) => page.locator(`[data-family="${family}"][data-part="${key}"]`).first();
  const depth = async (value) => page.locator(`#arch-level-${value}`).click();
  const screenshot = async (name) => page.screenshot({path: path.join(artifacts, name), fullPage: true});

  await page.goto(`${url}/#architecture`);
  await page.locator(".architecture-grid").waitFor();
  check(await page.locator("#compiler-view").isHidden(), "compiler view hidden on structural route");
  check(await page.locator(".arch-device").count() === 2, "GPU and TPU appear together");
  check(await page.locator('[data-family="gpu"][data-part="sm"]').count() === 6, "overview displays repeated SM samples");
  check(await page.locator('[data-family="tpu"][data-part="core"]').count() === 1, "v5e overview displays one TensorCore");
  check((await page.locator(".arch-scale-note").textContent()).includes("실제 배치"), "conceptual scale is disclosed");
  check((await page.locator(".arch-name-note").textContent()).includes("구성 단위가 다릅니다"), "Tensor Core naming distinction explained");
  await screenshot("architecture-01-overview-desktop.png");

  await part("gpu", "hbm").click();
  check((await page.locator(".arch-containment").textContent()).includes("die 외부"), "HBM is outside processor die");
  check(await part("gpu", "hbm").getAttribute("aria-pressed") === "true", "selected block is accessible");
  await part("gpu", "sm").focus();
  await page.keyboard.press("Enter");
  check(await page.locator("#arch-level-1").getAttribute("aria-current") === "step", "SM click zooms to core level");
  check(await page.locator('[data-family="gpu"][data-part="tensor"]').count() === 4, "A100 SM shows four Tensor Cores");
  check(await page.locator('[data-family="tpu"][data-part="mxu"]').count() === 4, "v5e core shows four MXUs");
  check(await page.locator('[data-family="tpu"][data-part="vmem"]').count() === 1, "TPU local memory is represented");
  await part("tpu", "smem").click();
  check((await page.locator(".arch-part-fact").textContent()).includes("scalar memory"), "SMEM abbreviation is disambiguated");
  await screenshot("architecture-02-cores-desktop.png");

  await part("tpu", "mxu").focus();
  await page.keyboard.press(" ");
  check(await page.locator("#arch-level-2").getAttribute("aria-current") === "step", "MXU click zooms to unit level");
  check(await page.locator(".arch-mac-cell").count() === 64, "abbreviated static MAC grid appears");
  check((await page.locator(".arch-part-fact").textContent()).includes("128 × 128"), "true MXU grid size is distinguished from illustration");
  await part("tpu", "mac").click();
  check(await page.locator("#arch-part-title").textContent() === "MAC cell", "MAC cell detail appears");
  check((await page.locator(".arch-containment").textContent()).includes("TensorCoreMXUMAC cell"), "MAC containment hierarchy is correct");
  await part("gpu", "tensor").click();
  check(await page.locator("#arch-part-title").textContent() === "Tensor Core", "GPU matrix unit remains independently selectable");
  await screenshot("architecture-03-matrix-desktop.png");

  await page.locator(".arch-reset").click();
  check(await page.locator("#arch-level-0").getAttribute("aria-current") === "step", "reset returns to overview");
  await part("tpu", "core").click();
  check(await page.locator("#arch-part-title").textContent() === "TensorCore", "TPU parent block remains the selected detail after zoom");
  await page.locator('.architecture-depth-nav [data-depth="0"]').click();
  check(await page.locator("#arch-level-0").getAttribute("aria-current") === "step", "sidebar depth navigation works");
  await page.locator(".arch-sources summary").click();
  check(await page.locator(".arch-sources a").count() === 4, "official references are available");
  check(apiRequests.length === 0, "structure exploration makes no compiler execution requests");

  // The existing lab retains its current configuration across view changes.
  await page.locator('[data-view="compiler"]').click();
  await page.waitForFunction(() => document.querySelector("#run-status").textContent.includes("✓ 실행 완료"));
  await page.locator("#m").fill("12");
  await page.locator('[data-view="architecture"]').click();
  await page.locator("#architecture-view").waitFor({state: "visible"});
  check(await page.locator("#architecture-view").isVisible(), "view switch to structure works");
  await page.goBack();
  await page.locator("#compiler-view").waitFor({state: "visible"});
  check(await page.locator("#compiler-view").isVisible(), "browser Back restores compiler view");
  check(await page.locator("#m").inputValue() === "12", "compiler settings survive view switching");
  await page.goForward();
  await page.locator("#architecture-view").waitFor({state: "visible"});
  check(await page.locator("#architecture-view").isVisible(), "browser Forward restores structure view");
  await page.locator(".skip-link").focus();
  await page.keyboard.press("Enter");
  check(await page.locator("#architecture-view").isVisible(), "skip link keeps active view");

  await page.emulateMedia({reducedMotion: "reduce"});
  await page.setViewportSize({width: 390, height: 844});
  for (const value of [0, 1, 2]) {
    await depth(value);
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `structure depth ${value} fits narrow viewport`);
  }
  await screenshot("architecture-03-matrix-narrow.png");
  await depth(0);
  await screenshot("architecture-01-overview-narrow.png");
  check(errors.length === 0, `no browser errors: ${errors.join("; ")}`);
  console.log(JSON.stringify({status: "PASS", assertions, screenshots: artifacts}, null, 2));
} finally {
  await browser.close();
}
