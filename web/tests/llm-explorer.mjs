// Offline browser smoke test. Playwright is a development tool, not an app dependency.
// PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs BROWSER_PATH=/path/to/chrome \
//   node web/tests/llm-explorer.mjs
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {existsSync} from "node:fs";
import {mkdir} from "node:fs/promises";
import path from "node:path";
import {fileURLToPath, pathToFileURL} from "node:url";

const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const url = pathToFileURL(path.join(root, "web/llm-explorer.html")).href;
const artifacts = process.env.ARTIFACT_DIR || "/tmp/llm-explorer-browser";
const python = process.env.PYTHON || (existsSync(path.join(root, ".venv/bin/python")) ? path.join(root, ".venv/bin/python") : "python3");
const spec = JSON.parse(execFileSync(python, ["-c", [
  "import json",
  "from dataclasses import asdict",
  "from llm_roofline.spec import PRESETS, op_specs",
  "p = PRESETS['colab']",
  "print(json.dumps({'model': asdict(p.model),",
  "'phases': {ph.name: asdict(ph) for ph in (p.prefill, p.decode)},",
  "'ops': {ph.name: [{'name': op.name, 'inputs': [a.shape for a in op.args], 'mm': op.mm_index}",
  "for op in op_specs(p.model, ph)] for ph in (p.prefill, p.decode)}}))",
].join("\n")], {cwd: root, encoding: "utf8"}));
await mkdir(artifacts, {recursive: true});
const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_PATH ? {executablePath: process.env.BROWSER_PATH} : {})});
let assertions = 0;
const check = (condition, message) => {assert.ok(condition, message); assertions++;};
const equal = (actual, expected, message) => {assert.deepEqual(actual, expected, message); assertions++;};

try {
  const context = await browser.newContext({viewport: {width: 1440, height: 900}, offline: true});
  await context.route(/^https?:/, route => route.abort());
  const page = await context.newPage();
  page.setDefaultTimeout(8000);
  const errors = [], requests = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {if (/^https?:/.test(request.url())) requests.push(request.url());});
  const node = id => page.locator('[data-node="' + id + '"]');
  const edge = (from, to) => page.locator('[data-from="' + from + '"][data-to="' + to + '"]');
  const camera = () => page.locator("#viewport").getAttribute("transform");
  const screenshot = name => page.screenshot({path: path.join(artifacts, name), fullPage: true, animations: "disabled"});
  const level = async value => {
    const slider = page.locator("#granularity");
    await slider.focus();
    await slider.press("Home");
    for (let i = 0; i < value; i++) await slider.press("ArrowRight");
    equal(await slider.inputValue(), String(value), "keyboard granularity reaches requested level");
    check(await slider.evaluate(el => el === document.activeElement), "slider retains keyboard focus");
  };

  await page.goto(url);
  equal(await page.locator("html").getAttribute("lang"), "ko", "Korean document language");
  const embedded = await page.locator("#model-config").evaluate(el => JSON.parse(el.textContent));
  equal(embedded.model, spec.model, "embedded model matches current Python specification");
  equal(embedded.phases, spec.phases, "phase settings match current Python specification");
  equal(await page.locator("[data-node]").count(), 7, "overview displays main flow and cache output");
  equal(await node("decoder").getAttribute("aria-pressed"), "true", "decoder selected by default");
  check(await page.locator("#back").isDisabled(), "initial history has no previous step");
  await screenshot("01-model-desktop.png");

  await node("embed").focus();
  await page.keyboard.press("Enter");
  equal(await page.locator(".node-heading h2").textContent(), "Embedding", "keyboard node selection updates inspector");
  equal(await node("embed").getAttribute("aria-pressed"), "true", "selected node accessible state");
  await node("decoder").click();
  await page.locator('[data-owner="decoder"]').click();
  equal(await page.locator("#granularity").inputValue(), "1", "plus button expands decoder");
  check(parseInt(await page.locator("#zoom-label").textContent()) >= 90, "expanded block opens at readable zoom");
  equal(await page.locator(".wire.residual").count(), 2, "both residual paths are explicit");
  check(await edge("embed", "residual1").count() === 1, "first residual bypasses pre-attention normalization");
  check(await edge("residual1", "residual2").count() === 1, "second residual bypasses pre-MLP normalization");
  await screenshot("02-block-desktop.png");
  await page.locator('[data-owner="mlp"]').click();
  equal(await page.locator("#granularity").inputValue(), "2", "MLP plus expands operations");
  equal(await node("gate_proj").getAttribute("aria-pressed"), "true", "MLP expansion selects gate projection");

  for (const phase of ["prefill", "decode"]) {
    await page.locator('[data-phase="' + phase + '"]').click();
    await level(2);
    const actual = await page.locator('[data-op]:not([data-op=""])').evaluateAll(els => els.map(el => ({
      name: el.dataset.op, inputs: JSON.parse(el.dataset.inputs), mm: Number(el.dataset.mm) || null,
    })));
    equal(actual.sort((a,b) => a.name.localeCompare(b.name)), spec.ops[phase].sort((a,b) => a.name.localeCompare(b.name)),
      phase + ": every operation, input shape and matmul index matches op_specs");
    equal(await page.locator('[data-mm]:not([data-mm=""])').count(), 9, phase + ": exactly nine per-layer matmuls");
    const B = spec.phases[phase].batch, T = spec.phases[phase].q_len;
    equal(JSON.parse(await node("logits").getAttribute("data-outputs")), [[B,T,32000]], phase + ": logits dimensions");
    equal(JSON.parse(await node("rope").getAttribute("data-outputs")), [[B,16,T,128],[B,16,T,128],[B,16,T,128]], phase + ": Q/K/V head dimensions");
    equal(JSON.parse(await node("qk").getAttribute("data-outputs")), [[B,16,T,2048]], phase + ": attention score dimensions");
    equal(JSON.parse(await node("cache").getAttribute("data-outputs")), [[B,16,2048,128]], phase + ": full cache dimensions");
    equal(await node("kv_write").count(), phase === "decode" ? 1 : 0, phase + ": cache write visibility");
    check(await edge("ln2", "gate_proj").count() === 1 && await edge("ln2", "up_proj").count() === 1, phase + ": MLP parallel branches");
    if (phase === "decode") {
      check(await edge("cache", "kv_write").count() === 1, "existing cache feeds write");
      check(await edge("kv_write", "qk").count() === 1 && await edge("kv_write", "pv").count() === 1, "updated K and V feed attention");
      await node("kv_write").focus();
      await page.keyboard.press("Enter");
      check((await page.locator(".explanation").textContent()).includes("2047"), "decode explains fixed last-slot write");
    } else {
      check(await edge("rope", "pv").count() === 1, "prefill V reaches PV directly");
      equal(await node("merge").getAttribute("data-op"), "merge_heads", "prefill head merge is counted");
    }
    await screenshot("03-operations-" + phase + ".png");
  }

  // Phase changes cannot leave a selected decode-only node in the inspector.
  await page.locator('[data-phase="prefill"]').click();
  equal(await node("cache").getAttribute("aria-pressed"), "true", "switching away from selected cache write selects cache");
  await page.locator("#fit").click();
  await page.locator("[data-collapse]").click();
  equal(await page.locator("#granularity").inputValue(), "1", "collapse returns to block level");
  await page.locator("#fit").click();
  await page.locator("[data-collapse]").click();
  equal(await page.locator("#granularity").inputValue(), "0", "collapse returns to overview");

  const original = await camera();
  await page.locator("#zoom-in").click();
  const enlarged = await camera();
  check(original !== enlarged, "zoom control changes viewport");
  await page.locator("#back").click();
  equal(await camera(), original, "Back restores viewport");
  await page.locator("#forward").click();
  equal(await camera(), enlarged, "Forward restores viewport");
  await page.locator("#fit").click();
  equal(await camera(), original, "fit restores full graph");
  const box = await page.locator("#graph").boundingBox();
  await page.mouse.move(box.x + 25, box.y + 35);
  await page.mouse.down();
  await page.mouse.move(box.x + 95, box.y + 95, {steps: 5});
  await page.mouse.up();
  check(await camera() !== original, "background drag pans viewport");
  await page.locator("#back").click();
  equal(await camera(), original, "pan gesture is recorded as one history step");
  await page.locator("#graph").focus();
  await page.keyboard.press("ArrowLeft");
  check(await camera() !== original, "keyboard arrow pans viewport");
  check(await page.locator("#forward").isDisabled(), "new action truncates forward history");
  await page.keyboard.press("0");
  equal(await camera(), original, "keyboard fit restores viewport");
  await page.keyboard.press("+");
  check(await camera() !== original, "keyboard zoom changes viewport");
  await page.locator("#fit").click();
  await page.mouse.move(box.x + box.width/2, box.y + box.height/2);
  await page.mouse.wheel(0, -220);
  await page.waitForTimeout(220);
  check(await camera() !== original, "wheel zoom changes viewport");
  await page.locator("#fit").click();

  // Restore phase, granularity and selection together.
  await page.locator('[data-phase="decode"]').click();
  await node("embed").click();
  await page.locator('[data-owner="decoder"]').click();
  await page.locator("#back").click();
  equal(await page.locator("#granularity").inputValue(), "0", "Back restores granularity");
  equal(await node("embed").getAttribute("aria-pressed"), "true", "Back restores selection");
  equal(await page.locator('[data-phase="decode"]').getAttribute("aria-pressed"), "true", "Back preserves phase snapshot");
  await page.locator("#back").click();
  await page.locator("#back").click();
  equal(await page.locator('[data-phase="prefill"]').getAttribute("aria-pressed"), "true", "Back restores earlier phase");

  await node("decoder").click();
  // Deterministically exercise allowed and rejected Clipboard API outcomes.
  await page.evaluate(() => Object.defineProperty(navigator, "clipboard", {configurable: true, value: {writeText: async text => {window.copiedText = text;}}}));
  await page.getByRole("button", {name: "구현 식별자 복사"}).click();
  equal(await page.evaluate(() => window.copiedText), "params.layers[0]", "copy uses exact implementation identifier");
  check((await page.locator("#notice").textContent()).includes("복사했습니다"), "copy success feedback");
  await page.evaluate(() => Object.defineProperty(navigator, "clipboard", {configurable: true, value: {writeText: async () => {throw new Error("Denied");}}}));
  await page.getByRole("button", {name: "구현 식별자 복사"}).click();
  check(await page.locator("#copy-fallback").isVisible(), "denied clipboard shows manual copy field");
  equal(await page.locator("#copy-fallback").inputValue(), "params.layers[0]", "fallback preserves identifier");
  check(await page.locator("#copy-fallback").evaluate(el => el.selectionStart === 0 && el.selectionEnd === el.value.length), "fallback selects all text");
  await page.getByRole("button", {name: "선택 해제"}).click();
  equal(await page.locator('[data-node][aria-pressed="true"]').count(), 0, "clear removes selected state");
  check(await page.locator(".about-title").isVisible(), "clear opens model information");
  await page.getByRole("button", {name: /Decoder block 살펴보기/}).click();
  equal(await node("decoder").getAttribute("aria-pressed"), "true", "model information links to decoder");

  await page.emulateMedia({reducedMotion: "reduce"});
  await page.waitForFunction(() => document.getElementById("notice").hidden);
  await page.setViewportSize({width: 390, height: 844});
  for (const value of [0,1,2]) {
    await level(value);
    check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "mobile level " + value + ": no page overflow");
    const graph = await page.locator("#graph").boundingBox(), inspector = await page.locator("#inspector").boundingBox();
    check(inspector.y > graph.y + graph.height, "mobile inspector follows graph");
    await page.locator("#fit").click();
    const content = await page.locator("#viewport").boundingBox();
    check(content.x >= graph.x && content.y >= graph.y &&
      content.x + content.width <= graph.x + graph.width &&
      content.y + content.height <= graph.y + graph.height,
      "mobile level " + value + ": fit contains the entire graph");
  }
  await level(0);
  await screenshot("04-model-mobile.png");
  await level(2);
  await page.locator('[data-phase="decode"]').click();
  await screenshot("05-operations-mobile.png");
  await level(0);
  await page.setViewportSize({width: 320, height: 740});
  check(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "320px narrow screen has no page overflow");

  // Touch pinch uses two pointers and should not select a node accidentally.
  const mobile = await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,offline:true});
  const touch = await mobile.newPage();
  touch.on("pageerror",error => errors.push(error.message));
  await touch.goto(url);
  const beforePinch = await touch.locator("#viewport").getAttribute("transform");
  const session = await mobile.newCDPSession(touch);
  const rect = await touch.locator("#graph").boundingBox();
  const cy = rect.y + rect.height/2, cx = rect.x + rect.width/2;
  await session.send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[{x:cx-30,y:cy,id:1},{x:cx+30,y:cy,id:2}]});
  await session.send("Input.dispatchTouchEvent",{type:"touchMove",touchPoints:[{x:cx-65,y:cy,id:1},{x:cx+65,y:cy,id:2}]});
  await session.send("Input.dispatchTouchEvent",{type:"touchEnd",touchPoints:[]});
  check(await touch.locator("#viewport").getAttribute("transform") !== beforePinch, "real touch pinch changes viewport");
  equal(await touch.locator('[data-node="decoder"]').getAttribute("aria-pressed"), "true", "pinch preserves selection");
  await mobile.close();
  equal(errors, [], "no browser errors");
  equal(requests, [], "standalone file makes no HTTP requests");
  console.log(JSON.stringify({status:"PASS",assertions,artifacts,verified:"offline file://; Python spec parity; desktop, mobile, keyboard and touch"},null,2));
} finally {
  await browser.close();
}
