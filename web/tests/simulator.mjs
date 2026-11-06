// End-to-end compiler-to-simulator integration, using real compilation and execution.
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';
import path from 'node:path';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.BASE_URL || 'http://127.0.0.1:8000';
const artifacts = process.env.ARTIFACT_DIR || '/tmp/tensor2silicon-browser';
await mkdir(artifacts, {recursive: true});
const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_PATH ? {executablePath: process.env.BROWSER_PATH} : {})});
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [], runs = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('request', r => {if (r.url().endsWith('/api/run')) runs.push(r.postDataJSON());});
  const done = () => page.waitForFunction(() => !document.querySelector('#run-button').disabled && document.querySelector('#run-status').textContent.includes('✓ 실행 완료'));
  const step = index => page.locator(`[data-step="${index}"]`).click();
  async function run() {
    const [response] = await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/run') && r.request().method() === 'POST'), page.locator('#run-button').click()]);
    assert.equal(response.status(), 200);
    const data = await response.json();
    await done();
    assert.equal(await page.locator('#error-message').isHidden(), true);
    return data;
  }
  await page.goto(`${base}/`);
  await done();
  assert.equal(await page.locator('.site-nav a').count(), 2, 'one compiler lab and one architecture atlas');
  await step(5);
  assert.equal(await page.locator('[data-run-gpu]').isVisible(), true);
  // GPU execution starts from the common controls and keeps the current example.
  await page.locator('[data-run-gpu]').click();
  await done();
  assert.equal(await page.locator('#backend').inputValue(), 'simgpu');
  assert.equal(await page.locator('#gpu').isVisible(), true);
  assert.equal(await page.locator('#tiled').isDisabled(), true);
  for (const id of ['m','k','n']) await page.locator('#'+id).fill('3');
  for (const example of ['matmul','relu_linear','shared']) {
    for (const fusion of ['none','epilogue','full']) {
      await page.locator('#example').selectOption(example);
      await page.locator('#fusion').selectOption(fusion);
      const data = await run();
      assert.equal(data.checks.simgpu.status, 'pass');
      assert.equal(await page.locator('#kernel option').count(), data.simulation.kernels.length);
      assert.equal(await page.locator('.gpu-graph .graph-node.selected').count(), 1);
      console.log(`${example}/${fusion}: ${data.simulation.kernels.length} kernels, ${data.simulation.stats.cycles} cycles`);
    }
  }
  await page.locator('#example').selectOption('relu_linear');
  await page.locator('#fusion').selectOption('full');
  await page.locator('#m').fill('16'); await page.locator('#k').fill('8'); await page.locator('#n').fill('4');
  await run();
  const completed = runs.length;
  await step(3);
  await page.locator('[data-tab="assembly"]').click();
  assert.match(await page.locator('#code-panel pre').textContent(), /fmul/);
  await page.locator('[data-tab="assembly"]').focus();
  await page.keyboard.press('Home');
  assert.equal(await page.locator('[data-tab="python"]').getAttribute('aria-selected'), 'true');
  await page.keyboard.press('End');
  assert.equal(await page.locator('[data-tab="assembly"]').getAttribute('aria-selected'), 'true');
  await step(4);
  assert.equal(await page.locator('.check-status.pass').count(), 4);
  assert.match(await page.locator('.checks-table').textContent(), /sim-gpu/);
  await step(2);
  await page.locator('[data-graph="optimized"].fusion').click();
  await page.locator('[data-open-kernel]').click();
  assert.equal(await page.locator('#kernel').inputValue(), '0');
  assert.equal(runs.length, completed, 'stage changes share the completed run');
  await page.locator('#instructions tr').filter({hasText: 'fmul'}).first().click();
  assert.equal(await page.locator('#instructions tr.selected').count(), 1);
  assert.equal(await page.locator('#instructions tr.selected').getAttribute('aria-pressed'), 'true');
  await page.locator('#window').selectOption('64');
  await page.locator('#start').fill('16'); await page.locator('#start').dispatchEvent('input');
  assert.match(await page.locator('#cycle-range').textContent(), /^16–/);
  const canvas = await page.locator('#timeline').boundingBox();
  await page.mouse.move(canvas.x + 115, canvas.y + 40);
  assert.match(await page.locator('#hover').textContent(), /cycle .*SM/);
  await page.locator('#gpu').screenshot({path: path.join(artifacts, 'integrated-gpu-panel.png')});
  await page.screenshot({path: path.join(artifacts, 'integrated-gpu-desktop.png'), fullPage: true});
  await page.locator('#fusion').selectOption('none');
  const separate = await run();
  const last = separate.simulation.kernels.at(-1);
  await page.locator(`.gpu-graph [data-node="${last.node_id}"]`).click();
  assert.equal(await page.locator('#kernel').inputValue(), '3');
  await page.locator('#kernel').selectOption('1');
  assert.equal(await page.locator('.gpu-graph .graph-node.selected').getAttribute('data-node'), separate.simulation.kernels[1].node_id);
  // The architecture atlas uses the same document and retains the full experiment.
  const beforeArchitecture = runs.length;
  await page.locator('[data-view="architecture"]').click();
  await page.locator('.architecture-grid').waitFor();
  assert.equal(await page.locator('#compiler-view').isVisible(), false);
  await page.locator('[data-view="compiler"]').click();
  await page.locator('#gpu').waitFor({state: 'visible'});
  assert.equal(await page.locator('#gpu').isVisible(), true);
  assert.equal(await page.locator('#kernel').inputValue(), '1');
  assert.equal(runs.length, beforeArchitecture);
  // Validate backend-specific limits without changing the user's dimensions.
  await page.locator('#m').fill('17');
  assert.equal(await page.locator('#run-button').isDisabled(), true);
  assert.equal(await page.locator('#m').getAttribute('aria-invalid'), 'true');
  await page.locator('#m').fill('16');
  await page.emulateMedia({reducedMotion: 'reduce'});
  await page.setViewportSize({width: 390, height: 844});
  for (const i of [0,1,2,3,4,5]) {
    await step(i);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `stage ${i+1} fits mobile`);
  }
  await page.screenshot({path: path.join(artifacts, 'integrated-gpu-narrow.png'), fullPage: true});
  // Missing optional backend stays actionable and does not claim GPU execution.
  await page.route('**/api/run', async route => {
    const response = await route.fetch();
    const data = await response.json();
    data.simulation = null;
    data.checks.simgpu = {status: 'unavailable', message: '테스트: sim-gpu 없음'};
    await route.fulfill({response, json: data});
  });
  await run();
  assert.match(await page.locator('.gpu-prompt').textContent(), /sim-gpu 없음/);
  await step(4);
  assert.match(await page.locator('.results-header h2').textContent(), /sim-gpu 실행은 사용할 수 없습니다/);
  assert.equal(await page.locator('.check-status.pass').count(), 3);
  await page.unroute('**/api/run');
  // Switching to CPU replaces simulation data and removes GPU assembly tabs.
  await page.locator('#backend').selectOption('c');
  await page.locator('#tiled').check();
  const cpu = await run();
  assert.equal(cpu.checks.c.status, 'pass');
  await step(3);
  assert.equal(await page.locator('[data-tab="assembly"]').count(), 0);
  await step(5);
  assert.equal(await page.locator('#gpu').count(), 0);
  assert.equal(await page.locator('[data-run-gpu]').isVisible(), true);
  await page.goto(`${base}/simulator/`);
  await done();
  assert.equal(page.url(), `${base}/#gpu`);
  assert.equal(await page.locator('#backend').inputValue(), 'simgpu');
  assert.equal(await page.locator('#gpu').isVisible(), true);
  assert.deepEqual(errors, []);
  console.log('PASS: unified settings, 9 GPU runs, assembly tabs, graph-to-kernel selection, timeline, architecture state, mobile, optional backend failure, CPU fallback, legacy redirect.');
} finally {
  await browser.close();
}
