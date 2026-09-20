// Optional compiler-to-simulator browser regression. See README for setup.
import assert from 'node:assert/strict';
import {mkdir} from 'node:fs/promises';
import path from 'node:path';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.BASE_URL || 'http://127.0.0.1:8000';
const artifacts = process.env.ARTIFACT_DIR || '/tmp/tensor2silicon-browser';
await mkdir(artifacts, {recursive: true});
(async () => {
  const browser = await chromium.launch({headless: true, ...(process.env.BROWSER_PATH ? {executablePath: process.env.BROWSER_PATH} : {})});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}, deviceScaleFactor: 1});
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(`${base}/simulator`);
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('검증 완료'));
  assert.equal(await page.locator('#gpu').isVisible(), false);
  await page.locator('#backend').selectOption('simgpu');
  assert.equal(await page.locator('#tiled').isDisabled(), true);
  for (const id of ['m','k','n']) await page.locator('#'+id).fill('3');
  async function run() {
    const [response] = await Promise.all([page.waitForResponse(r => r.url().endsWith('/api/run') && r.request().method() === 'POST'), page.locator('#run').click()]);
    assert.equal(response.status(), 200);
    const data = await response.json();
    await page.waitForFunction(() => !document.querySelector('#run').disabled);
    assert.match(await page.locator('#status').textContent(), /검증 완료/);
    return data;
  }
  for (const example of ['matmul','relu_linear','shared']) {
    for (const fusion of ['none','epilogue','full']) {
      await page.locator('#example').selectOption(example);
      await page.locator('#fusion').selectOption(fusion);
      const data = await run();
      assert.equal(data.checks.simgpu.status, 'pass');
      assert.equal(await page.locator('#gpu').isVisible(), true);
      assert.equal(await page.locator('#kernel option').count(), data.simulation.kernels.length);
      console.log(`${example}/${fusion}: ${data.simulation.kernels.length} kernels, ${data.simulation.stats.cycles} cycles`);
    }
  }
  await page.locator('#example').selectOption('relu_linear');
  await page.locator('#fusion').selectOption('full');
  await page.locator('#m').fill('16'); await page.locator('#k').fill('8'); await page.locator('#n').fill('4');
  await run();
  await page.locator('#gpu').scrollIntoViewIfNeeded();
  await page.locator('#instructions tr').filter({hasText: 'fmul'}).first().click();
  assert.equal(await page.locator('#instructions tr.selected').count(), 1);
  await page.locator('#window').selectOption('64');
  await page.locator('#start').fill('16'); await page.locator('#start').dispatchEvent('input');
  assert.match(await page.locator('#cycle-range').textContent(), /^16–/);
  const canvas = await page.locator('#timeline').boundingBox();
  await page.mouse.move(canvas.x + 115, canvas.y + 40);
  assert.match(await page.locator('#hover').textContent(), /cycle .*SM/);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({path:path.join(artifacts, 'compiler-gpu.png'), fullPage:true});
  await page.locator('#gpu').screenshot({path:path.join(artifacts, 'gpu-panel.png')});
  await page.locator('#fusion').selectOption('none');
  await run();
  await page.locator('#optimized-graph button').last().click();
  assert.equal(await page.locator('#kernel').inputValue(), '3');
  await page.locator('#kernel').selectOption('1');
  assert.equal(await page.locator('#optimized-graph button.selected').count(), 1);
  await page.locator('#backend').selectOption('c');
  await page.locator('#tiled').check();
  const cpu = await run(); assert.equal(cpu.checks.c.status, 'pass');
  assert.equal(await page.locator('#gpu').isVisible(), false);
  const bad = await page.request.post(`${base}/api/run`, {data: {backend:'simgpu',m:17}});
  assert.equal(bad.status(),400);
  await page.locator('.lab-links a[href="/#architecture"]').click();
  await page.locator('.architecture-grid').waitFor();
  assert.equal(await page.locator('#compiler-view').isVisible(), false);
  await page.locator('.view-switcher a[href="/simulator"]').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('검증 완료'));
  assert.deepEqual(errors, []);
  console.log('Browser flow, instruction selection, timeline, graph-to-kernel selection, C fallback, and validation passed.');
  await browser.close();
})().catch(error => {console.error(error);process.exit(1)});
