"use strict";
const $ = id => document.getElementById(id);
let catalog, result, selectedKernel = 0, selectedPc = null, codeTab = "python", timelineGeometry;
const format = n => Number(n).toLocaleString("en-US");
const element = (tag, text, cls) => { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e; };

function backendChanged() {
  const sim = $("backend").value === "simgpu";
  $("tiled").disabled = sim;
  if (sim) $("tiled").checked = false;
  for (const id of ["m", "k", "n"]) $(id).max = sim ? 16 : 128;
  $("backend-note").textContent = sim
    ? "sim-gpu: M, K, N ≤ 16 · 한 스레드당 출력 원소 하나 · CPU tiling 미지원 · fusion 전후 커널 수와 메모리 접근을 비교해 보세요."
    : "C backend: M, K, N ≤ 128 · CPU tiling은 matmul의 루프 순서와 블록 크기를 바꿉니다.";
}

function renderGraph(target, graph, interactive) {
  target.replaceChildren();
  for (const node of graph.nodes) {
    const button = element(interactive ? "button" : "div", undefined, `graph-node ${node.op}`);
    button.dataset.node = node.id;
    button.append(element("span", node.name, "muted"), element("span", node.op, "op"),
      element("span", node.inputs.length ? "← " + node.inputs.join(", ") : "input", "deps"),
      element("span", `[${node.shape.join(" × ")}]`, "dims"));
    if (interactive) {
      button.type = "button";
      button.addEventListener("click", () => {
        const index = result.simulation?.kernels.findIndex(k => k.node_id === node.id) ?? -1;
        if (index >= 0) { selectedKernel = index; selectedPc = null; renderKernel(); $("gpu").scrollIntoView({block: "start"}); }
      });
    }
    target.append(button);
    if (node.body) target.append(element("div", node.body.filter(n => n.op !== "param").map(n => n.op).join(" → "), "inner-ops"));
  }
}

function renderCode() {
  $("generated-code").textContent = codeTab === "python" ? result.python_code : result.c_code;
  $("python-tab").classList.toggle("active", codeTab === "python");
  $("c-tab").classList.toggle("active", codeTab === "c");
}

function render() {
  $("results").hidden = false;
  $("model-source").textContent = result.source;
  $("seed").textContent = "FIXED SEED · " + result.seed;
  const preset = catalog.examples.find(e => e.id === result.config.example);
  $("description").textContent = preset.description;
  document.querySelector(".formula").firstChild.textContent = preset.formula;
  $("output-shape").textContent = "f32 [" + result.output.shape.join(" × ") + "]";
  $("output").textContent = result.output.preview.map(row => row.map(v => v.toFixed(4).padStart(9)).join(" ")).join("\n");
  $("checks").replaceChildren();
  const names = {python: "Python", interpreter: "IR", optimized_interpreter: "Optimized IR", c: "C", simgpu: "sim-gpu"};
  for (const [key, check] of Object.entries(result.checks)) {
    const e = element("span", `${check.status === "pass" ? "✓" : "!"} ${names[key]} · ${check.status}`, `check ${check.status}`);
    e.title = check.message || `max abs error: ${check.max_abs_error}`;
    $("checks").append(e);
  }
  $("tolerance").textContent = `NumPy reference와 비교 · rtol ${result.tolerance.rtol}, atol ${result.tolerance.atol} · 출력의 처음 4 × 4 표시`;
  renderGraph($("raw-graph"), result.raw, false);
  renderGraph($("optimized-graph"), result.optimized, true);
  $("raw-count").textContent = `${result.raw.operation_count} operations`;
  $("optimized-count").textContent = `${result.optimized.operation_count} operations`;
  $("ir").textContent = result.optimized.ir;
  $("branch-source").textContent = result.branch.source;
  $("branch-error").textContent = result.branch.error;
  renderCode();
  $("gpu").hidden = !result.simulation;
  if (result.simulation) {
    const sim = result.simulation;
    $("stats").replaceChildren();
    for (const [name, value] of [["KERNEL LAUNCHES", sim.kernels.length], ["SIMULATED CYCLES", sim.stats.cycles], ["GLOBAL LOADS / STORES", `${format(sim.stats.global_loads)} / ${format(sim.stats.global_stores)}`], ["MEMORY TRANSACTIONS", sim.stats.mem_transactions]]) {
      const card = element("div", undefined, "stat"); card.append(element("strong", typeof value === "number" ? format(value) : value), element("span", name)); $("stats").append(card);
    }
    $("kernel").replaceChildren(...sim.kernels.map((k, i) => { const o = element("option", `${i + 1}. ${k.name} · [${k.shape.join(" × ")}]`); o.value = i; return o; }));
    $("memory-layout").replaceChildren(...sim.layout.map(buffer => {
      const row = element("tr"); for (const value of [`${buffer.node_id} · ${buffer.name}`, buffer.address, buffer.words, buffer.shape.join(" × ")]) row.append(element("td", value)); return row;
    }));
    selectedKernel = 0; selectedPc = null; renderKernel();
  }
}

function currentKernel() { return result?.simulation?.kernels[selectedKernel]; }

function renderKernel() {
  const kernel = currentKernel(); if (!kernel) return;
  $("kernel").value = selectedKernel;
  const cfg = kernel.configuration;
  $("launch").textContent = `${cfg.num_blocks} blocks × ${cfg.block_size} threads · ${kernel.report.cycles} cycles · 각 커널의 cycle은 0부터 시작`;
  $("assembly-source").textContent = kernel.source;
  $("instructions").replaceChildren(...kernel.report.table.map(ins => {
    const row = element("tr"); row.dataset.pc = ins.pc; row.tabIndex = 0; row.setAttribute("role", "button");
    row.title = `Graph node: ${kernel.source_map[ins.pc] || kernel.node_id} · average active lanes ${ins.avgLanes.toFixed(2)}`;
    row.append(element("td", ins.pc), element("td", ins.text), element("td", ins.issues));
    const select = () => {
      selectedPc = ins.pc;
      const first = kernel.report.samples.find(s => s[2] === 0 && s[3] === ins.pc);
      if (first) $("start").value = Math.max(0, first[0] - 8);
      highlightInstruction(); drawTimeline();
    };
    row.addEventListener("click", select);
    row.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
    return row;
  }));
  document.querySelectorAll("#optimized-graph .graph-node").forEach(e => e.classList.toggle("selected", e.dataset.node === kernel.node_id));
  $("insights").replaceChildren(...kernel.report.insights.map(insight => {
    const p = element("p"); p.append(element("b", insight.title || ""), document.createTextNode(" " + (insight.body || insight.text || ""))); return p;
  }));
  $("start").value = 0;
  $("hover").textContent = "셀에 마우스를 올리면 cycle, warp와 active lanes가 표시됩니다.";
  highlightInstruction(); drawTimeline();
}

function highlightInstruction() {
  document.querySelectorAll("#instructions tr").forEach(e => e.classList.toggle("selected", Number(e.dataset.pc) === selectedPc));
}

function drawTimeline() {
  const kernel = currentKernel(); if (!kernel || $("gpu").hidden) return;
  const report = kernel.report, canvas = $("timeline"), visible = Math.min(Number($("window").value), report.cycles);
  $("start").max = Math.max(0, report.cycles - visible);
  const start = Number($("start").value), end = Math.min(report.cycles, start + visible);
  $("cycle-range").textContent = `${start}–${end - 1}`;
  const width = canvas.getBoundingClientRect().width || 600, height = Math.max(100, report.rows.length * 25 + 40), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const left = 88, top = 29, cell = (width - left - 12) / visible, rowHeight = 25;
  ctx.font = "9px ui-monospace, monospace"; ctx.fillStyle = "#64716c";
  const step = visible <= 64 ? 8 : visible <= 128 ? 16 : visible <= 256 ? 32 : 64;
  for (let c = Math.ceil(start / step) * step; c < end; c += step) ctx.fillText(c, left + (c - start) * cell, 16);
  report.rows.forEach((row, i) => { ctx.fillStyle = "#64716c"; ctx.fillText(`s${row.sm} b${row.block} w${row.warp}`, 8, top + i * rowHeight + 13); ctx.fillStyle = "#f0f3eb"; ctx.fillRect(left, top + i * rowHeight, width - left - 12, rowHeight - 3); });
  const colors = ["#237660", "#e2b75b", "#a7a5d1", "#e6eadf"], lookup = new Map();
  for (const sample of report.samples) {
    const [cycle, row, state, pc, mask] = sample; if (cycle < start || cycle >= end) continue;
    const x = left + (cycle - start) * cell, y = top + row * rowHeight;
    lookup.set(`${cycle}:${row}`, sample);
    ctx.fillStyle = colors[state];
    if (state === 0) {
      const laneHeight = (rowHeight - 3) / report.config.warpSize;
      for (let i = 0; i < mask.length; i++) if (mask[i] === "1") ctx.fillRect(x, y + i * laneHeight, Math.max(.5, cell - .3), Math.max(.5, laneHeight - .3));
    } else ctx.fillRect(x, y, Math.max(.5, cell - .3), rowHeight - 3);
    if (state === 0 && pc === selectedPc) { ctx.strokeStyle = "#ec693d"; ctx.lineWidth = 1.5; ctx.strokeRect(x, y, Math.max(1, cell), rowHeight - 3); }
  }
  timelineGeometry = {left, top, cell, rowHeight, start, end, lookup, rows: report.rows};
}

$("timeline").addEventListener("mousemove", event => {
  if (!timelineGeometry) return;
  const bounds = event.currentTarget.getBoundingClientRect(), g = timelineGeometry;
  const cycle = Math.floor((event.clientX - bounds.left - g.left) / g.cell) + g.start, row = Math.floor((event.clientY - bounds.top - g.top) / g.rowHeight);
  if (cycle < g.start || cycle >= g.end || !g.rows[row]) return;
  const s = g.lookup.get(`${cycle}:${row}`), r = g.rows[row];
  $("hover").textContent = `cycle ${cycle} · SM ${r.sm} / block ${r.block} / warp ${r.warp} · ` + (s ? ["issue", "memory stall", "barrier wait", "ready"][s[2]] + (s[2] === 0 ? ` · pc ${s[3]} · mask ${s[4]} · ${currentKernel().report.listing[s[3]]}` : "") : "inactive");
});

async function run(event) {
  if (event) event.preventDefault();
  if (!$("controls").reportValidity()) return;
  const config = Object.fromEntries(["example", "fusion", "backend"].map(id => [id, $(id).value]));
  for (const id of ["m", "k", "n"]) config[id] = Number($(id).value);
  config.tiled = $("tiled").checked;
  $("run").disabled = true; $("status").className = ""; $("status").textContent = "추적 → 최적화 → 컴파일 → 실행 중…";
  $("results").hidden = true;
  try {
    const response = await fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(config)});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    result = data; render();
    const problems = Object.values(data.checks).filter(c => c.status !== "pass");
    $("status").className = problems.length ? "error" : "";
    $("status").textContent = problems.length ? problems.map(c => c.message || "검증 실패: reference와 결과가 다릅니다.").join(" · ") : "검증 완료 · 모든 실행 결과가 NumPy reference와 일치합니다.";
  } catch (error) { $("status").className = "error"; $("status").textContent = error.message; }
  finally { $("run").disabled = false; }
}

$("controls").addEventListener("submit", run);
$("backend").addEventListener("change", backendChanged);
$("kernel").addEventListener("change", () => { selectedKernel = Number($("kernel").value); selectedPc = null; renderKernel(); });
$("window").addEventListener("change", drawTimeline); $("start").addEventListener("input", drawTimeline);
$("python-tab").addEventListener("click", () => { codeTab = "python"; renderCode(); });
$("c-tab").addEventListener("click", () => { codeTab = "c"; renderCode(); });
window.addEventListener("resize", drawTimeline);
(async () => {
  try {
    const response = await fetch("/api/examples"); if (!response.ok) throw new Error("예제를 불러오지 못했습니다."); catalog = await response.json();
    $("example").replaceChildren(...catalog.examples.map(example => { const o = element("option", example.title); o.value = example.id; return o; }));
    for (const [id, value] of Object.entries(catalog.defaults)) { if (id === "tiled") $(id).checked = value; else $(id).value = value; }
    backendChanged(); await run();
  } catch (error) { $("status").className = "error"; $("status").textContent = error.message; }
})();
