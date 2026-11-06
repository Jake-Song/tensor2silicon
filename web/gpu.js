// The compiler owns requests and settings; this view only renders a completed run.
export function gpuPanel() {
  return `<section id="gpu" class="panel gpu-viewer"><div class="section-head"><h2>GPU에서 실행된 명령어</h2><span class="tag">SIMULATED · FLOAT32</span></div><p id="gpu-config-note" class="note">출력 원소 하나를 스레드 하나가 계산합니다. 블록은 8 threads, warp는 4 threads, SM은 2개입니다. Cycle은 이 시뮬레이터의 측정값이며 실제 GPU 시간 예측이 아닙니다.</p><div id="stats" class="stats"></div><div class="kernel-toolbar"><label>실행 커널<select id="kernel"></select></label><span id="launch" class="muted"></span></div>
      <div class="gpu-columns"><div><h3>SIMT assembly</h3><p class="note">명령어를 선택하면 timeline에 실행 위치가 표시됩니다.</p><div class="assembly-scroll"><table class="assembly"><thead><tr><th>PC</th><th>명령어</th><th>issue</th></tr></thead><tbody id="instructions"></tbody></table></div><details><summary>원본 assembly · labels 포함</summary><pre id="assembly-source"></pre></details></div><div class="timeline-panel"><div class="section-head"><h3>Warp timeline</h3><label class="window-label">표시 cycles<select id="window"><option>64</option><option selected>128</option><option>256</option><option>512</option></select></label></div><div class="legend"><span><i class="issue"></i>명령어 실행</span><span><i class="memory"></i>메모리 대기</span><span><i class="ready"></i>스케줄 대기</span><span><i class="barrier"></i>barrier</span></div><canvas id="timeline" height="200" aria-label="Warp별 명령어 실행과 메모리 대기 timeline"></canvas><div class="timeline-controls"><label for="start">시작 cycle</label><input id="start" type="range" value="0" min="0" max="0"><output id="cycle-range"></output></div><p id="hover" class="timeline-detail">셀에 마우스를 올리면 cycle, warp와 active lanes가 표시됩니다.</p><div id="insights" class="insights"></div></div></div>
      <details><summary>글로벌 메모리 · word 단위 버퍼 배치</summary><div class="table-scroll"><table><thead><tr><th>Graph node</th><th>Address</th><th>Words</th><th>Shape</th></tr></thead><tbody id="memory-layout"></tbody></table></div></details>
    </section>`;
}

export function mountGpu(root, simulation, options = {}) {
  const $ = id => root.querySelector(`#${id}`);
  const element = (tag, text, cls) => {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  };
  const format = n => Number(n).toLocaleString("en-US");
  let selectedKernel = Math.min(options.selectedKernel ?? 0, simulation.kernels.length - 1);
  let selectedPc = null, timelineGeometry;
  const sim = simulation;
  for (const [name, value] of [["KERNEL LAUNCHES", sim.kernels.length], ["SIMULATED CYCLES", sim.stats.cycles], ["GLOBAL LOADS / STORES", `${format(sim.stats.global_loads)} / ${format(sim.stats.global_stores)}`], ["MEMORY TRANSACTIONS", sim.stats.mem_transactions]]) {
    const card = element("div", undefined, "stat");
    card.append(element("strong", typeof value === "number" ? format(value) : value), element("span", name));
    $("stats").append(card);
  }
  $("kernel").replaceChildren(...sim.kernels.map((k, i) => {
    const o = element("option", `${i + 1}. ${k.name} · [${k.shape.join(" × ")}]`);
    o.value = i;
    return o;
  }));
  $("memory-layout").replaceChildren(...sim.layout.map(buffer => {
    const row = element("tr");
    for (const value of [`${buffer.node_id} · ${buffer.name}`, buffer.address, buffer.words, buffer.shape.join(" × ")]) row.append(element("td", value));
    return row;
  }));
  function currentKernel() { return simulation.kernels[selectedKernel]; }

  function renderKernel() {
    const kernel = currentKernel(); if (!kernel) return;
    $("kernel").value = selectedKernel;
    const cfg = kernel.configuration;
    $("gpu-config-note").textContent = `출력 원소 하나를 스레드 하나가 계산합니다. 블록은 ${cfg.block_size} threads, warp는 ${cfg.warp_size} threads, SM은 ${cfg.num_sms}개입니다. Cycle은 이 시뮬레이터의 측정값이며 실제 GPU 시간 예측이 아닙니다.`;
    $("launch").textContent = `${cfg.num_blocks} blocks × ${cfg.block_size} threads · ${kernel.report.cycles} cycles · 각 커널의 cycle은 0부터 시작`;
    $("assembly-source").textContent = kernel.source;
    $("instructions").replaceChildren(...kernel.report.table.map(ins => {
      const row = element("tr"); row.dataset.pc = ins.pc; row.tabIndex = 0; row.setAttribute("role", "button"); row.setAttribute("aria-pressed", "false");
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
    options.onKernelChange?.(selectedKernel, kernel.node_id);
    $("insights").replaceChildren(...kernel.report.insights.map(insight => {
      const p = element("p"); p.append(element("b", insight.title || ""), document.createTextNode(" " + (insight.body || insight.text || ""))); return p;
    }));
    $("start").value = 0;
    $("hover").textContent = "셀에 마우스를 올리면 cycle, warp와 active lanes가 표시됩니다.";
    highlightInstruction(); drawTimeline();
  }

  function highlightInstruction() {
    root.querySelectorAll("#instructions tr").forEach(e => {
      const selected = Number(e.dataset.pc) === selectedPc;
      e.classList.toggle("selected", selected);
      e.setAttribute("aria-pressed", String(selected));
    });
  }

  function drawTimeline() {
    const kernel = currentKernel(); if (!kernel || !root.isConnected || root.closest("[hidden]")) return;
    const report = kernel.report, canvas = $("timeline"), visible = Math.min(Number($("window").value), report.cycles);
    $("start").max = Math.max(0, report.cycles - visible);
    const start = Number($("start").value), end = Math.min(report.cycles, start + visible);
    $("cycle-range").textContent = `${start}–${end - 1}`;
    const width = canvas.getBoundingClientRect().width || 600, height = Math.max(100, report.rows.length * 25 + 40), ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
    const left = 88, top = 29, cell = (width - left - 12) / visible, rowHeight = 25;
    const theme = getComputedStyle(canvas);
    const color = name => theme.getPropertyValue(name).trim();
    ctx.font = "9px ui-monospace, monospace"; ctx.fillStyle = color("--muted");
    const step = visible <= 64 ? 8 : visible <= 128 ? 16 : visible <= 256 ? 32 : 64;
    for (let c = Math.ceil(start / step) * step; c < end; c += step) ctx.fillText(c, left + (c - start) * cell, 16);
    report.rows.forEach((row, i) => { ctx.fillStyle = color("--muted"); ctx.fillText(`s${row.sm} b${row.block} w${row.warp}`, 8, top + i * rowHeight + 13); ctx.fillStyle = color("--timeline-track"); ctx.fillRect(left, top + i * rowHeight, width - left - 12, rowHeight - 3); });
    const colors = ["--timeline-issue", "--timeline-memory", "--timeline-barrier", "--timeline-ready"].map(color), lookup = new Map();
    for (const sample of report.samples) {
      const [cycle, row, state, pc, mask] = sample; if (cycle < start || cycle >= end) continue;
      const x = left + (cycle - start) * cell, y = top + row * rowHeight;
      lookup.set(`${cycle}:${row}`, sample);
      ctx.fillStyle = colors[state];
      if (state === 0) {
        const laneHeight = (rowHeight - 3) / report.config.warpSize;
        for (let i = 0; i < mask.length; i++) if (mask[i] === "1") ctx.fillRect(x, y + i * laneHeight, Math.max(.5, cell - .3), Math.max(.5, laneHeight - .3));
      } else ctx.fillRect(x, y, Math.max(.5, cell - .3), rowHeight - 3);
      if (state === 0 && pc === selectedPc) { ctx.strokeStyle = color("--timeline-selection"); ctx.lineWidth = 1.5; ctx.strokeRect(x, y, Math.max(1, cell), rowHeight - 3); }
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
  $("kernel").addEventListener("change", () => {
    selectedKernel = Number($("kernel").value);
    selectedPc = null;
    renderKernel();
  });
  $("window").addEventListener("change", drawTimeline);
  $("start").addEventListener("input", drawTimeline);
  const observer = new ResizeObserver(drawTimeline);
  observer.observe($("timeline"));
  renderKernel();
  return {
    selectNode(id) {
      const index = simulation.kernels.findIndex(k => k.node_id === id);
      if (index < 0) return;
      selectedKernel = index;
      selectedPc = null;
      renderKernel();
    },
    destroy() { observer.disconnect(); },
  };
}
