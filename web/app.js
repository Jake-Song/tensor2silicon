import {mountArchitecture} from "./architecture.js";

const $ = (selector) => document.querySelector(selector);
const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const shape = (dims) => `[${dims.join(" × ")}]`;
const pad = (n) => String(n).padStart(2, "0");
const state = {step: 0, data: null, busy: false, selected: null, codeTab: "python", examples: []};
const stages = [
  ["THE SOURCE", "모든 것은 함수에서 시작됩니다.", "익숙한 Python 식이 컴파일러의 언어로 바뀌는 과정을 따라가 보세요."],
  ["GRAPH CAPTURE", "계산하지 않고, 연산을 기록합니다.", "값 대신 shape을 가진 Tensor를 전달하면, 연산과 의존 관계가 그래프로 남습니다."],
  ["GRAPH FUSION", "함께 계산할 연산을 찾아봅니다.", "중간 배열을 줄일 수 있도록 연결된 연산을 합칩니다. 모든 연산을 합칠 수 있는 것은 아닙니다."],
  ["CODE GENERATION", "그래프가 실행할 코드로 바뀝니다.", "같은 계산을 NumPy 함수와 C 루프로 표현합니다. 실제 생성된 소스를 비교해 보세요."],
  ["EXECUTE & VERIFY", "표현이 달라도, 결과는 같아야 합니다.", "원래 함수의 NumPy 결과를 기준으로 각 실행 경로가 같은 값을 계산하는지 확인합니다."],
];
const explanations = {
  input: "함수에 전달하는 입력입니다. 추적할 때는 실제 숫자 대신 shape과 dtype만 가진 추상 Tensor가 됩니다.",
  matmul: "X[M, K]와 W[K, N]의 K축을 곱하고 더합니다. 결과는 [M, N]입니다. 각 출력 원소에 K개의 곱셈이 필요합니다.",
  broadcast_in_dim: "b[N]을 [M, N]에 맞춥니다. Python에서 암묵적이었던 broadcasting이 IR에서는 명시적인 노드가 됩니다.",
  add: "같은 shape인 두 텐서의 원소를 각각 더합니다. 입력 shape이 다르면 먼저 broadcasting이 필요합니다.",
  relu: "각 원소를 max(x, 0)으로 바꿉니다. 음수는 0이 되고, 양수는 그대로 통과합니다.",
  fusion: "여러 연산을 하나의 계산 영역에 담았습니다. C 코드는 내부 연산을 스칼라 계산으로 펼쳐 중간 배열을 줄입니다.",
  output: "함수가 돌려주는 값입니다. 마지막 연산이 만든 텐서를 반환하며, 별도 계산을 추가하지 않습니다.",
};

function highlightedLine(line) {
  if (/^\s*(#|\/\/)/.test(line)) return `<span class="token-comment">${escape(line)}</span>`;
  const keywords = /^(def|return|if|raise|import|as|for|in|void|float|const|int)$/;
  const functions = /^(relu|matmul|forward|np|range|maximum|asarray|broadcast_to|expand_dims)$/;
  return line.split(/(\b(?:def|return|if|raise|import|as|for|in|void|float|const|int|relu|matmul|forward|np|range|maximum|asarray|broadcast_to|expand_dims)\b|\b\d+(?:\.\d+)?\b)/g).map((part) => {
    const cls = keywords.test(part) ? "token-keyword" : functions.test(part) ? "token-function" : /^\d/.test(part) ? "token-number" : "";
    return cls ? `<span class="${cls}">${escape(part)}</span>` : escape(part);
  }).join("");
}

function codePanel(source, title, key, caption = "") {
  const code = source.trimEnd().split("\n").map((line, i) => `<span class="code-line"><span class="line-number" aria-hidden="true">${i + 1}</span>${highlightedLine(line)}</span>`).join("");
  return `<section class="panel"><div class="panel-head"><div class="code-toolbar"><span class="window-dots" aria-hidden="true"><i></i><i></i><i></i></span><span class="panel-label">${escape(title)}</span></div><button class="copy-button" data-copy="${key}" aria-label="${escape(title)} 복사">복사</button></div><pre tabindex="0"><code>${code}</code></pre>${caption ? `<div class="code-caption">${caption}</div>` : ""}</section>`;
}

function callout(title, text) {
  return `<div class="lesson-callout"><span class="callout-icon" aria-hidden="true">↳</span><div><h3>${title}</h3><p>${text}</p></div></div>`;
}

function matrixPart(name, dimensions, result = false) {
  return `<div class="shape-part ${result ? "result" : ""}"><b>${name}</b><div class="mini-matrix" aria-hidden="true">${"<i></i>".repeat(12)}</div><small>${shape(dimensions)}</small></div>`;
}

function sourceStage() {
  const d = state.data, c = d.config;
  const example = state.examples.find((e) => e.id === c.example);
  const inputs = d.raw.nodes.filter((n) => n.op === "input");
  return `<div class="source-grid">
    ${codePanel(d.source, "model.py", "source", "<code>relu</code>는 toy의 연산입니다. 이 함수에 일반 배열을 넣으면 eager로 계산하고, 추상 Tensor를 넣으면 연산을 기록합니다.")}
    <div class="source-side"><section class="panel formula-card"><span class="section-kicker">ONE EXPRESSION, MANY REPRESENTATIONS</span><div class="formula">${escape(example.formula)}</div><p>${escape(example.description)}</p></section>
      <section class="panel"><div class="panel-head"><h2>행렬 곱의 shape</h2><span class="pill">M × K · K × N</span></div><div class="shape-equation">${matrixPart("X", [c.m, c.k])}<span class="shape-operator">@</span>${matrixPart("W", [c.k, c.n])}<span class="shape-operator">=</span>${matrixPart("XW", [c.m, c.n], true)}</div></section>
    </div></div>
    <div class="tensor-strip">${inputs.map((n) => `<div class="tensor-tile"><span class="tensor-symbol">${escape(n.name.toUpperCase())}</span><div><strong>${shape(n.shape)} <span class="muted">· f32</span></strong><small>${n.name === "x" ? "입력 · M개의 행" : n.name === "w" ? "가중치 · K축으로 연결" : "편향 · N개의 원소"}</small></div></div>`).join("")}</div>
    ${callout("Python을 실행하는 두 가지 방법", "지금은 실제 배열로 정답을 계산했습니다. 다음 단계에서는 동일한 함수에 <code>shape</code>과 <code>dtype</code>만 전달합니다. 덧셈과 행렬 곱이 실행되는 대신, 그래프의 노드로 기록됩니다.")}
    <div class="exercise"><span class="exercise-label">TRY IT</span><span>M을 바꾸고 실행해 보세요. W의 shape은 그대로인데, 출력의 행 수는 왜 바뀔까요?</span></div>`;
}

function graphNodes(key) {
  const graph = state.data[key];
  const root = graph.nodes.find((n) => n.id === graph.outputs[0]);
  return [...graph.nodes, {id: "output", name: "Y", op: "output", inputs: graph.outputs, shape: root.shape, dtype: root.dtype, attrs: {}}];
}

function graphPanel(key, title) {
  const graph = state.data[key], nodes = graphNodes(key), depths = new Map(), levels = [];
  nodes.forEach((node) => {
    const depth = node.inputs.length ? Math.max(...node.inputs.map((id) => depths.get(id))) + 1 : 0;
    depths.set(node.id, depth);
    (levels[depth] ??= []).push(node);
  });
  const boxW = 176, boxH = 56, gap = 24, rowH = 88;
  const width = Math.max(400, Math.max(...levels.map((row) => row.length)) * (boxW + gap) + 16);
  const height = levels.length * rowH + 10;
  const positions = new Map();
  levels.forEach((row, depth) => row.forEach((node, i) => positions.set(node.id, {x: (width - row.length * (boxW + gap) + gap) / 2 + i * (boxW + gap), y: 14 + depth * rowH})));
  const paths = nodes.flatMap((node) => node.inputs.map((id) => {
    const p = positions.get(id), q = positions.get(node.id), x1 = p.x + boxW / 2, x2 = q.x + boxW / 2, y1 = p.y + boxH, y2 = q.y;
    // A shared value can skip several layers. Route it outside the nodes so
    // the edge cannot appear to pass through an unrelated operation.
    const startY = p.y + boxH / 2, endY = q.y + boxH / 2;
    const path = depths.get(node.id) - depths.get(id) > 1
      ? `M${p.x},${startY} H24 Q12,${startY} 12,${startY + 12} V${endY - 12} Q12,${endY} 24,${endY} H${q.x}`
      : `M${x1},${y1} C${x1},${y1 + 19} ${x2},${y2 - 19} ${x2},${y2}`;
    return `<path class="graph-edge" data-from="${id}" data-to="${node.id}" d="${path}" marker-end="url(#arrow-${key})"/>`;
  })).join("");
  const boxes = nodes.map((node) => {
    const p = positions.get(node.id), name = node.op === "input" ? `${node.name} · input` : node.op === "output" ? "Y · output" : node.op;
    return `<g class="graph-node ${node.op}" data-node="${node.id}" data-graph="${key}" tabindex="0" role="button" aria-label="${escape(node.name)} ${node.op} ${shape(node.shape)}" aria-pressed="false" transform="translate(${p.x},${p.y})"><rect width="${boxW}" height="${boxH}" rx="6"/><text class="node-title" x="${boxW / 2}" y="22" text-anchor="middle">${escape(name)}</text><text class="node-subtitle" x="${boxW / 2}" y="42" text-anchor="middle">${node.op === "input" || node.op === "output" ? "" : escape(node.name) + " · "}${shape(node.shape)}</text></g>`;
  }).join("");
  return `<section class="panel graph-container" data-graph-panel="${key}"><div class="panel-head"><h2>${title}</h2><span class="pill ${key === "optimized" ? "amber" : ""}">${graph.operation_count} ops</span></div><div class="graph-canvas" tabindex="0" aria-label="${title} 스크롤 영역"><svg viewBox="0 0 ${width} ${height}" style="--graph-width:${width}px" role="group" aria-label="${title} 의존 관계 그래프"><defs><marker id="arrow-${key}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M1,1 L6,3.5 L1,6" fill="none" stroke="#708f9f"/></marker></defs>${paths}${boxes}</svg></div><div class="graph-key"><span><i></i>노드를 선택해 살펴보세요</span>${key === "optimized" ? '<span><i class="fusion-key"></i>fusion</span>' : '<span>↓ data flow</span>'}</div></section>`;
}

function inspector() {
  const key = state.selected?.graph ?? (state.step === 2 ? "optimized" : "raw");
  const nodes = graphNodes(key);
  const node = nodes.find((n) => n.id === state.selected?.id) ?? nodes.find((n) => !["input", "output"].includes(n.op));
  state.selected = {graph: key, id: node.id};
  const inputs = node.inputs.map((id) => nodes.find((n) => n.id === id)?.name ?? id);
  const attrs = Object.entries(node.attrs).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("\n");
  return `<div class="inspector"><span class="section-kicker">NODE INSPECTOR · ${key === "raw" ? "RAW" : "OPTIMIZED"}</span><h3>${escape(node.op)}</h3><p>${explanations[node.op] ?? "그래프의 내부 연산입니다."}</p><dl><div><dt>노드 이름</dt><dd>${escape(node.name)}</dd></div><div><dt>입력 의존 관계</dt><dd>${escape(inputs.join(", ") || "함수 입력")}</dd></div><div><dt>출력 shape / dtype</dt><dd>${shape(node.shape)} · ${node.dtype}</dd></div>${attrs ? `<div><dt>속성</dt><dd>${escape(attrs)}</dd></div>` : ""}</dl>${node.body ? `<details open><summary>Fusion 내부 보기 · ${node.body.filter((n) => n.op !== "param").length}개 연산</summary>${node.body.map((inner) => `<div class="inner-node">${escape(inner.name)} = ${escape(inner.op)}<small>${inner.op === "param" ? "외부 입력 " + escape(inputs[inner.attrs.index]) : escape(inner.inputs.map((id) => node.body.find((n) => n.id === id).name).join(", "))} · ${shape(inner.shape)}</small></div>`).join("")}</details>` : ""}</div>`;
}

function rawStage() {
  return `<div class="graph-and-detail">${graphPanel("raw", "추출된 그래프")}<div id="inspector">${inspector()}</div></div>
    <details class="ir-details"><summary>텍스트 IR도 함께 읽어보기</summary><pre tabindex="0">${escape(state.data.raw.ir)}</pre></details>
    ${callout("화살표는 데이터의 의존 관계입니다", "<code>matmul</code>이 만든 값을 <code>add</code>가 읽습니다. <code>b[N]</code>처럼 차원이 다른 입력을 더할 때는 <code>broadcast_in_dim</code>이 별도 노드로 드러납니다. 이 그래프는 현재 입력 shape에 특수화됩니다.")}`;
}

function fusionStage() {
  const d = state.data;
  const note = d.config.example === "shared" ? "공유 예제의 <code>h = x @ w</code>는 두 연산에서 사용됩니다. 이 컴파일러는 소비자가 하나인 생산자만 합치므로, matmul이 독립된 노드로 남습니다." : "이 패스는 소비자가 하나인 연산을 소비자 쪽으로 합칩니다. 출력으로 반환하는 중간값과 여러 소비자가 공유하는 중간값은 보존합니다. 노드 수 감소가 곧 같은 비율의 속도 향상을 뜻하지는 않습니다.";
  return `<div class="fusion-summary"><b>${d.raw.operation_count}</b><span>→</span><b>${d.optimized.operation_count}</b><span>바깥쪽 연산 노드 · 입력과 출력 제외</span></div><div class="comparison-grid">${graphPanel("raw", "01 · 원래 그래프")}${graphPanel("optimized", "02 · 최적화된 그래프")}</div><div id="inspector" class="comparison-inspector">${inspector()}</div>
    <details class="ir-details"><summary>최적화된 IR · fusion 내부와 tile 속성</summary><pre tabindex="0">${escape(d.optimized.ir)}</pre></details>
    ${callout("무엇을 합칠 수 있을까요?", note)}<div class="exercise"><span class="exercise-label">TRY IT</span><span>Fusion을 ‘원소별 연산만’으로 바꾸어 실행하거나 ‘공유되는 중간값’ 예제를 선택해 비교하세요.</span></div>`;
}

function codeStage() {
  const python = state.codeTab === "python";
  return `<div class="tabs" role="tablist" aria-label="생성 코드 언어"><button role="tab" id="tab-python" class="tab" data-tab="python" aria-selected="${python}" aria-controls="code-panel" tabindex="${python ? 0 : -1}">NumPy / Python</button><button role="tab" id="tab-c" class="tab" data-tab="c" aria-selected="${!python}" aria-controls="code-panel" tabindex="${python ? -1 : 0}">Native / C</button></div>
    <div id="code-panel" role="tabpanel" aria-labelledby="tab-${python ? "python" : "c"}" class="code-stage">${codePanel(python ? state.data.python_code : state.data.c_code, python ? "generated_forward.py · RAW GRAPH" : "toy_run.c · OPTIMIZED GRAPH", python ? "python_code" : "c_code")}</div>
    ${callout(python ? "이 소스가 GraphModule의 실행 함수입니다" : "최적화가 루프 구조에 드러납니다", python ? "함수를 다시 추적하지 않습니다. 입력을 float32로 변환하고 shape을 확인한 뒤, 생성된 NumPy 연산을 실행합니다. 이 코드는 퓨전 전 그래프를 표현합니다." : "C 코드는 최적화된 그래프를 사용합니다. Fusion 내부 값은 스칼라로 계산하고, tiling을 켜면 matmul에 16 × 16 × 16 블록 루프가 추가됩니다. GCC가 이 소스를 로컬 CPU에서 실행할 라이브러리로 컴파일합니다.")}`;
}

function resultsStage() {
  const d = state.data, values = Object.values(d.checks);
  const failed = values.some((c) => c.status === "fail"), unavailable = values.some((c) => c.status === "unavailable"), warning = failed || unavailable;
  const title = failed ? "실행 결과에 차이가 있습니다." : unavailable ? "Python 검증 완료 · C 실행은 사용할 수 없습니다." : "모든 실행 경로의 결과가 일치합니다.";
  const labels = {python: "Generated Python", interpreter: "Raw IR interpreter", optimized_interpreter: "Optimized IR interpreter", c: "Compiled C"};
  return `<div class="results-header ${warning ? "warning" : ""}"><span class="result-icon" aria-hidden="true">${warning ? "!" : "✓"}</span><div><h2>${title}</h2><p>실제 로컬 CPU 실행 결과를 eager NumPy와 비교했습니다.</p></div></div><div class="results-grid"><section class="panel"><div class="panel-head"><h2>정확성 검증</h2><span class="pill mint">CPU · f32</span></div><table class="checks-table"><thead><tr><th>실행 경로</th><th>상태</th><th>최대 절대 오차</th></tr></thead><tbody>${Object.entries(d.checks).map(([key, check]) => `<tr><td>${labels[key]}</td><td><span class="check-status ${check.status}">${check.status.toUpperCase()}</span></td><td class="error-value">${check.max_abs_error === undefined ? "—" : check.max_abs_error.toExponential(2)}</td></tr>`).join("")}</tbody></table>${d.checks.c.message ? `<p class="unavailable-message">${escape(d.checks.c.message)}</p>` : ""}<div class="tolerance">np.allclose · rtol=${d.tolerance.rtol} · atol=${d.tolerance.atol}</div></section>
    <section class="panel"><div class="panel-head"><h2>출력 Y 미리보기</h2><span class="pill">${shape(d.output.shape)}</span></div><div class="matrix-wrapper"><table class="matrix-output" aria-label="eager NumPy 출력의 왼쪽 위 원소"><tbody>${d.output.preview.map((row) => `<tr>${row.map((value) => `<td>${value.toFixed(3)}</td>`).join("")}</tr>`).join("")}</tbody></table></div><p class="matrix-note">Eager NumPy 기준 · 왼쪽 위 최대 4 × 4 원소 · 소수 셋째 자리까지 표시</p></section></div>
    <details class="branch-example"><summary>더 알아보기 · 텐서 값에 따라 if로 분기하면 어떻게 될까요?</summary><div class="branch-body"><p>추적 중인 Tensor에는 실제 값이 없습니다. 따라서 <code>if x</code>의 참·거짓을 판단할 수 없어 이 toy tracer는 TraceError를 발생시킵니다.</p><pre tabindex="0">${escape(d.branch.source)}</pre><div class="branch-error">TraceError: ${escape(d.branch.error)}</div><div class="branch-caption">백엔드에서 실제 추적해 얻은 예상된 오류입니다. 이 예제는 부분 그래프 실행이나 eager fallback을 구현하지 않습니다.</div></div></details>
    ${callout("다음 실험: 결과를 유지하면서 표현 바꾸기", "Fusion과 tiling 설정을 바꾸고 다시 실행해 보세요. 생성되는 그래프와 루프는 달라지지만, 허용 오차 안에서 같은 값을 계산해야 합니다. 이 페이지의 작은 예제는 정확성을 확인하기 위한 것이며 성능을 비교하는 벤치마크가 아닙니다.")}`;
}

function updateSelection() {
  document.querySelectorAll("[data-node]").forEach((el) => {
    const selected = el.dataset.graph === state.selected?.graph && el.dataset.node === state.selected?.id;
    el.classList.toggle("selected", selected);
    el.setAttribute("aria-pressed", String(selected));
  });
  document.querySelectorAll("[data-graph-panel]").forEach((panel) => {
    panel.querySelectorAll(".graph-edge").forEach((el) => el.classList.toggle("highlight", panel.dataset.graphPanel === state.selected?.graph && (el.dataset.from === state.selected.id || el.dataset.to === state.selected.id)));
  });
}

function render() {
  const [eyebrow, title, description] = stages[state.step];
  $("#eyebrow").textContent = `${pad(state.step + 1)} / ${eyebrow}`;
  $("#lesson-title").textContent = title;
  $("#lesson-description").textContent = description;
  $("#chapter-number").textContent = pad(state.step + 1);
  $("#footer-step").textContent = `STEP ${pad(state.step + 1)} OF 05`;
  document.querySelectorAll(".stage-nav[data-step]").forEach((button) => {
    const active = Number(button.dataset.step) === state.step;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
  });
  $("#previous").disabled = state.step === 0;
  $("#next").innerHTML = `${["그래프 추출 살펴보기", "연산 합치기 살펴보기", "코드 생성 살펴보기", "실행 결과 확인하기", "처음부터 다시 살펴보기"][state.step]} <span>→</span>`;
  if (!state.data) return;
  $("#stage-content").innerHTML = [sourceStage, rawStage, fusionStage, codeStage, resultsStage][state.step]();
  updateSelection();
}

function navigate(step) {
  state.step = step;
  state.selected = null;
  render();
}

function readConfig() {
  return {backend: "c", example: $("#example").value, m: Number($("#m").value), k: Number($("#k").value), n: Number($("#n").value), fusion: $("#fusion").value, tiled: $("#tiled").checked};
}

function settingsChanged() {
  let valid = true;
  for (const key of ["m", "k", "n"]) {
    const input = $(`#${key}`), value = Number(input.value);
    const ok = input.value.trim() !== "" && Number.isInteger(value) && value >= 1 && value <= 128;
    input.setAttribute("aria-invalid", String(!ok));
    valid &&= ok;
  }
  const stale = state.data && Object.entries(readConfig()).some(([key, value]) => state.data.config[key] !== value);
  $(".status-line").classList.toggle("stale", Boolean(stale || !valid));
  $("#stage-content").classList.toggle("stale-content", Boolean(stale || !valid));
  $("#run-button").disabled = !valid || state.busy || !state.examples.length;
  if (state.busy) $("#run-status").textContent = "추적 → 최적화 → 컴파일 → 검증 중…";
  else if (!valid) $("#run-status").textContent = "M, K, N에는 1–128 사이 정수를 입력하세요. 아래는 이전 실행 결과입니다.";
  else if (stale) $("#run-status").textContent = "설정이 변경되었습니다. 실행하기를 눌러 갱신하세요. 아래는 이전 실행 결과입니다.";
  else if (state.data) $("#run-status").textContent = `✓ 실행 완료 · ${state.data.config.m} × ${state.data.config.k} × ${state.data.config.n} · 실제 CPU 결과`;
  return valid;
}

async function runExperiment(event) {
  event?.preventDefault();
  if (!settingsChanged() || state.busy) return;
  state.busy = true;
  const config = readConfig();
  $("#error-message").hidden = true;
  $("#stage-content").setAttribute("aria-busy", "true");
  $("#run-button").innerHTML = '<span class="spinner"></span> 실행 중';
  settingsChanged();
  try {
    const response = await fetch("/api/run", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(config)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error ?? "실행에 실패했습니다.");
    state.data = data;
    state.selected = null;
    render();
  } catch (error) {
    $("#error-message").textContent = `${error.message}\n서버 연결과 설정을 확인한 뒤 다시 실행해 주세요.`;
    $("#error-message").hidden = false;
    if (!state.data) $("#stage-content").innerHTML = '<div class="empty-state">실험을 불러오지 못했습니다. 실행하기로 다시 시도할 수 있습니다.</div>';
  } finally {
    state.busy = false;
    $("#stage-content").setAttribute("aria-busy", "false");
    $("#run-button").innerHTML = '<span>↻</span> 실행하기';
    settingsChanged();
    if (!$("#error-message").hidden) $("#run-status").textContent = state.data ? "실행 실패 · 아래는 마지막으로 완료된 실행 결과입니다." : "실행 실패 · 다시 시도할 수 있습니다.";
  }
}

document.querySelectorAll(".stage-nav[data-step]").forEach((button) => button.addEventListener("click", () => navigate(Number(button.dataset.step))));
$("#previous").addEventListener("click", () => navigate(Math.max(0, state.step - 1)));
$("#next").addEventListener("click", () => navigate((state.step + 1) % 5));
$("#config-form").addEventListener("input", settingsChanged);
$("#config-form").addEventListener("submit", runExperiment);

$("#stage-content").addEventListener("click", async (event) => {
  const node = event.target.closest("[data-node]");
  if (node) {
    state.selected = {graph: node.dataset.graph, id: node.dataset.node};
    $("#inspector").innerHTML = inspector();
    updateSelection();
  }
  const tab = event.target.closest("[data-tab]");
  if (tab) {
    state.codeTab = tab.dataset.tab;
    render();
    $(`[data-tab="${state.codeTab}"]`).focus();
  }
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    try {
      await navigator.clipboard.writeText(state.data[copy.dataset.copy]);
      copy.textContent = "복사됨 ✓";
    } catch {
      copy.textContent = "소스를 선택해 복사하세요";
    }
  }
});

$("#stage-content").addEventListener("keydown", (event) => {
  const node = event.target.closest("[data-node]");
  if (node && ["Enter", " "].includes(event.key)) {
    event.preventDefault();
    node.dispatchEvent(new MouseEvent("click", {bubbles: true}));
  }
  const tab = event.target.closest("[data-tab]");
  if (tab && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    event.preventDefault();
    state.codeTab = event.key === "Home" ? "python" : event.key === "End" ? "c" : state.codeTab === "python" ? "c" : "python";
    render();
    $(`[data-tab="${state.codeTab}"]`).focus();
  }
});

let compilerInitialized = false;
async function initialize() {
  if (compilerInitialized) return;
  compilerInitialized = true;
  try {
    const response = await fetch("/api/examples");
    if (!response.ok) throw new Error("예제를 불러오지 못했습니다.");
    const data = await response.json();
    state.examples = data.examples;
    $("#example").innerHTML = data.examples.map((e) => `<option value="${escape(e.id)}">${escape(e.title)}</option>`).join("");
    $("#example").value = data.defaults.example;
    $("#example").disabled = false;
    for (const key of ["m", "k", "n", "fusion"]) $(`#${key}`).value = data.defaults[key];
    $("#tiled").checked = data.defaults.tiled;
    await runExperiment();
  } catch (error) {
    $("#error-message").textContent = `${error.message} 페이지를 새로고침해 다시 연결하세요.`;
    $("#error-message").hidden = false;
    $("#run-status").textContent = "로컬 서버에 연결할 수 없습니다.";
    $("#stage-content").setAttribute("aria-busy", "false");
    $("#stage-content").innerHTML = '<div class="empty-state">서버가 실행 중인지 확인해 주세요.</div>';
  }
}

mountArchitecture($("#architecture-view"));

function showView() {
  const architecture = location.hash === "#architecture";
  $("#compiler-view").hidden = architecture;
  $("#architecture-view").hidden = !architecture;
  $(".compiler-sidebar").hidden = architecture;
  $(".architecture-sidebar").hidden = !architecture;
  $("#environment-label").textContent = architecture ? "ARCHITECTURE ATLAS" : "LOCAL CPU / FLOAT32";
  document.title = architecture ? "GPU / TPU 구조 — Tensor to Silicon" : "Tensor to Silicon — Compiler Playground";
  document.querySelectorAll("[data-view]").forEach((link) => {
    if ((link.dataset.view === "architecture") === architecture) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (!architecture) initialize();
}

window.addEventListener("hashchange", showView);
$(".skip-link").addEventListener("click", (event) => {
  event.preventDefault();
  $("#lesson").focus();
});
showView();
