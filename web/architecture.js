// Structural teaching diagrams. Rectangles describe containment, not die area.
// Sources are also exposed in the page so model-specific facts stay reviewable.
const sources = {
  gpu: {title: "NVIDIA · Ampere / A100", url: "https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/"},
  tpu: {title: "Google · TPU v5e", url: "https://docs.cloud.google.com/tpu/docs/v5e"},
  mxu: {title: "Google · TPU architecture", url: "https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm"},
  memory: {title: "JAX · TPU memory spaces", url: "https://docs.jax.dev/en/latest/pallas/tpu/details.html"},
};
const parts = {
  gpu: {
    sm: {name: "SM", full: "Streaming Multiprocessor", kind: "group", path: ["GPU die", "SM"], role: "GPU 안에서 반복되는 처리 블록입니다. 스케줄러, 연산 유닛, 레지스터와 로컬 메모리가 함께 속합니다.", fact: "이 그림은 SM 일부만 반복해서 표시하며 GPC/TPC와 세부 배선은 생략합니다.", zoom: 1},
    hbm: {name: "HBM", full: "High Bandwidth Memory", kind: "memory", path: ["GPU 장치", "프로세서 die 외부", "HBM"], role: "GPU의 큰 데이터 저장 공간입니다. 프로세서 die 옆의 별도 메모리로, SM 내부 메모리와 구분됩니다.", fact: "같은 패키지의 메모리여도 GPU 프로세서 die 안에 있는 것은 아닙니다."},
    l2: {name: "L2 cache", full: "공유 캐시", kind: "memory", path: ["GPU die", "L2 cache"], role: "여러 SM이 공유하는 칩 내부 캐시입니다. SM마다 있는 로컬 메모리보다 바깥 계층에 놓입니다.", fact: "하나의 상자로 묶어 표시했으며, 실제 캐시 파티션과 연결망은 생략합니다."},
    scheduler: {name: "Warp scheduler", full: "워프 스케줄러", kind: "control", path: ["GPU die", "SM", "Warp scheduler"], role: "SM 내부에서 warp의 명령 발행을 담당하는 제어 부품입니다.", fact: "여러 스케줄러를 역할별 한 영역으로 표시했습니다. 실행 시간표를 나타내는 그림은 아닙니다."},
    cuda: {name: "CUDA cores", full: "범용 산술 연산 유닛", kind: "compute", path: ["GPU die", "SM", "CUDA cores"], role: "SM에 속한 산술 연산 유닛들입니다. 같은 SM 안에 행렬 연산 전용 Tensor Core도 있습니다.", fact: "CUDA core 하나와 SM 하나는 서로 다른 크기의 구성 단위입니다."},
    tensor: {name: "Tensor Core", full: "행렬 연산 유닛", kind: "compute", path: ["GPU die", "SM", "Tensor Core"], role: "NVIDIA GPU의 SM 안에 있는 행렬 연산 전용 유닛입니다.", fact: "A100은 SM당 Tensor Core 4개를 포함합니다. TPU의 TensorCore 전체와 같은 계층은 아닙니다.", zoom: 2},
    registers: {name: "Registers", full: "레지스터 파일", kind: "memory", path: ["GPU die", "SM", "Registers"], role: "SM 내부에서 실행에 필요한 값을 보관하는 작은 저장 공간입니다.", fact: "레지스터 파일을 한 영역으로 묶었습니다. 실제 SM의 subpartition 경계는 생략합니다."},
    shared: {name: "Shared memory / L1", full: "SM 로컬 메모리", kind: "memory", path: ["GPU die", "SM", "Shared memory / L1"], role: "SM에 붙은 온칩 저장 공간입니다. Shared memory는 명시적으로 사용하는 영역이고 L1은 캐시 역할을 합니다.", fact: "A100에서는 통합된 물리 자원을 사용하지만 shared memory와 L1의 역할은 다릅니다."},
  },
  tpu: {
    core: {name: "TensorCore", full: "TPU의 처리 코어", kind: "group", path: ["TPU die", "TensorCore"], role: "MXU, vector unit, scalar unit 등을 포함하는 큰 처리 블록입니다.", fact: "TPU v5e는 칩당 TensorCore 1개, 그 안에 MXU 4개를 포함합니다.", zoom: 1},
    hbm: {name: "HBM", full: "High Bandwidth Memory", kind: "memory", path: ["TPU 장치", "프로세서 die 외부", "HBM"], role: "TPU의 큰 데이터 저장 공간입니다. TensorCore의 로컬 메모리와 별도의 계층입니다.", fact: "개념도에서는 프로세서 die 밖의 하나의 메모리 블록으로 묶어 표시합니다."},
    scalar: {name: "Scalar unit", full: "스칼라 유닛", kind: "control", path: ["TPU die", "TensorCore", "Scalar unit"], role: "제어 흐름과 주소 계산 등을 담당하는 유닛입니다.", fact: "TPU의 TensorCore에는 MXU 이외에도 제어를 위한 구성 요소가 있습니다."},
    vector: {name: "Vector unit", full: "벡터 유닛", kind: "compute", path: ["TPU die", "TensorCore", "Vector unit"], role: "활성화 함수 등 벡터 연산을 담당하는 유닛입니다. 행렬 연산 유닛인 MXU와 함께 코어에 속합니다.", fact: "TPU 전체를 MXU 격자 하나로만 그리면 이 구성 요소가 빠집니다.", source: "mxu"},
    vmem: {name: "VMEM", full: "Vector memory", kind: "memory", path: ["TPU die", "TensorCore", "VMEM"], role: "벡터 데이터를 보관하는 온칩 scratchpad입니다. 칩 외부 HBM과 구분되는 로컬 메모리입니다.", fact: "VMEM은 GPU의 L2 cache와 같은 종류의 저장 공간이 아닙니다.", source: "memory"},
    smem: {name: "SMEM", full: "Scalar memory", kind: "memory", path: ["TPU die", "TensorCore", "SMEM"], role: "스칼라 유닛에 연결된 로컬 메모리입니다.", fact: "TPU의 SMEM은 scalar memory를 뜻합니다. GPU shared memory의 약칭과 구분하세요.", source: "memory"},
    registers: {name: "Registers", full: "스칼라·벡터 레지스터", kind: "memory", path: ["TPU die", "TensorCore", "Registers"], role: "코어 내부의 스칼라 값과 벡터 값을 담는 저장 공간입니다.", fact: "이 그림에서는 여러 종류의 레지스터를 하나의 역할 영역으로 묶습니다.", source: "memory"},
    mxu: {name: "MXU", full: "Matrix Multiply Unit", kind: "compute", path: ["TPU die", "TensorCore", "MXU"], role: "곱셈-누산 셀들이 규칙적으로 연결된 systolic array를 포함하는 행렬 연산 유닛입니다.", fact: "TPU v5e의 MXU는 128 × 128 MAC 격자입니다. 확대 그림은 8 × 8로 축약해서 보여줍니다.", zoom: 2, source: "mxu"},
    mac: {name: "MAC cell", full: "Multiply-accumulate cell", kind: "compute", path: ["TPU die", "TensorCore", "MXU", "MAC cell"], role: "곱셈과 누산을 담당하는 격자의 한 셀입니다. 이웃 셀과 연결되어 큰 배열을 이룹니다.", fact: "작은 정사각형 하나가 MAC 셀을 뜻합니다. 선은 이웃 간 연결이며 데이터의 실행 순서를 표시하지 않습니다.", source: "mxu"},
  },
};

const escape = (value) => String(value).replace(/[&<>"']/g, (c) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
const kinds = {compute: "연산", memory: "메모리", control: "제어", group: "복합 블록"};
const levels = ["전체 구조", "주요 블록 내부", "행렬 연산 유닛"];

function boundary(x, y, width, height, title, caption = "") {
  return `<rect class="arch-boundary" x="${x}" y="${y}" width="${width}" height="${height}" rx="12"/><text class="arch-boundary-title" x="${x + 18}" y="${y + 27}">${title}</text>${caption ? `<text class="arch-svg-note" x="${x + width - 17}" y="${y + 27}" text-anchor="end">${caption}</text>` : ""}`;
}

function block(family, part, x, y, width, height, title, subtitle = "", extra = "") {
  const info = parts[family][part];
  const center = x + width / 2;
  return `<g class="arch-block kind-${info.kind}" data-family="${family}" data-part="${part}" role="button" tabindex="0" aria-pressed="false" aria-label="${family.toUpperCase()} ${escape(title)} 선택${info.zoom ? ' 및 내부 확대' : ''}"><rect x="${x}" y="${y}" width="${width}" height="${height}" rx="7"/><text class="arch-block-title" x="${center}" y="${y + (subtitle ? height / 2 - 2 : height / 2 + 4)}" text-anchor="middle">${escape(title)}</text>${subtitle ? `<text class="arch-block-subtitle" x="${center}" y="${y + height / 2 + 16}" text-anchor="middle">${escape(subtitle)}</text>` : ""}${extra}</g>`;
}

function overview(family) {
  let svg = boundary(108, 30, 396, 356, family === "gpu" ? "GPU die" : "TPU die", "PROCESSOR");
  svg += block(family, "hbm", 12, 157, 70, 146, "HBM", "메모리");
  if (family === "gpu") {
    svg += '<path class="arch-wire" d="M82 230 H96 V324 H128 M306 266 V296"/>';
    svg += '<rect class="arch-group-outline" x="120" y="78" width="372" height="188" rx="8"/>';
    for (let i = 0; i < 6; i++) svg += block("gpu", "sm", 130 + (i % 3) * 118, 91 + Math.floor(i / 3) * 76, 108, 63, "SM", "내부 확대 ↗");
    svg += '<text class="arch-svg-note" x="306" y="249" text-anchor="middle">반복되는 SM 중 일부만 표시</text>';
    svg += block("gpu", "l2", 128, 296, 356, 56, "L2 cache", "여러 SM이 공유하는 캐시");
  } else {
    svg += '<path class="arch-wire" d="M82 230 H128"/>';
    svg += `<g class="arch-block kind-group" data-family="tpu" data-part="core" role="button" tabindex="0" aria-pressed="false" aria-label="TPU TensorCore 선택 및 내부 확대"><rect x="128" y="84" width="356" height="268" rx="8"/><text class="arch-block-title" x="150" y="115">TensorCore</text><text class="arch-block-subtitle" x="464" y="115" text-anchor="end">1 core / chip ↗</text>`;
    for (let i = 0; i < 4; i++) svg += `<rect class="arch-mini-compute" x="${150 + i * 79}" y="144" width="72" height="75" rx="4"/><text class="arch-mini-label" x="${186 + i * 79}" y="186" text-anchor="middle">MXU</text>`;
    svg += '<rect class="arch-mini-compute" x="150" y="236" width="148" height="37" rx="4"/><text class="arch-mini-label" x="224" y="260" text-anchor="middle">Vector unit</text><rect class="arch-mini-control" x="314" y="236" width="148" height="37" rx="4"/><text class="arch-mini-label" x="388" y="260" text-anchor="middle">Scalar unit</text><rect class="arch-mini-memory" x="150" y="291" width="312" height="37" rx="4"/><text class="arch-mini-label" x="306" y="315" text-anchor="middle">Local memory · Registers</text></g>';
  }
  return svg;
}

function coreView(family) {
  let svg = boundary(20, 28, 480, 364, family === "gpu" ? "SM" : "TensorCore", "주요 구성 요소");
  if (family === "gpu") {
    svg += block("gpu", "scheduler", 44, 83, 432, 48, "Warp schedulers", "제어 영역");
    svg += block("gpu", "registers", 44, 147, 432, 40, "Registers");
    svg += block("gpu", "cuda", 44, 203, 202, 116, "CUDA cores", "산술 연산 유닛들");
    for (let i = 0; i < 4; i++) svg += block("gpu", "tensor", 262 + (i % 2) * 110, 203 + Math.floor(i / 2) * 64, 104, 52, "Tensor Core", `${i + 1} / 4 ↗`);
    svg += block("gpu", "shared", 44, 335, 432, 40, "Shared memory / L1 cache");
  } else {
    svg += block("tpu", "scalar", 44, 83, 202, 50, "Scalar unit", "제어·주소 계산");
    svg += block("tpu", "vector", 262, 83, 214, 50, "Vector unit", "벡터 연산");
    svg += block("tpu", "smem", 44, 149, 202, 42, "SMEM · Scalar memory");
    svg += block("tpu", "vmem", 262, 149, 214, 42, "VMEM · Vector memory");
    svg += block("tpu", "registers", 44, 207, 432, 38, "Scalar / Vector registers");
    for (let i = 0; i < 4; i++) svg += block("tpu", "mxu", 44 + (i % 2) * 218, 261 + Math.floor(i / 2) * 62, 214 - (i % 2 === 0 ? 12 : 0), 50, "MXU", `${i + 1} / 4 · 내부 확대 ↗`);
  }
  return svg;
}

function matrixView(family) {
  let svg = boundary(20, 28, 480, 364, family === "gpu" ? "SM" : "TensorCore", "상위 블록");
  if (family === "gpu") {
    svg += block("gpu", "tensor", 62, 103, 396, 156, "Tensor Core", "SM 내부의 행렬 연산 유닛");
    svg += '<text class="arch-svg-note" x="260" y="283" text-anchor="middle">세부 내부 회로는 생략</text>';
    svg += block("gpu", "cuda", 44, 316, 132, 48, "CUDA cores");
    svg += block("gpu", "registers", 190, 316, 132, 48, "Registers");
    svg += block("gpu", "shared", 336, 316, 140, 48, "Shared / L1");
  } else {
    svg += '<rect class="arch-mxu-boundary" x="86" y="76" width="348" height="295" rx="9"/><text class="arch-block-title" x="105" y="102">MXU</text><text class="arch-svg-note" x="415" y="102" text-anchor="end">1 of 4</text>';
    svg += '<g class="arch-block arch-mac-grid kind-compute" data-family="tpu" data-part="mac" role="button" tabindex="0" aria-pressed="false" aria-label="TPU MAC cell 격자 선택"><rect class="arch-grid-hit" x="147" y="111" width="230" height="230" fill="transparent"/>';
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const x = 151 + col * 28, y = 115 + row * 28;
        if (col < 7) svg += `<path class="arch-cell-wire" d="M${x + 22} ${y + 11} h6"/>`;
        if (row < 7) svg += `<path class="arch-cell-wire" d="M${x + 11} ${y + 22} v6"/>`;
        svg += `<rect class="arch-mac-cell ${row === 3 && col === 3 ? "example-cell" : ""}" x="${x}" y="${y}" width="22" height="22" rx="2"/>`;
      }
    }
    svg += '</g><text class="arch-svg-note" x="260" y="354" text-anchor="middle">128 × 128 MAC을 8 × 8로 축약 표시</text>';
  }
  return svg;
}

function devicePanel(family, depth) {
  const titles = {gpu: ["SM들이 모인 GPU", "SM 하나의 내부", "SM 안의 Tensor Core"], tpu: ["TensorCore를 품은 TPU", "TensorCore 하나의 내부", "MXU 안의 MAC 격자"]};
  const notes = {
    gpu: ["SM과 공유 L2는 die 안에, HBM은 die 밖에 있습니다.", "역할별 묶음입니다. SM의 subpartition 경계와 세부 배선은 생략합니다.", "상위 SM의 일부만 표시합니다. Tensor Core의 내부 회로는 그리지 않습니다."],
    tpu: ["v5e는 TensorCore 1개를 포함합니다. 코어를 눌러 내부를 보세요.", "v5e의 코어에는 MXU 4개와 vector/scalar unit이 있습니다.", "한 칸은 MAC 셀, 짧은 선은 이웃 셀 간 연결입니다. 실제 격자 크기와 다릅니다."],
  };
  return `<article class="arch-device"><header class="arch-device-header"><div><span class="arch-device-tag">${family === "gpu" ? "NVIDIA · A100" : "GOOGLE · TPU v5e"}</span><h2>${titles[family][depth]}</h2></div><span class="arch-family">${family.toUpperCase()}</span></header><div class="arch-drawing" tabindex="0" aria-label="${family.toUpperCase()} 구조도 스크롤 영역"><svg viewBox="0 0 520 418" role="group" aria-label="${family.toUpperCase()} ${levels[depth]} 구조도"><title>${family.toUpperCase()} ${titles[family][depth]}</title>${[overview, coreView, matrixView][depth](family)}</svg></div><p class="arch-device-note">${notes[family][depth]}</p></article>`;
}

export function mountArchitecture(container) {
  const state = {depth: 0, family: "gpu", part: "sm"};

  function detail() {
    const part = parts[state.family][state.part], source = sources[part.source ?? state.family];
    return `<div class="arch-detail-head"><span class="arch-role kind-${part.kind}">${kinds[part.kind]}</span><span class="arch-detail-family">${state.family === "gpu" ? "NVIDIA A100" : "GOOGLE TPU v5e"}</span></div><div class="arch-detail-grid"><div><h2 id="arch-part-title">${part.name}</h2><p class="arch-part-full">${part.full}</p><p class="arch-part-role">${part.role}</p></div><div><span class="section-kicker">이 부품의 위치</span><ol class="arch-containment" aria-label="구성 요소 포함 관계">${part.path.map((label) => `<li>${label}</li>`).join("")}</ol><p class="arch-part-fact">${part.fact}</p><a class="arch-source-link" href="${source.url}" target="_blank" rel="noopener noreferrer">공식 자료 · ${source.title} ↗</a></div></div>`;
  }

  function updateSelection() {
    container.querySelectorAll("[data-part]").forEach((el) => {
      const selected = el.dataset.family === state.family && el.dataset.part === state.part;
      el.classList.toggle("selected", selected);
      el.setAttribute("aria-pressed", String(selected));
    });
  }

  function render() {
    container.innerHTML = `<div class="lesson-heading arch-heading"><div><p class="eyebrow">HARDWARE ATLAS / ${String(state.depth + 1).padStart(2, "0")}</p><h1>GPU와 TPU, 안쪽을 들여다보면.</h1><p class="lesson-description">같은 색은 같은 역할을, 큰 테두리는 부품의 포함 관계를 나타냅니다.</p></div><span class="arch-outline-icon" aria-hidden="true">▦</span></div>
      <div class="arch-toolbar"><nav class="arch-depth-tabs" aria-label="구조 확대 수준 선택">${levels.map((label, i) => `<button id="arch-level-${i}" data-depth="${i}" ${state.depth === i ? 'aria-current="step"' : ""}><span>0${i + 1}</span>${label}</button>`).join("")}</nav><button class="arch-reset" data-depth="0">↺ 전체 보기</button></div>
      <div class="arch-legend"><span><i class="compute"></i>연산</span><span><i class="memory"></i>메모리</span><span><i class="control"></i>제어</span><span><i class="group"></i>복합 블록</span><span class="arch-legend-note">블록 선택 · ↗ 내부 확대</span></div>
      <div class="architecture-grid">${devicePanel("gpu", state.depth)}${devicePanel("tpu", state.depth)}</div>
      <p class="arch-scale-note">구조를 설명하는 개념도입니다. 상자의 위치·면적·선 길이는 실제 배치를 재현하지 않습니다.</p>
      <section id="arch-detail" class="arch-detail" aria-label="선택한 부품 설명" aria-live="polite">${detail()}</section>
      <div class="arch-name-note"><span aria-hidden="true">≠</span><div><h2>이름은 비슷해도, 구성 단위가 다릅니다.</h2><p><b>GPU Tensor Core</b>는 SM 안의 행렬 연산 유닛입니다. <b>TPU TensorCore</b>는 MXU·vector·scalar unit을 포함하는 코어입니다.</p><div class="arch-name-paths"><span>GPU die › SM › <strong>Tensor Core</strong></span><span>TPU die › <strong>TensorCore</strong> › MXU</span></div></div></div>
      <details class="arch-sources"><summary>그림의 기준과 공식 자료</summary><p>GPU는 A100, TPU는 v5e를 기준으로 필요한 블록만 표시합니다. 메모리는 기능상 속한 계층을 표현하며 실제 면적이나 모든 연결을 나타내지 않습니다. 다른 세대의 구성과 개수는 다를 수 있습니다.</p><ul>${Object.values(sources).map((s) => `<li><a href="${s.url}" target="_blank" rel="noopener noreferrer">${s.title} ↗</a></li>`).join("")}</ul></details>`;
    document.querySelectorAll(".architecture-depth-nav [data-depth]").forEach((button) => {
      const current = Number(button.dataset.depth) === state.depth;
      button.classList.toggle("active", current);
      if (current) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
    });
    updateSelection();
  }

  function setDepth(depth, preserveSelection = false) {
    state.depth = depth;
    if (!preserveSelection) {
      state.family = depth === 2 ? "tpu" : "gpu";
      state.part = depth === 2 ? "mxu" : "sm";
    }
    render();
    container.querySelector(`#arch-level-${depth}`).focus({preventScroll: true});
  }

  container.addEventListener("click", (event) => {
    const depth = event.target.closest("[data-depth]");
    if (depth) return setDepth(Number(depth.dataset.depth));
    const block = event.target.closest("[data-part]");
    if (!block) return;
    state.family = block.dataset.family;
    state.part = block.dataset.part;
    const nextDepth = parts[state.family][state.part].zoom;
    if (nextDepth !== undefined && nextDepth > state.depth) return setDepth(nextDepth, true);
    container.querySelector("#arch-detail").innerHTML = detail();
    updateSelection();
  });
  container.addEventListener("keydown", (event) => {
    if (event.target.closest("[data-part]") && ["Enter", " "].includes(event.key)) {
      event.preventDefault();
      event.target.closest("[data-part]").dispatchEvent(new MouseEvent("click", {bubbles: true}));
    }
  });
  document.querySelectorAll(".architecture-depth-nav [data-depth]").forEach((button) => button.addEventListener("click", () => setDepth(Number(button.dataset.depth))));
  render();
}
