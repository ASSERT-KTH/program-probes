"use strict";

const DATA_ROOT = "data";
let state = {
  manifest: [],
  runId: null,
  meta: null,
  samples: [],
  probeResults: {},
  activeProbe: null,
  selectedSampleIdx: null,
  selectedGenIdx: 0,
  filterLabel: "all",
};

// ── Data loading ──────────────────────────────────────────────────────────────

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

function setLoading(on) {
  document.getElementById("loading-overlay").classList.toggle("visible", on);
}

async function loadRun(runId) {
  setLoading(true);
  await new Promise(r => requestAnimationFrame(r));
  state.runId = runId;
  try {
    const [meta, samples, probeResults] = await Promise.all([
      fetchJSON(`${DATA_ROOT}/${runId}/meta.json`),
      fetchJSON(`${DATA_ROOT}/${runId}/samples.json`),
      fetchJSON(`${DATA_ROOT}/${runId}/probe_results.json`),
    ]);
    state.meta = meta;
    state.samples = samples;
    state.probeResults = probeResults;
    state.activeProbe = meta.probes[0] ?? null;
    state.selectedSampleIdx = null;
    state.selectedGenIdx = 0;
    renderAll();
  } finally {
    setLoading(false);
  }
}

async function init() {
  try {
    state.manifest = await fetchJSON(`${DATA_ROOT}/manifest.json`);
  } catch {
    state.manifest = [];
  }

  const sel = document.getElementById("run-selector");
  sel.innerHTML = "";
  for (const entry of state.manifest) {
    const opt = document.createElement("option");
    opt.value = entry.run_id;
    opt.textContent = `${entry.run_id} (${entry.model}, ${entry.task})`;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => loadRun(sel.value));

  if (state.manifest.length > 0) {
    await loadRun(state.manifest[0].run_id);
  }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function renderAll() {
  renderProbeTabs();
  renderHeatmap();
  renderFilterControls();
  renderSampleList();
  renderDetail();
}

// Panel 1: probe tabs + heatmap

function renderProbeTabs() {
  const container = document.getElementById("probe-tabs");
  container.innerHTML = "";
  for (const probe of (state.meta?.probes ?? [])) {
    const btn = document.createElement("button");
    btn.className = "probe-tab" + (probe === state.activeProbe ? " active" : "");
    btn.textContent = probe;
    btn.addEventListener("click", () => {
      state.activeProbe = probe;
      renderProbeTabs();
      renderHeatmap();
    });
    container.appendChild(btn);
  }
}

function renderHeatmap() {
  const heatContainer = document.getElementById("heatmap-container");
  const lineContainer = document.getElementById("linechart-container");
  heatContainer.innerHTML = "";
  lineContainer.innerHTML = "";

  if (!state.activeProbe || !state.meta) return;
  const probeData = state.probeResults[state.activeProbe];
  if (!probeData) return;

  const layers = state.meta.probe_layers ?? Object.keys(probeData).map(Number).sort((a, b) => a - b);
  const nBins = 10;
  const majority = state.meta.majority_baseline?.[state.activeProbe] ?? 0.5;

  const margin = { top: 20, right: 60, bottom: 50, left: 70 };
  const width = 420 - margin.left - margin.right;
  const cellH = 28;
  const height = layers.length * cellH;

  const svg = d3.select(heatContainer)
    .append("svg")
    .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const colorScale = d3.scaleSequential()
    .domain([majority, 1.0])
    .interpolator(d3.interpolateBlues);

  const xScale = d3.scaleBand().domain(d3.range(nBins)).range([0, width]).padding(0.04);
  const yScale = d3.scaleBand().domain(layers).range([0, height]).padding(0.04);

  // Tooltip
  let tooltip = document.getElementById("pp-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "pp-tooltip";
    tooltip.className = "tooltip";
    document.body.appendChild(tooltip);
  }

  layers.forEach(layer => {
    const layerData = probeData[String(layer)] ?? {};
    for (let bin = 0; bin < nBins; bin++) {
      const cell = layerData[String(bin)];
      const acc = cell?.test_acc ?? null;
      svg.append("rect")
        .attr("class", "heatmap-cell")
        .attr("x", xScale(bin))
        .attr("y", yScale(layer))
        .attr("width", xScale.bandwidth())
        .attr("height", yScale.bandwidth())
        .attr("fill", acc !== null ? colorScale(acc) : "#eee")
        .on("mousemove", (event) => {
          tooltip.style.display = "block";
          tooltip.style.left = (event.pageX + 12) + "px";
          tooltip.style.top = (event.pageY - 20) + "px";
          tooltip.innerHTML = acc !== null
            ? `Layer ${layer} | Bin ${bin}<br>test: ${(acc * 100).toFixed(1)}%<br>val: ${(cell.val_acc * 100).toFixed(1)}%<br>n_test: ${cell.n_test}`
            : `Layer ${layer} | Bin ${bin}<br>No data`;
        })
        .on("mouseleave", () => { tooltip.style.display = "none"; });
    }
  });

  svg.append("g").attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(xScale).tickFormat(i => `${(i / nBins).toFixed(1)}`))
    .selectAll("text").style("font-size", "10px");

  svg.append("g")
    .call(d3.axisLeft(yScale).tickFormat(l => `L${l}`))
    .selectAll("text").style("font-size", "10px");

  svg.append("text")
    .attr("x", width / 2).attr("y", height + margin.bottom - 4)
    .attr("text-anchor", "middle").style("font-size", "11px")
    .text("Relative position bin");

  // Colour bar
  const defs = svg.append("defs");
  const lgId = "hm-lg";
  const lg = defs.append("linearGradient").attr("id", lgId).attr("x1", "0%").attr("y1", "0%").attr("x2", "0%").attr("y2", "100%");
  lg.append("stop").attr("offset", "0%").attr("stop-color", colorScale(1.0));
  lg.append("stop").attr("offset", "100%").attr("stop-color", colorScale(majority));
  svg.append("rect")
    .attr("x", width + 8).attr("y", 0).attr("width", 12).attr("height", height)
    .attr("fill", `url(#${lgId})`);
  svg.append("text").attr("x", width + 22).attr("y", 6).style("font-size", "10px").text("1.0");
  svg.append("text").attr("x", width + 22).attr("y", height).style("font-size", "10px").text(majority.toFixed(2));

  // Line chart
  const lm = { top: 10, right: 20, bottom: 40, left: 50 };
  const lw = 420 - lm.left - lm.right;
  const lh = 120 - lm.top - lm.bottom;
  const lsvg = d3.select(lineContainer)
    .append("svg")
    .attr("viewBox", `0 0 ${lw + lm.left + lm.right} ${lh + lm.top + lm.bottom}`)
    .append("g")
    .attr("transform", `translate(${lm.left},${lm.top})`);

  const xL = d3.scaleLinear().domain([0, 1]).range([0, lw]);
  const yL = d3.scaleLinear().domain([0, 1]).range([lh, 0]);
  const colours = d3.schemeCategory10;

  layers.forEach((layer, li) => {
    const layerData = probeData[String(layer)] ?? {};
    const pts = [];
    for (let bin = 0; bin < nBins; bin++) {
      const cell = layerData[String(bin)];
      if (cell?.test_acc !== undefined) {
        pts.push([(bin + 0.5) / nBins, cell.test_acc]);
      }
    }
    if (pts.length === 0) return;
    const line = d3.line().x(d => xL(d[0])).y(d => yL(d[1]));
    lsvg.append("path").datum(pts)
      .attr("fill", "none")
      .attr("stroke", colours[li % 10])
      .attr("stroke-width", 1.5)
      .attr("d", line);
    lsvg.append("text")
      .attr("x", xL(pts[pts.length - 1][0]) + 4)
      .attr("y", yL(pts[pts.length - 1][1]))
      .style("font-size", "9px")
      .style("fill", colours[li % 10])
      .text(`L${layer}`);
  });

  lsvg.append("line")
    .attr("x1", 0).attr("x2", lw).attr("y1", yL(majority)).attr("y2", yL(majority))
    .attr("stroke", "#999").attr("stroke-dasharray", "4,2").attr("stroke-width", 1);

  lsvg.append("g").attr("transform", `translate(0,${lh})`).call(d3.axisBottom(xL).ticks(5));
  lsvg.append("g").call(d3.axisLeft(yL).ticks(4));

  lsvg.append("text")
    .attr("x", lw / 2).attr("y", lh + 32)
    .attr("text-anchor", "middle").style("font-size", "10px")
    .text("Relative position");
}

// Panel 2: sample browser

function renderFilterControls() {
  const sel = document.getElementById("filter-probe");
  sel.innerHTML = "";
  for (const probe of (state.meta?.probes ?? [])) {
    const opt = document.createElement("option");
    opt.value = probe;
    opt.textContent = probe;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", renderSampleList);
  document.getElementById("filter-label").addEventListener("change", e => {
    state.filterLabel = e.target.value;
    renderSampleList();
  });
}

function _sampleCorrectFraction(sample, probeFilter) {
  const gens = sample.generations;
  if (gens.length === 0) return 0;
  const correct = gens.filter(g => {
    const lbl = g.labels?.[probeFilter];
    return lbl === true;
  }).length;
  return correct / gens.length;
}

function renderSampleList() {
  const container = document.getElementById("sample-list");
  container.innerHTML = "";
  const probe = document.getElementById("filter-probe").value || state.meta?.probes?.[0];
  const labelFilter = state.filterLabel;

  for (let i = 0; i < state.samples.length; i++) {
    const sample = state.samples[i];
    const frac = _sampleCorrectFraction(sample, probe);

    if (labelFilter === "correct" && frac < 1.0) continue;
    if (labelFilter === "incorrect" && frac > 0.0) continue;
    if (labelFilter === "mixed" && (frac === 0.0 || frac === 1.0)) continue;

    const row = document.createElement("div");
    row.className = "sample-row" + (i === state.selectedSampleIdx ? " selected" : "");

    const groupEl = document.createElement("span");
    groupEl.className = "sample-group";
    groupEl.textContent = sample.group_id;

    const bar = document.createElement("div");
    bar.className = "correctness-bar";
    const nGen = sample.generations.length;
    const nCorrect = Math.round(frac * nGen);
    if (nCorrect > 0) {
      const g = document.createElement("div");
      g.className = "bar-correct";
      g.style.width = `${(nCorrect / nGen) * 100}%`;
      bar.appendChild(g);
    }
    if (nCorrect < nGen) {
      const b = document.createElement("div");
      b.className = "bar-incorrect";
      b.style.width = `${((nGen - nCorrect) / nGen) * 100}%`;
      bar.appendChild(b);
    }

    const badges = document.createElement("div");
    badges.className = "probe-badges";
    for (const p of (state.meta?.probes ?? [])) {
      const trueFrac = sample.generations.filter(g => g.labels?.[p] === true).length / Math.max(nGen, 1);
      const badge = document.createElement("span");
      badge.className = "probe-badge";
      badge.textContent = `${p.replace("_", " ")}: ${(trueFrac * 100).toFixed(0)}%`;
      badges.appendChild(badge);
    }

    row.appendChild(groupEl);
    row.appendChild(bar);
    row.appendChild(badges);

    row.addEventListener("click", () => {
      state.selectedSampleIdx = i;
      state.selectedGenIdx = 0;
      renderSampleList();
      renderDetail();
    });

    container.appendChild(row);
  }
}

// Panel 3: detail view

function renderDetail() {
  const container = document.getElementById("detail-content");
  if (state.selectedSampleIdx === null) {
    container.innerHTML = '<p class="placeholder">Select a sample to view details.</p>';
    return;
  }

  const sample = state.samples[state.selectedSampleIdx];
  container.innerHTML = "";

  // Prompt
  const promptToggle = document.createElement("div");
  promptToggle.className = "prompt-toggle";
  promptToggle.textContent = "▶ Prompt";
  let promptVisible = false;
  const promptEl = document.createElement("pre");
  promptEl.className = "prompt-block";
  promptEl.style.display = "none";
  promptEl.textContent = sample.prompt;
  promptToggle.addEventListener("click", () => {
    promptVisible = !promptVisible;
    promptEl.style.display = promptVisible ? "block" : "none";
    promptToggle.textContent = (promptVisible ? "▼ " : "▶ ") + "Prompt";
  });
  container.appendChild(promptToggle);
  container.appendChild(promptEl);

  // Generation tabs
  const genTabs = document.createElement("div");
  genTabs.className = "gen-tabs";
  for (let gi = 0; gi < sample.generations.length; gi++) {
    const tab = document.createElement("button");
    tab.className = "gen-tab" + (gi === state.selectedGenIdx ? " active" : "");
    tab.textContent = `Gen ${gi}`;
    tab.addEventListener("click", () => {
      state.selectedGenIdx = gi;
      renderDetail();
    });
    genTabs.appendChild(tab);
  }
  container.appendChild(genTabs);

  const gen = sample.generations[state.selectedGenIdx];
  if (!gen) return;

  // Generated text
  const textEl = document.createElement("pre");
  textEl.className = "generated-text";
  textEl.textContent = gen.generated_text;
  container.appendChild(textEl);

  // Probe labels
  const probesEl = document.createElement("div");
  probesEl.className = "detail-probes";
  for (const probe of (state.meta?.probes ?? [])) {
    const lbl = gen.labels?.[probe];
    const row = document.createElement("div");
    row.className = "detail-probe-row";

    const nameEl = document.createElement("span");
    nameEl.className = "detail-probe-name";
    nameEl.textContent = probe;
    row.appendChild(nameEl);

    if (Array.isArray(lbl)) {
      // dynamic probe: timeline
      const bar = document.createElement("div");
      bar.className = "timeline-bar";
      for (const val of lbl) {
        const seg = document.createElement("div");
        seg.className = "timeline-segment " + (val === true ? "timeline-true" : val === false ? "timeline-false" : "timeline-none");
        seg.style.flex = "1";
        bar.appendChild(seg);
      }
      row.appendChild(bar);
    } else if (lbl === null || lbl === undefined) {
      const badge = document.createElement("span");
      badge.className = "label-badge label-none";
      badge.textContent = "n/a (agent-mode only)";
      row.appendChild(badge);
    } else {
      const badge = document.createElement("span");
      badge.className = "label-badge " + (lbl ? "label-true" : "label-false");
      badge.textContent = lbl ? "True" : "False";
      row.appendChild(badge);
    }

    probesEl.appendChild(row);
  }
  container.appendChild(probesEl);
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
init().catch(err => {
  document.getElementById("panels").innerHTML = `<p style="padding:20px;color:red">Failed to load data: ${err.message}. Make sure to run run_export_dashboard.py first.</p>`;
});
