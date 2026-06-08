"use strict";

const DATA_ROOT = "data";
const FIGURES_ROOT = "../paper/figures";
let state = {
  manifest: [],
  runId: null,
  meta: null,
  samples: [],
  probeResults: {},
  stats: null,
  activeProbe: null,
  selectedSampleIdx: null,
  selectedGenIdx: 0,
  filterLabel: "all",
  metricMode: "auc",
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
    // stats.json is optional (agentic runs only)
    state.stats = null;
    try {
      state.stats = await fetchJSON(`${DATA_ROOT}/${runId}/stats.json`);
    } catch { /* non-agentic run */ }
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
  renderStats();
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

function setMetric(mode) {
  state.metricMode = mode;
  document.querySelectorAll(".metric-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.metric === mode);
  });
  renderHeatmap();
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
  const evalBinAxis = state.meta.eval_bin_axis ?? "position";
  const nBins = 10;
  const majorityPerBin = state.meta.majority_baseline_per_bin?.[state.activeProbe] ?? null;
  const mode = state.metricMode;
  const metricField = mode === "auc" ? "test_auc" : mode === "ece" ? "test_ece" : mode === "brier" ? "test_brier" : "test_acc";
  const valField   = mode === "auc" ? "val_auc"  : mode === "ece" ? "val_ece"  : mode === "brier" ? "val_brier"  : "val_acc";
  // ECE and Brier: lower is better; invert colour scale (red=high=bad, green=low=good)
  const lowerIsBetter = mode === "ece" || mode === "brier";
  const fixedBaseline = mode === "auc" ? 0.5 : mode === "brier" ? 0.25 : mode === "ece" ? null : null;
  const metricLabel = mode === "auc" ? "AUROC" : mode === "ece" ? "ECE" : mode === "brier" ? "Brier" : "Accuracy";

  if (evalBinAxis === "step_absolute") {
    renderStepAbsoluteChart(heatContainer, lineContainer, probeData, layers, majorityPerBin);
    return;
  }

  const margin = { top: 20, right: 60, bottom: 50, left: 70 };
  const width = 420 - margin.left - margin.right;
  const cellH = 28;
  const height = layers.length * cellH;

  const svg = d3.select(heatContainer)
    .append("svg")
    .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // For lower-is-better metrics invert the colour ramp so green=low, red=high
  const diverging = lowerIsBetter
    ? t => d3.interpolateRgbBasis(["#1a9850", "#ffffff", "#d73027"])(t)
    : t => d3.interpolateRgbBasis(["#d73027", "#ffffff", "#1a9850"])(t);
  const makeColorScale = (baseline) =>
    d3.scaleDiverging(diverging).domain([0, baseline, lowerIsBetter ? baseline * 2 : 1.0]);

  const getBaseline = (bin) => {
    if (fixedBaseline !== null) return fixedBaseline;
    if (mode === "accuracy") return majorityPerBin?.[String(bin)] ?? 0.5;
    return 0.5;
  };
  const colorScaleForBin = d3.range(nBins).map(b => makeColorScale(getBaseline(b)));
  const getColor = (val, bin) => val !== null ? colorScaleForBin[bin](val) : "#eee";

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
      const val = cell?.[metricField] ?? null;
      svg.append("rect")
        .attr("class", "heatmap-cell")
        .attr("x", xScale(bin))
        .attr("y", yScale(layer))
        .attr("width", xScale.bandwidth())
        .attr("height", yScale.bandwidth())
        .attr("fill", getColor(val, bin))
        .on("mousemove", (event) => {
          tooltip.style.display = "block";
          tooltip.style.left = (event.pageX + 12) + "px";
          tooltip.style.top = (event.pageY - 20) + "px";
          if (val !== null) {
            const bl = getBaseline(bin);
            const diff = val - bl;
            const diffStr = (diff >= 0 ? "+" : "") + (diff * 100).toFixed(2) + "%";
            const baselineLabel = fixedBaseline !== null ? `random (${bl.toFixed(2)})` : `majority: ${(bl * 100).toFixed(1)}%`;
            tooltip.innerHTML = `Layer ${layer} | Bin ${bin}<br>${metricLabel}: ${(val * 100).toFixed(2)}%<br>${baselineLabel}<br>diff: ${diffStr}<br>val ${metricLabel}: ${(cell[valField] * 100).toFixed(2)}%<br>auc: ${(cell.test_auc * 100).toFixed(1)}%  acc: ${(cell.test_acc * 100).toFixed(1)}%<br>n_test: ${cell.n_test}`;
          } else {
            tooltip.innerHTML = `Layer ${layer} | Bin ${bin}<br>No data`;
          }
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
    .text(evalBinAxis === "step_relative" ? "Relative step bin" : "Relative position bin");

  // Colour bar
  const colorBarBaseline = fixedBaseline !== null ? fixedBaseline : (majorityPerBin
    ? (d3.range(nBins).reduce((s, b) => s + (majorityPerBin[String(b)] ?? 0.5), 0) / nBins)
    : 0.5);
  const colorBarMax = lowerIsBetter ? colorBarBaseline * 2 : 1.0;
  const baselineOffsetPct = ((1.0 - colorBarBaseline / colorBarMax) * 100).toFixed(1) + "%";
  const defs = svg.append("defs");
  const lgId = "hm-lg";
  const lg = defs.append("linearGradient").attr("id", lgId).attr("x1", "0%").attr("y1", "0%").attr("x2", "0%").attr("y2", "100%");
  lg.append("stop").attr("offset", "0%").attr("stop-color", lowerIsBetter ? "#1a9850" : "#1a9850");
  lg.append("stop").attr("offset", baselineOffsetPct).attr("stop-color", "#ffffff");
  lg.append("stop").attr("offset", "100%").attr("stop-color", lowerIsBetter ? "#d73027" : "#d73027");
  svg.append("rect")
    .attr("x", width + 8).attr("y", 0).attr("width", 12).attr("height", height)
    .attr("fill", `url(#${lgId})`);
  svg.append("text").attr("x", width + 22).attr("y", 6).style("font-size", "10px").text(lowerIsBetter ? "0" : "1.0");
  svg.append("text").attr("x", width + 22).attr("y", height * (colorBarBaseline / colorBarMax))
    .style("font-size", "10px").text(`${colorBarBaseline.toFixed(2)} ←`);
  svg.append("text").attr("x", width + 22).attr("y", height).style("font-size", "10px").text(lowerIsBetter ? colorBarMax.toFixed(2) : "0.0");

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
  const yDomainMin = mode === "auc" ? 0.45 : 0;
  const yDomainMax = lowerIsBetter ? colorBarBaseline * 2 : 1.0;
  const yL = d3.scaleLinear().domain([yDomainMin, yDomainMax]).range([lh, 0]);
  const colours = d3.schemeCategory10;

  layers.forEach((layer, li) => {
    const layerData = probeData[String(layer)] ?? {};
    const pts = [];
    for (let bin = 0; bin < nBins; bin++) {
      const cell = layerData[String(bin)];
      if (cell?.[metricField] !== undefined && cell[metricField] !== null) {
        pts.push([(bin + 0.5) / nBins, cell[metricField]]);
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

  // Baseline line(s)
  if (fixedBaseline !== null) {
    lsvg.append("line")
      .attr("x1", xL(0)).attr("x2", xL(1))
      .attr("y1", yL(fixedBaseline)).attr("y2", yL(fixedBaseline))
      .attr("stroke", "#999").attr("stroke-dasharray", "4,2").attr("stroke-width", 1);
  } else if (mode === "accuracy" && majorityPerBin) {
    for (let b = 0; b < nBins; b++) {
      const bl = majorityPerBin?.[String(b)] ?? 0.5;
      const x0 = xL(b / nBins);
      const x1 = xL((b + 1) / nBins);
      lsvg.append("line")
        .attr("x1", x0).attr("x2", x1).attr("y1", yL(bl)).attr("y2", yL(bl))
        .attr("stroke", "#999").attr("stroke-dasharray", "4,2").attr("stroke-width", 1);
    }
  }

  lsvg.append("g").attr("transform", `translate(0,${lh})`).call(d3.axisBottom(xL).ticks(5));
  lsvg.append("g").call(d3.axisLeft(yL).ticks(4));

  const baselineNote = fixedBaseline !== null ? ` (random = ${fixedBaseline})` : "";
  lsvg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -lh / 2).attr("y", -38)
    .attr("text-anchor", "middle").style("font-size", "9px")
    .text(`${metricLabel}${baselineNote}`);

  lsvg.append("text")
    .attr("x", lw / 2).attr("y", lh + 32)
    .attr("text-anchor", "middle").style("font-size", "10px")
    .text(evalBinAxis === "step_relative" ? "Relative step" : "Relative position");
}

function renderStepAbsoluteChart(heatContainer, lineContainer, probeData, layers, majorityPerBin) {
  // Collect all step keys across all layers
  const allSteps = new Set();
  layers.forEach(layer => {
    Object.keys(probeData[String(layer)] ?? {}).forEach(k => allSteps.add(Number(k)));
  });
  const steps = Array.from(allSteps).sort((a, b) => a - b);
  if (steps.length === 0) return;

  const colours = d3.schemeCategory10;
  const lm = { top: 10, right: 20, bottom: 40, left: 50 };
  const lw = 420 - lm.left - lm.right;
  const lh = 140 - lm.top - lm.bottom;

  // Summary line chart (always visible)
  const xL = d3.scaleLinear().domain([steps[0], steps[steps.length - 1]]).range([0, lw]);
  const yL = d3.scaleLinear().domain([0, 1]).range([lh, 0]);
  const lsvg = d3.select(lineContainer)
    .append("svg")
    .attr("viewBox", `0 0 ${lw + lm.left + lm.right} ${lh + lm.top + lm.bottom}`)
    .append("g").attr("transform", `translate(${lm.left},${lm.top})`);

  const _m = state.metricMode;
  const stepMetricField = _m === "auc" ? "test_auc" : _m === "ece" ? "test_ece" : _m === "brier" ? "test_brier" : "test_acc";
  layers.forEach((layer, li) => {
    const layerData = probeData[String(layer)] ?? {};
    const pts = steps.map(s => [s, layerData[String(s)]?.[stepMetricField]]).filter(d => d[1] !== undefined && d[1] !== null);
    if (pts.length === 0) return;
    lsvg.append("path").datum(pts)
      .attr("fill", "none").attr("stroke", colours[li % 10]).attr("stroke-width", 1.5)
      .attr("d", d3.line().x(d => xL(d[0])).y(d => yL(d[1])));
    lsvg.append("text")
      .attr("x", xL(pts[pts.length - 1][0]) + 4).attr("y", yL(pts[pts.length - 1][1]))
      .style("font-size", "9px").style("fill", colours[li % 10]).text(`L${layer}`);
  });

  // Per-step majority baseline
  steps.forEach(s => {
    const bl = majorityPerBin?.[String(s)];
    if (bl === undefined) return;
    lsvg.append("line")
      .attr("x1", xL(s) - 4).attr("x2", xL(s) + 4)
      .attr("y1", yL(bl)).attr("y2", yL(bl))
      .attr("stroke", "#999").attr("stroke-dasharray", "2,2").attr("stroke-width", 1);
  });

  lsvg.append("g").attr("transform", `translate(0,${lh})`).call(d3.axisBottom(xL).ticks(Math.min(steps.length, 10)).tickFormat(d3.format("d")));
  lsvg.append("g").call(d3.axisLeft(yL).ticks(4));
  lsvg.append("text").attr("x", lw / 2).attr("y", lh + 32)
    .attr("text-anchor", "middle").style("font-size", "10px").text("Step number");

  // Collapsible full table
  const details = document.createElement("details");
  details.style.marginTop = "8px";
  const summary = document.createElement("summary");
  summary.textContent = `Per-step breakdown (${steps.length} steps)`;
  summary.style.cssText = "cursor:pointer;font-size:11px;color:#555;user-select:none";
  details.appendChild(summary);

  const table = document.createElement("table");
  table.style.cssText = "font-size:10px;border-collapse:collapse;margin-top:6px;width:100%";
  const header = table.insertRow();
  ["Step", ...layers.map(l => `L${l} ${stepMetricField}`)].forEach(h => {
    const th = document.createElement("th");
    th.textContent = h;
    th.style.cssText = "padding:2px 6px;border-bottom:1px solid #ddd;text-align:right";
    header.appendChild(th);
  });
  steps.forEach(s => {
    const row = table.insertRow();
    [String(s), ...layers.map(layer => {
      const cell = probeData[String(layer)]?.[String(s)];
      return cell?.[stepMetricField] != null ? (cell[stepMetricField] * 100).toFixed(1) + "%" : "—";
    })].forEach((val, ci) => {
      const td = row.insertCell();
      td.textContent = val;
      td.style.cssText = "padding:2px 6px;border-bottom:1px solid #f0f0f0;text-align:right" + (ci === 0 ? ";font-weight:600" : "");
    });
  });
  details.appendChild(table);
  heatContainer.appendChild(details);
}

// Panel 1: run statistics (agentic + non-agentic)

function _renderHistogram(container, data, xLabel, color) {
  if (!data || data.length === 0) return;
  const m = { top: 10, right: 10, bottom: 36, left: 40 };
  const w = 380 - m.left - m.right;
  const h = 90 - m.top - m.bottom;

  const thresholds = d3.thresholdSturges(data);
  const bins = d3.bin().thresholds(thresholds)(data);

  const xS = d3.scaleLinear().domain([bins[0].x0, bins[bins.length - 1].x1]).range([0, w]);
  const yS = d3.scaleLinear().domain([0, d3.max(bins, d => d.length)]).range([h, 0]).nice();

  const svg = d3.select(container).append("svg")
    .attr("viewBox", `0 0 ${w + m.left + m.right} ${h + m.top + m.bottom}`)
    .append("g").attr("transform", `translate(${m.left},${m.top})`);

  svg.selectAll("rect").data(bins).join("rect")
    .attr("x", d => xS(d.x0) + 1)
    .attr("width", d => Math.max(0, xS(d.x1) - xS(d.x0) - 1))
    .attr("y", d => yS(d.length))
    .attr("height", d => h - yS(d.length))
    .attr("fill", color);

  svg.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xS).ticks(5)).selectAll("text").style("font-size", "9px");
  svg.append("g").call(d3.axisLeft(yS).ticks(3)).selectAll("text").style("font-size", "9px");
  svg.append("text").attr("x", w / 2).attr("y", h + 30).attr("text-anchor", "middle").style("font-size", "10px").text(xLabel);
}

function _renderTransitionBar(container, fToT, tToF, probeName) {
  if (fToT === 0 && tToF === 0) return;
  const m = { top: 6, right: 10, bottom: 36, left: 90 };
  const w = 380 - m.left - m.right;
  const h = 50 - m.top - m.bottom;

  const cats = ["F → T", "T → F"];
  const vals = [fToT, tToF];
  const colors = ["#4caf50", "#f44336"];

  const xS = d3.scaleLinear().domain([0, d3.max(vals)]).range([0, w]).nice();
  const yS = d3.scaleBand().domain(cats).range([0, h]).padding(0.2);

  const svg = d3.select(container).append("svg")
    .attr("viewBox", `0 0 ${w + m.left + m.right} ${h + m.top + m.bottom}`)
    .append("g").attr("transform", `translate(${m.left},${m.top})`);

  svg.selectAll("rect").data(vals).join("rect")
    .attr("y", (_, i) => yS(cats[i]))
    .attr("width", d => xS(d))
    .attr("height", yS.bandwidth())
    .attr("fill", (_, i) => colors[i]);

  svg.selectAll(".val-label").data(vals).join("text")
    .attr("class", "val-label")
    .attr("x", d => xS(d) + 4)
    .attr("y", (_, i) => yS(cats[i]) + yS.bandwidth() / 2 + 4)
    .style("font-size", "9px")
    .text(d => d);

  svg.append("g").call(d3.axisLeft(yS)).selectAll("text").style("font-size", "10px");
  svg.append("g").attr("transform", `translate(0,${h})`).call(d3.axisBottom(xS).ticks(4)).selectAll("text").style("font-size", "9px");
  svg.append("text").attr("x", w / 2).attr("y", h + 30).attr("text-anchor", "middle").style("font-size", "10px").text("Total transitions");
}

function renderStats() {
  const container = document.getElementById("stats-container");
  container.innerHTML = "";

  const s = state.stats;
  const isAgentic = s?.is_agentic ?? false;

  // Token size histogram (always shown, use stats if available, else n_captured_steps from samples)
  let tokenData = s?.n_tokens ?? null;
  if (!tokenData && state.samples.length > 0) {
    tokenData = state.samples.map(samp => samp.generations[0]?.n_captured_steps).filter(v => v != null);
  }

  if (tokenData && tokenData.length > 0) {
    const sep = document.createElement("div");
    sep.className = "stats-section";
    const title = document.createElement("div");
    title.className = "stats-title";
    title.textContent = "Run Statistics";
    sep.appendChild(title);

    const tokenLabel = document.createElement("div");
    tokenLabel.className = "stats-subtitle";
    tokenLabel.textContent = isAgentic ? "Token count per trajectory" : "Token count per generation";
    sep.appendChild(tokenLabel);
    _renderHistogram(sep, tokenData, "tokens", "#5566cc");

    if (isAgentic && s?.n_turns?.length > 0) {
      const turnLabel = document.createElement("div");
      turnLabel.className = "stats-subtitle";
      turnLabel.textContent = "Assistant turns per trajectory";
      sep.appendChild(turnLabel);
      _renderHistogram(sep, s.n_turns, "turns", "#888");
    }

    container.appendChild(sep);
  }

  // Per-probe transition stats (dynamic probes only)
  if (s?.probe_transitions) {
    for (const [probeName, pt] of Object.entries(s.probe_transitions)) {
      if (!pt.is_dynamic) continue;
      if (pt.total_false_to_true === 0 && pt.total_true_to_false === 0) continue;

      const sec = document.createElement("div");
      sec.className = "stats-section";

      const title = document.createElement("div");
      title.className = "stats-subtitle";
      title.textContent = `${probeName} — label changes`;
      sec.appendChild(title);

      _renderHistogram(sec, pt.changes_per_traj, "changes per trajectory", "#f0a500");

      const transTitle = document.createElement("div");
      transTitle.className = "stats-subtitle";
      transTitle.textContent = `${probeName} — transition direction`;
      sec.appendChild(transTitle);

      _renderTransitionBar(sec, pt.total_false_to_true, pt.total_true_to_false, probeName);

      container.appendChild(sec);
    }
  }
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

function _sampleCorrectFraction(sample) {
  // For agentic runs, use the trajectory outcome field directly
  if (state.meta?.is_agentic) {
    return sample.outcome === true ? 1.0 : 0.0;
  }
  const probeFilter = document.getElementById("filter-probe").value || state.meta?.probes?.[0];
  const gens = sample.generations;
  if (gens.length === 0) return 0;
  const correct = gens.filter(g => g.labels?.[probeFilter] === true).length;
  return correct / gens.length;
}

function renderSampleList() {
  const container = document.getElementById("sample-list");
  container.innerHTML = "";
  const labelFilter = state.filterLabel;

  for (let i = 0; i < state.samples.length; i++) {
    const sample = state.samples[i];
    const frac = _sampleCorrectFraction(sample);

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

    // For agentic: show outcome badge; for non-agentic: show probe fraction badges
    const badges = document.createElement("div");
    badges.className = "probe-badges";
    if (state.meta?.is_agentic) {
      const badge = document.createElement("span");
      badge.className = "probe-badge " + (sample.outcome ? "badge-resolved" : "badge-unresolved");
      badge.textContent = sample.outcome ? "resolved" : "unresolved";
      badges.appendChild(badge);
    } else {
      for (const p of (state.meta?.probes ?? [])) {
        const trueFrac = sample.generations.filter(g => g.labels?.[p] === true).length / Math.max(nGen, 1);
        const badge = document.createElement("span");
        badge.className = "probe-badge";
        badge.textContent = `${p.replace("_", " ")}: ${(trueFrac * 100).toFixed(0)}%`;
        badges.appendChild(badge);
      }
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

  // For agentic runs show instance metadata + message history
  if (state.meta?.is_agentic) {
    const meta = document.createElement("div");
    meta.className = "detail-meta";
    const resolved = sample.outcome;
    meta.innerHTML = `
      <span class="detail-meta-id">${sample.sample_id}</span>
      <span class="label-badge ${resolved ? "label-true" : "label-false"}">${resolved ? "resolved" : "unresolved"}</span>
    `;
    container.appendChild(meta);

    if (sample.messages?.length > 0) {
      const historyToggle = document.createElement("div");
      historyToggle.className = "prompt-toggle";
      historyToggle.textContent = `▶ Message history (${sample.messages.length} messages)`;
      let historyVisible = false;

      const historyEl = document.createElement("div");
      historyEl.className = "message-history";
      historyEl.style.display = "none";

      const gen0 = sample.generations?.[0];
      const perTurnLabels = gen0?.per_turn_labels ?? {};
      let assistantTurnIdx = 0;

      for (const msg of sample.messages) {
        const bubble = document.createElement("div");
        bubble.className = `message-bubble message-${msg.role}`;

        const roleRow = document.createElement("div");
        roleRow.className = "message-role-row";

        const roleEl = document.createElement("span");
        roleEl.className = "message-role";
        roleEl.textContent = msg.role;
        roleRow.appendChild(roleEl);

        // For assistant turns, show current probe label values
        if (msg.role === "assistant") {
          for (const [probeName, turnLabels] of Object.entries(perTurnLabels)) {
            const val = turnLabels[assistantTurnIdx] ?? null;
            const badge = document.createElement("span");
            badge.className = "message-probe-badge " +
              (val === true ? "label-true" : val === false ? "label-false" : "label-none");
            badge.textContent = probeName.replace(/_swe$/, "").replace(/_/g, " ") +
              ": " + (val === null ? "?" : val ? "✓" : "✗");
            roleRow.appendChild(badge);
          }
          assistantTurnIdx++;
        }

        bubble.appendChild(roleRow);

        const contentEl = document.createElement("pre");
        contentEl.className = "message-content";
        contentEl.textContent = msg.content ?? "";
        bubble.appendChild(contentEl);

        historyEl.appendChild(bubble);
      }

      historyToggle.addEventListener("click", () => {
        historyVisible = !historyVisible;
        historyEl.style.display = historyVisible ? "flex" : "none";
        historyToggle.textContent = `${historyVisible ? "▼" : "▶"} Message history (${sample.messages.length} messages)`;
      });

      container.appendChild(historyToggle);
      container.appendChild(historyEl);
    }
  } else {
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
    if (gen) {
      const textEl = document.createElement("pre");
      textEl.className = "generated-text";
      textEl.textContent = gen.generated_text;
      container.appendChild(textEl);
    }
  }

  const gen = sample.generations[state.selectedGenIdx];
  if (!gen) return;

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
      badge.textContent = "n/a";
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

// ── Tab switching ─────────────────────────────────────────────────────────────

function switchTab(name) {
  document.getElementById("panels").style.display = name === "probes" ? "" : "none";
  document.getElementById("tab-figures").style.display = name === "figures" ? "" : "none";
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  event.target.classList.add("active");
  if (name === "figures") renderGallery();
}

// ── Figures gallery ───────────────────────────────────────────────────────────

async function loadFiguresManifest() {
  try {
    return await fetchJSON(`${FIGURES_ROOT}/manifest.json`);
  } catch {
    return [];
  }
}

function renderGallery() {
  const container = document.getElementById("figures-gallery");
  const search = document.getElementById("figures-search").value.toLowerCase();
  container.innerHTML = "";

  loadFiguresManifest().then(figures => {
    const filtered = figures.filter(f =>
      !search || f.path.toLowerCase().includes(search) || (f.title || "").toLowerCase().includes(search)
    );

    if (filtered.length === 0) {
      container.innerHTML = '<p style="color:#888;font-size:13px;">No figures found. Run run_paper_figures.py to generate figures and manifest.</p>';
      return;
    }

    // Group by run_id
    const byRun = {};
    for (const fig of filtered) {
      const run = fig.run_id || "other";
      if (!byRun[run]) byRun[run] = [];
      byRun[run].push(fig);
    }

    for (const [runId, figs] of Object.entries(byRun)) {
      const section = document.createElement("div");
      section.style.cssText = "margin-bottom:24px;";

      const heading = document.createElement("h3");
      heading.textContent = runId;
      heading.style.cssText = "font-size:13px;color:#444;margin:0 0 10px;border-bottom:1px solid #e0e0e0;padding-bottom:4px;";
      section.appendChild(heading);

      const grid = document.createElement("div");
      grid.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;";

      for (const fig of figs) {
        const card = document.createElement("div");
        card.style.cssText = "cursor:pointer;border:1px solid #ddd;border-radius:6px;overflow:hidden;width:200px;background:#fafafa;";
        card.title = fig.title || fig.path;

        const cacheBust = `?v=${Date.now()}`;
        const figSrc = `${FIGURES_ROOT}/${fig.path}${cacheBust}`;
        const embed = document.createElement("embed");
        embed.src = figSrc;
        embed.type = "application/pdf";
        embed.style.cssText = "width:200px;height:130px;background:#fff;pointer-events:none;";

        const caption = document.createElement("div");
        const ext = fig.path.split(".").pop();
        caption.textContent = fig.title || fig.path.split("/").pop().replace(`.${ext}`, "");
        caption.style.cssText = "font-size:10px;color:#555;padding:4px 6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";

        card.appendChild(embed);
        card.appendChild(caption);
        card.addEventListener("click", () => openLightbox(figSrc, fig.title || fig.path));
        grid.appendChild(card);
      }

      section.appendChild(grid);
      container.appendChild(section);
    }
  });
}

function openLightbox(src, caption) {
  const lb = document.getElementById("lightbox");
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-caption").textContent = caption;
  lb.style.display = "flex";
}

function closeLightbox() {
  document.getElementById("lightbox").style.display = "none";
}

document.getElementById("figures-search").addEventListener("input", renderGallery);
document.getElementById("lightbox").addEventListener("click", e => {
  if (e.target === e.currentTarget) closeLightbox();
});

// ── Bootstrap ─────────────────────────────────────────────────────────────────
init().catch(err => {
  document.getElementById("panels").innerHTML = `<p style="padding:20px;color:red">Failed to load data: ${err.message}. Make sure to run run_export_dashboard.py first.</p>`;
});
