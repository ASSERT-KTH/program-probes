"use strict";

const DATA_ROOT = "data";
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
  baselineMode: "global",
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

function setBaselineMode(mode) {
  state.baselineMode = mode;
  document.getElementById("btn-baseline-global").classList.toggle("active", mode === "global");
  document.getElementById("btn-baseline-per-bin").classList.toggle("active", mode === "per-bin");
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
  const nBins = 10;
  const majority = state.meta.majority_baseline?.[state.activeProbe] ?? 0.5;
  const majorityPerBin = state.meta.majority_baseline_per_bin?.[state.activeProbe] ?? null;
  const perBin = state.baselineMode === "per-bin" && majorityPerBin !== null;

  const margin = { top: 20, right: 60, bottom: 50, left: 70 };
  const width = 420 - margin.left - margin.right;
  const cellH = 28;
  const height = layers.length * cellH;

  const svg = d3.select(heatContainer)
    .append("svg")
    .attr("viewBox", `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  // Diverging scale: red (0) → white (baseline) → green (1.0), centered at baseline.
  const diverging = t => d3.interpolateRgbBasis(["#d73027", "#ffffff", "#1a9850"])(t);
  const makeColorScale = (baseline) =>
    d3.scaleDiverging(diverging).domain([0, baseline, 1.0]);

  const colorScaleForBin = perBin
    ? d3.range(nBins).map(b => makeColorScale(majorityPerBin[String(b)]))
    : null;
  const globalColorScale = makeColorScale(majority);
  const getColor = (acc, bin) => acc !== null
    ? (perBin ? colorScaleForBin[bin](acc) : globalColorScale(acc))
    : "#eee";
  const getBaseline = (bin) => perBin ? majorityPerBin[String(bin)] : majority;

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
        .attr("fill", getColor(acc, bin))
        .on("mousemove", (event) => {
          tooltip.style.display = "block";
          tooltip.style.left = (event.pageX + 12) + "px";
          tooltip.style.top = (event.pageY - 20) + "px";
          if (acc !== null) {
            const bl = getBaseline(bin);
            const diff = acc - bl;
            const diffStr = (diff >= 0 ? "+" : "") + (diff * 100).toFixed(1) + "%";
            tooltip.innerHTML = `Layer ${layer} | Bin ${bin}<br>test: ${(acc * 100).toFixed(1)}%<br>baseline: ${(bl * 100).toFixed(1)}%<br>diff: ${diffStr}<br>val: ${(cell.val_acc * 100).toFixed(1)}%<br>n_test: ${cell.n_test}`;
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
    .text("Relative position bin");

  // Colour bar: green at top (1.0), white at baseline, red at bottom (0).
  const colorBarBaseline = perBin
    ? (d3.range(nBins).reduce((s, b) => s + majorityPerBin[String(b)], 0) / nBins)
    : majority;
  const baselineOffsetPct = ((1.0 - colorBarBaseline) * 100).toFixed(1) + "%";
  const defs = svg.append("defs");
  const lgId = "hm-lg";
  const lg = defs.append("linearGradient").attr("id", lgId).attr("x1", "0%").attr("y1", "0%").attr("x2", "0%").attr("y2", "100%");
  lg.append("stop").attr("offset", "0%").attr("stop-color", "#1a9850");
  lg.append("stop").attr("offset", baselineOffsetPct).attr("stop-color", "#ffffff");
  lg.append("stop").attr("offset", "100%").attr("stop-color", "#d73027");
  svg.append("rect")
    .attr("x", width + 8).attr("y", 0).attr("width", 12).attr("height", height)
    .attr("fill", `url(#${lgId})`);
  svg.append("text").attr("x", width + 22).attr("y", 6).style("font-size", "10px").text("1.0");
  svg.append("text").attr("x", width + 22).attr("y", height * (1 - colorBarBaseline))
    .style("font-size", "10px").text(perBin ? `~${colorBarBaseline.toFixed(2)}` : majority.toFixed(2));
  svg.append("text").attr("x", width + 22).attr("y", height).style("font-size", "10px").text("0.0");

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

  if (perBin) {
    // Step function: one horizontal segment per bin at its own baseline
    for (let b = 0; b < nBins; b++) {
      const x0 = xL(b / nBins);
      const x1 = xL((b + 1) / nBins);
      const y = yL(majorityPerBin[String(b)]);
      lsvg.append("line")
        .attr("x1", x0).attr("x2", x1).attr("y1", y).attr("y2", y)
        .attr("stroke", "#999").attr("stroke-dasharray", "4,2").attr("stroke-width", 1);
    }
  } else {
    lsvg.append("line")
      .attr("x1", 0).attr("x2", lw).attr("y1", yL(majority)).attr("y2", yL(majority))
      .attr("stroke", "#999").attr("stroke-dasharray", "4,2").attr("stroke-width", 1);
  }

  lsvg.append("g").attr("transform", `translate(0,${lh})`).call(d3.axisBottom(xL).ticks(5));
  lsvg.append("g").call(d3.axisLeft(yL).ticks(4));

  lsvg.append("text")
    .attr("x", lw / 2).attr("y", lh + 32)
    .attr("text-anchor", "middle").style("font-size", "10px")
    .text("Relative position");
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

// ── Bootstrap ─────────────────────────────────────────────────────────────────
init().catch(err => {
  document.getElementById("panels").innerHTML = `<p style="padding:20px;color:red">Failed to load data: ${err.message}. Make sure to run run_export_dashboard.py first.</p>`;
});
