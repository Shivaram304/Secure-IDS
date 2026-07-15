/**
 * IDS Dashboard v2 — JavaScript Controller
 * Includes: SHAP chart, Anomaly gauge, Feature pruning panel, WebSocket
 */

const API_BASE      = "http://127.0.0.1:5000";
const MAX_ALERTS    = 100;
const TRAFFIC_WIN   = 60;

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  connected:     false,
  totalPackets:  0,
  labelCounts:   { normal:0, dos:0, probe:0, r2l:0, u2r:0 },
  trafficSeries: [],
  alerts:        [],
  attackRate:    0,
  filter:        "all",
  anomalyScore:  0,
  zeroDayCount:  0,
  shapLoaded:    false,
  liteLoaded:    false,
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dom = {
  statusPill:    document.getElementById("status-pill"),
  statusText:    document.getElementById("status-text"),
  totalPackets:  document.getElementById("kpi-total"),
  totalAttacks:  document.getElementById("kpi-attacks"),
  attackRate:    document.getElementById("kpi-rate"),
  normalCount:   document.getElementById("kpi-normal"),
  alertFeed:     document.getElementById("alert-feed"),
  alertFilter:   document.getElementById("alert-filter"),
  btnStart:      document.getElementById("btn-start"),
  btnStop:       document.getElementById("btn-stop"),
  toastContainer:document.getElementById("toast-container"),
  threatLabel:   document.getElementById("threat-label"),
  threatPct:     document.getElementById("threat-pct"),
  // Novelty panels
  anomalyScoreVal: document.getElementById("anomaly-score-val"),
  zeroDayRisk:     document.getElementById("zero-day-risk"),
  zeroDayCount:    document.getElementById("zeroday-count"),
  aeAccuracy:      document.getElementById("ae-accuracy"),
  aeRecall:        document.getElementById("ae-recall"),
  fullAcc:         document.getElementById("full-acc"),
  liteAcc:         document.getElementById("lite-acc"),
  speedupVal:      document.getElementById("speedup-val"),
  accDrop:         document.getElementById("acc-drop"),
  liteFeatList:    document.getElementById("lite-features-list"),
};

// ── Charts ────────────────────────────────────────────────────────────────────
let trafficChart, donutChart, threatGaugeChart, shapChart, anomalyGaugeChart;

const CHART_COLORS = {
  normal:"#10b981", dos:"#ef4444", probe:"#f97316", r2l:"#f59e0b", u2r:"#8b5cf6"
};

function initTrafficChart() {
  const ctx = document.getElementById("traffic-chart").getContext("2d");
  const g1 = ctx.createLinearGradient(0,0,0,220);
  g1.addColorStop(0,"rgba(59,130,246,0.35)"); g1.addColorStop(1,"rgba(59,130,246,0)");
  const g2 = ctx.createLinearGradient(0,0,0,220);
  g2.addColorStop(0,"rgba(239,68,68,0.3)");  g2.addColorStop(1,"rgba(239,68,68,0)");
  trafficChart = new Chart(ctx, {
    type:"line",
    data:{ labels:[], datasets:[
      { label:"Total", data:[], borderColor:"#3b82f6", backgroundColor:g1,
        borderWidth:2, fill:true, tension:0.4, pointRadius:0 },
      { label:"Attacks", data:[], borderColor:"#ef4444", backgroundColor:g2,
        borderWidth:2, fill:true, tension:0.4, pointRadius:0 },
    ]},
    options:{ responsive:true, maintainAspectRatio:false, animation:{duration:300},
      interaction:{intersect:false,mode:"index"},
      plugins:{ legend:{labels:{color:"#94a3b8",font:{size:11},boxWidth:12}},
        tooltip:{backgroundColor:"rgba(12,17,32,0.9)",borderColor:"rgba(59,130,246,0.3)",borderWidth:1} },
      scales:{ x:{ticks:{color:"#475569",maxTicksLimit:8,font:{size:10}},grid:{color:"rgba(255,255,255,0.04)"}},
               y:{ticks:{color:"#475569",font:{size:10}},grid:{color:"rgba(255,255,255,0.04)"},beginAtZero:true} } }
  });
}

function initDonutChart() {
  const ctx = document.getElementById("donut-chart").getContext("2d");
  donutChart = new Chart(ctx, {
    type:"doughnut",
    data:{ labels:["Normal","DoS","Probe","R2L","U2R"],
      datasets:[{ data:[1,0,0,0,0], backgroundColor:Object.values(CHART_COLORS),
        borderColor:"rgba(5,8,15,0.8)", borderWidth:3, hoverOffset:6 }] },
    options:{ responsive:true, maintainAspectRatio:false, cutout:"72%",
      plugins:{ legend:{display:false},
        tooltip:{backgroundColor:"rgba(12,17,32,0.9)",borderColor:"rgba(59,130,246,0.3)",borderWidth:1} } }
  });
}

function initThreatGauge() {
  const ctx = document.getElementById("threat-gauge").getContext("2d");
  threatGaugeChart = new Chart(ctx, {
    type:"doughnut",
    data:{ datasets:[{ data:[0,100], backgroundColor:["#ef4444","rgba(255,255,255,0.04)"],
        borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false, circumference:180, rotation:-90,
      cutout:"75%", plugins:{legend:{display:false},tooltip:{enabled:false}}, animation:{duration:500} }
  });
}

function initSHAPChart() {
  const ctx = document.getElementById("shap-chart").getContext("2d");
  shapChart = new Chart(ctx, {
    type:"bar",
    data:{ labels:[], datasets:[{
      label:"Mean |SHAP| Value",
      data:[],
      backgroundColor: [
        "rgba(139,92,246,0.8)","rgba(99,102,241,0.8)","rgba(59,130,246,0.8)",
        "rgba(6,182,212,0.8)","rgba(16,185,129,0.8)","rgba(245,158,11,0.8)",
        "rgba(249,115,22,0.8)","rgba(239,68,68,0.8)","rgba(236,72,153,0.8)",
        "rgba(168,85,247,0.8)"
      ],
      borderColor:"rgba(255,255,255,0.1)", borderWidth:1,
      borderRadius:4,
    }]},
    options:{
      indexAxis:"y",
      responsive:true, maintainAspectRatio:false,
      animation:{duration:600},
      plugins:{
        legend:{display:false},
        tooltip:{
          backgroundColor:"rgba(12,17,32,0.9)",
          borderColor:"rgba(139,92,246,0.4)",
          borderWidth:1,
          callbacks:{ label:(c) => ` SHAP: ${c.parsed.x.toFixed(4)}` }
        }
      },
      scales:{
        x:{ ticks:{color:"#475569",font:{size:10}},
            grid:{color:"rgba(255,255,255,0.04)"}, beginAtZero:true },
        y:{ ticks:{color:"#94a3b8",font:{size:10,family:"JetBrains Mono"}},
            grid:{display:false} }
      }
    }
  });
}

function initAnomalyGauge() {
  const ctx = document.getElementById("anomaly-gauge").getContext("2d");
  anomalyGaugeChart = new Chart(ctx, {
    type:"doughnut",
    data:{ datasets:[{ data:[0,100],
        backgroundColor:["#10b981","rgba(255,255,255,0.04)"], borderWidth:0 }] },
    options:{ responsive:true, maintainAspectRatio:false, circumference:180, rotation:-90,
      cutout:"75%", plugins:{legend:{display:false},tooltip:{enabled:false}}, animation:{duration:400} }
  });
}

// ── Chart Updates ─────────────────────────────────────────────────────────────

function updateTrafficChart() {
  if (!trafficChart) return;
  const s = state.trafficSeries.slice(-TRAFFIC_WIN);
  trafficChart.data.labels = s.map(x => x.ts.split("T")[1] || x.ts);
  trafficChart.data.datasets[0].data = s.map(x => x.count);
  trafficChart.data.datasets[1].data = s.map(x => x.attacks);
  trafficChart.update("none");
}

function updateDonutChart() {
  if (!donutChart) return;
  const c = state.labelCounts;
  donutChart.data.datasets[0].data = [c.normal||0,c.dos||0,c.probe||0,c.r2l||0,c.u2r||0];
  donutChart.update("none");
  ["normal","dos","probe","r2l","u2r"].forEach(k => {
    const el = document.getElementById(`legend-${k}`);
    if (el) el.textContent = (c[k]||0).toLocaleString();
  });
}

function updateThreatGauge() {
  if (!threatGaugeChart) return;
  const rate  = state.attackRate;
  const color = rate<5 ? "#10b981" : rate<20 ? "#f59e0b" : rate<50 ? "#f97316" : "#ef4444";
  threatGaugeChart.data.datasets[0].data = [rate, Math.max(0,100-rate)];
  threatGaugeChart.data.datasets[0].backgroundColor[0] = color;
  threatGaugeChart.update("none");
  const lbl = rate<5 ? "LOW" : rate<20 ? "MEDIUM" : rate<50 ? "HIGH" : "CRITICAL";
  if (dom.threatLabel) { dom.threatLabel.textContent = lbl; dom.threatLabel.style.color = color; }
  if (dom.threatPct)   dom.threatPct.textContent = rate.toFixed(1)+"% attack rate";
}

function updateSHAPChart(features) {
  if (!shapChart || !features || !features.length) return;
  const top10 = features.slice(0, 10);
  shapChart.data.labels   = top10.map(f => f.feature);
  shapChart.data.datasets[0].data = top10.map(f => f.importance);
  shapChart.update();
  state.shapLoaded = true;
}

function updateAnomalyGauge(score) {
  if (!anomalyGaugeChart) return;
  const s     = Math.min(100, Math.max(0, score));
  const color = s<40 ? "#10b981" : s<70 ? "#f59e0b" : "#ef4444";
  anomalyGaugeChart.data.datasets[0].data = [s, 100-s];
  anomalyGaugeChart.data.datasets[0].backgroundColor[0] = color;
  anomalyGaugeChart.update("none");

  if (dom.anomalyScoreVal) {
    dom.anomalyScoreVal.textContent = Math.round(s);
    dom.anomalyScoreVal.style.color = color;
  }
  const risk = s<40 ? "LOW RISK" : s<70 ? "MEDIUM RISK" : "HIGH RISK - ZERO-DAY?";
  if (dom.zeroDayRisk) { dom.zeroDayRisk.textContent = risk; dom.zeroDayRisk.style.color = color; }
}

function updateProgressBars() {
  const c = state.labelCounts;
  const total = Math.max(Object.values(c).reduce((a,b)=>a+b,0),1);
  ["dos","probe","r2l","u2r"].forEach(cls => {
    const bar = document.getElementById(`bar-${cls}`);
    const pct = document.getElementById(`pct-${cls}`);
    if (bar && pct) {
      const p = ((c[cls]||0)/total*100).toFixed(1);
      bar.style.width = p+"%";
      pct.textContent = p+"%";
    }
  });
}

// ── Fetch novelty data ────────────────────────────────────────────────────────

async function fetchSHAPImportance() {
  try {
    const res  = await fetch(`${API_BASE}/api/feature-importance?top=10`);
    const data = await res.json();
    if (data.features) updateSHAPChart(data.features);
    if (data.lite_features && data.lite_features.length) renderLiteFeatures(data.lite_features);
  } catch (_) {}
}

async function fetchLiteModelStats() {
  try {
    const res  = await fetch(`${API_BASE}/api/lite-model`);
    const data = await res.json();
    if (data.comparison) {
      const c = data.comparison;
      if (dom.fullAcc)   dom.fullAcc.textContent   = (c.full.accuracy*100).toFixed(1)+"%";
      if (dom.liteAcc)   dom.liteAcc.textContent   = (c.lite.accuracy*100).toFixed(1)+"%";
      if (dom.speedupVal)dom.speedupVal.textContent = c.speedup.toFixed(1)+"x";
      if (dom.accDrop)   dom.accDrop.textContent    = c.acc_drop.toFixed(2)+"%";
    }
    if (data.lite_features) renderLiteFeatures(data.lite_features);
    state.liteLoaded = true;
  } catch (_) {}
}

function renderLiteFeatures(features) {
  if (!dom.liteFeatList || !features.length) return;
  const colors = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b",
                  "#f97316","#ef4444","#ec4899","#a855f7","#6366f1",
                  "#14b8a6","#84cc16","#f43f5e","#0ea5e9","#d946ef"];
  dom.liteFeatList.innerHTML = features.map((f, i) => `
    <span style="padding:3px 10px;border-radius:999px;font-size:0.72rem;
                 font-family:'JetBrains Mono',monospace;font-weight:500;
                 background:${colors[i%colors.length]}22;
                 border:1px solid ${colors[i%colors.length]}55;
                 color:${colors[i%colors.length]}">${f}</span>
  `).join("");
}

// ── Detection Handler ─────────────────────────────────────────────────────────

function handleDetection(det) {
  state.alerts.unshift(det);
  if (state.alerts.length > MAX_ALERTS) state.alerts.pop();
  state.labelCounts[det.label] = (state.labelCounts[det.label]||0) + 1;
  state.totalPackets++;

  // Anomaly score from detection
  if (det.anomaly_score !== undefined) {
    state.anomalyScore = det.anomaly_score;
    updateAnomalyGauge(det.anomaly_score);
    if (det.is_anomaly) {
      state.zeroDayCount++;
      if (dom.zeroDayCount) dom.zeroDayCount.textContent = state.zeroDayCount;
    }
  }

  if (state.filter==="all" || state.filter===det.label ||
      (state.filter==="threats" && det.label!=="normal")) {
    prependAlertRow(det);
  }
  if (det.label !== "normal") {
    animateKpi(dom.totalAttacks);
    // Trigger Threat Map beam animation
    if (det.src_geo) {
      triggerThreatMapBeam(det.src_geo, det.label);
    }
    // Update AI Narrative panel with the latest threat automatically
    selectAlertForNarrative(det);
    // Refresh Attacker Profiles
    fetchAttackerProfiles();
  }
  if (det.severity==="critical") {
    showToast(`ALERT: ${det.label.toUpperCase()} from ${det.src_ip}`, "alert");
  }
  updateKpiValues();
}

// ── Stats Update ──────────────────────────────────────────────────────────────

function updateStats(stats) {
  state.totalPackets  = stats.total_packets  || state.totalPackets;
  state.labelCounts   = stats.label_counts   || state.labelCounts;
  state.attackRate    = stats.attack_rate    || state.attackRate;
  state.trafficSeries = stats.traffic_series || state.trafficSeries;
  updateKpiValues();
  updateTrafficChart();
  updateDonutChart();
  updateThreatGauge();
  updateProgressBars();
  // Update anomaly from latest score
  if (stats.anomaly_scores && stats.anomaly_scores.length) {
    const last = stats.anomaly_scores[stats.anomaly_scores.length-1];
    updateAnomalyGauge(last.score || 0);
  }
}

// ── KPI ───────────────────────────────────────────────────────────────────────

function updateKpiValues() {
  const attacks = Object.entries(state.labelCounts)
    .filter(([k])=>k!=="normal").reduce((s,[,v])=>s+v,0);
  animateNumber(dom.totalPackets, state.totalPackets);
  animateNumber(dom.totalAttacks, attacks);
  if (dom.attackRate) dom.attackRate.textContent = state.attackRate.toFixed(1)+"%";
  animateNumber(dom.normalCount, state.labelCounts.normal||0);
}

function animateNumber(el, target) {
  if (!el) return;
  const current = parseInt(el.textContent.replace(/,/g,""))||0;
  if (current===target) return;
  const diff=target-current, step=Math.ceil(Math.abs(diff)/10);
  const newVal=current+(diff>0?step:-step);
  el.textContent=newVal.toLocaleString();
  if (Math.abs(target-newVal)>step) requestAnimationFrame(()=>animateNumber(el,target));
  else el.textContent=target.toLocaleString();
}

function animateKpi(el) {
  if (!el) return;
  el.style.color="var(--sev-critical)";
  el.style.transform="scale(1.08)";
  setTimeout(()=>{el.style.color="";el.style.transform="";},600);
}

// ── Alert Feed ────────────────────────────────────────────────────────────────

function prependAlertRow(det) {
  const feed = dom.alertFeed;
  while (feed.children.length>=80) feed.removeChild(feed.lastChild);
  const time = det.timestamp?.split("T")[1]||"--:--";
  const anomBadge = det.is_anomaly
    ? `<span style="font-size:0.65rem;color:var(--sev-critical);margin-left:4px">ANOMALY</span>` : "";
  const item = document.createElement("div");
  item.className = `alert-item sev-${det.severity}`;
  item.style.cursor = "pointer";
  item.onclick = () => selectAlertForNarrative(det);
  item.innerHTML = `
    <span class="alert-badge badge-${det.severity}">${det.label}</span>
    <div class="alert-meta">
      <span class="alert-ip">${det.src_ip} &rarr; ${det.dst_ip} ${anomBadge}</span>
      <span class="alert-label">${det.protocol} &middot; conf ${(det.confidence*100).toFixed(1)}%</span>
    </div>
    <span class="alert-time">${time}</span>`;
  feed.insertBefore(item, feed.firstChild);
}

function rebuildAlertFeed() {
  dom.alertFeed.innerHTML="";
  const filtered = state.filter==="all"    ? state.alerts
                 : state.filter==="threats"? state.alerts.filter(a=>a.label!=="normal")
                 : state.alerts.filter(a=>a.label===state.filter);
  filtered.slice(0,60).forEach(prependAlertRow);
}

// ── Socket.IO ─────────────────────────────────────────────────────────────────

let socket;
function connectSocket() {
  socket = io(API_BASE, {transports:["websocket","polling"]});
  socket.on("connect",       ()=>{ state.connected=true;  updateConnectionUI(true);  showToast("Connected to IDS server","info"); });
  socket.on("disconnect",    ()=>{ state.connected=false; updateConnectionUI(false); showToast("Disconnected","alert"); });
  socket.on("detection",     (det)=>handleDetection(det));
  socket.on("stats_update",  (s)=>updateStats(s));
  socket.on("connect_error", ()=>updateConnectionUI(false));
}

function updateConnectionUI(online) {
  if (!dom.statusPill||!dom.statusText) return;
  dom.statusPill.className=`status-pill ${online?"online":"offline"}`;
  dom.statusText.textContent = online?"LIVE":"OFFLINE";
}

// ── Capture controls ──────────────────────────────────────────────────────────

async function startCapture() {
  try {
    const r = await fetch(`${API_BASE}/api/capture/start`,{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({demo:true})});
    const d = await r.json();
    showToast(d.message||"Capture started","info");
    dom.btnStart?.setAttribute("disabled","");
    dom.btnStop?.removeAttribute("disabled");
  } catch(_){ showToast("Could not start capture","alert"); }
}

async function stopCapture() {
  try {
    const r = await fetch(`${API_BASE}/api/capture/stop`,{method:"POST"});
    const d = await r.json();
    showToast(d.message||"Capture stopped","info");
    dom.btnStop?.setAttribute("disabled","");
    dom.btnStart?.removeAttribute("disabled");
  } catch(_){ showToast("Could not stop capture","alert"); }
}

// ── Toast ─────────────────────────────────────────────────────────────────────

function showToast(msg, type="info") {
  const t = document.createElement("div");
  t.className=`toast toast-${type}`; t.textContent=msg;
  dom.toastContainer?.appendChild(t);
  setTimeout(()=>t.remove(),3200);
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function fetchInitialData() {
  try {
    const [sr, ar] = await Promise.all([
      fetch(`${API_BASE}/api/stats`),
      fetch(`${API_BASE}/api/alerts?limit=50`)
    ]);
    updateStats(await sr.json());
    const alertData = await ar.json();
    state.alerts = alertData.alerts||[];
    rebuildAlertFeed();
  } catch(_) {}
  // Load novelty data
  fetchSHAPImportance();
  fetchLiteModelStats();
  fetchAttackerProfiles();
  fetchAdversarialReport();
}

// AE stats from status endpoint
async function fetchAEStats() {
  try {
    const r = await fetch(`${API_BASE}/api/status`);
    const d = await r.json();
    if (dom.aeAccuracy && d.autoencoder) dom.aeAccuracy.textContent = "Loaded";
  } catch(_) {}
}

// ── UNIQUE FEATURE 1: LIVE CYBER THREAT WORLD MAP ──────────────────────────────
const threatMapState = {
  canvas: null,
  ctx: null,
  beams: [],
  ripples: [],
  width: 0,
  height: 0,
  target: { lat: 17.38, lon: 78.47, label: "India Hub" }, // Hyderabad Target
};

// stylized continent coordinates approximation
function isLandCoord(lat, lon) {
  if (lat > 10 && lat < 75 && lon > -170 && lon < -50) return true; // North America
  if (lat > -55 && lat < 12 && lon > -85 && lon < -35) return true; // South America
  if (lat > 10 && lat < 80 && lon > -15 && lon < 170) return true; // Eurasia
  if (lat > -35 && lat < 37 && lon > -20 && lon < 52) return true; // Africa
  if (lat > -42 && lat < -10 && lon > 110 && lon < 155) return true; // Australia
  if (lat > 60 && lat < 83 && lon > -75 && lon < -10) return true; // Greenland
  return false;
}

function initThreatMap() {
  const canvas = document.getElementById("threat-map-canvas");
  if (!canvas) return;
  threatMapState.canvas = canvas;
  threatMapState.ctx = canvas.getContext("2d");

  // Resize handler
  const resize = () => {
    const rect = canvas.parentNode.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    threatMapState.width = rect.width;
    threatMapState.height = rect.height;
  };
  resize();
  window.addEventListener("resize", resize);

  // Start Animation Loop
  requestAnimationFrame(threatMapLoop);
}

function getCanvasCoords(lat, lon) {
  const W = threatMapState.width;
  const H = threatMapState.height;
  // Map longitude to x: -180 to 180 -> 0 to W
  const x = (lon + 180) * (W / 360);
  // Map latitude to y: 90 to -90 -> 0 to H
  const y = (90 - lat) * (H / 180);
  return { x, y };
}

function triggerThreatMapBeam(srcGeo, label) {
  if (!threatMapState.canvas) return;
  const start = getCanvasCoords(srcGeo.lat, srcGeo.lon);
  const end = getCanvasCoords(threatMapState.target.lat, threatMapState.target.lon);

  // Control point offset for curve
  const midX = (start.x + end.x) / 2;
  const midY = (start.y + end.y) / 2 - Math.max(50, Math.abs(start.x - end.x) * 0.2);

  const color = CHART_COLORS[label] || "#ef4444";

  threatMapState.beams.push({
    x0: start.x, y0: start.y,
    x1: midX, y1: midY,
    x2: end.x, y2: end.y,
    t: 0,
    speed: 0.015 + Math.random() * 0.01,
    color: color,
  });
}

function threatMapLoop() {
  const ctx = threatMapState.ctx;
  const W = threatMapState.width;
  const H = threatMapState.height;
  if (!ctx) return;

  // Clear Canvas
  ctx.fillStyle = "rgba(5, 8, 15, 0.4)";
  ctx.fillRect(0, 0, W, H);

  // 1. Draw stylized cyber dot-matrix world map
  ctx.fillStyle = "rgba(59, 130, 246, 0.08)";
  for (let lat = -60; lat < 80; lat += 6) {
    for (let lon = -180; lon < 180; lon += 6) {
      if (isLandCoord(lat, lon)) {
        const pt = getCanvasCoords(lat, lon);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 1.2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // 2. Draw Target Hub (India)
  const targetPt = getCanvasCoords(threatMapState.target.lat, threatMapState.target.lon);
  ctx.shadowBlur = 15;
  ctx.shadowColor = "var(--accent-blue)";
  ctx.fillStyle = "var(--accent-blue)";
  ctx.beginPath();
  ctx.arc(targetPt.x, targetPt.y, 6, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(59, 130, 246, 0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(targetPt.x, targetPt.y, 14, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0; // reset shadow

  // 3. Draw and update Ripples
  threatMapState.ripples.forEach((r, idx) => {
    r.radius += r.speed;
    r.alpha -= 0.02;
    if (r.alpha <= 0) {
      threatMapState.ripples.splice(idx, 1);
      return;
    }
    ctx.strokeStyle = `rgba(239, 68, 68, ${r.alpha})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2);
    ctx.stroke();
  });

  // 4. Draw and update Laser Beams
  threatMapState.beams.forEach((b, idx) => {
    b.t += b.speed;
    if (b.t >= 1) {
      // Trigger ripple at India
      threatMapState.ripples.push({
        x: b.x2, y: b.y2,
        radius: 4,
        speed: 1.2,
        alpha: 1.0
      });
      threatMapState.beams.splice(idx, 1);
      return;
    }

    // Draw Arc Trail
    ctx.strokeStyle = b.color;
    ctx.globalAlpha = 0.25;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(b.x0, b.y0);
    ctx.quadraticCurveTo(b.x1, b.y1, b.x2, b.y2);
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Calculate moving pulse using Quadratic Bezier formula
    // B(t) = (1-t)^2*P0 + 2*(1-t)*t*P1 + t^2*P2
    const mt = 1 - b.t;
    const px = mt * mt * b.x0 + 2 * mt * b.t * b.x1 + b.t * b.t * b.x2;
    const py = mt * mt * b.y0 + 2 * mt * b.t * b.y1 + b.t * b.t * b.y2;

    // Draw flying particle
    ctx.shadowBlur = 10;
    ctx.shadowColor = b.color;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = b.color;
    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  });

  requestAnimationFrame(threatMapLoop);
}


// ── UNIQUE FEATURE 2: AI NARRATIVE ENGINE ──────────────────────────────────────
function selectAlertForNarrative(det) {
  const container = document.getElementById("narrative-panel-content");
  if (!container) return;

  container.innerHTML = `
    <div style="display:flex; justify-content:center; align-items:center; min-height: 200px;">
      <div style="border: 3px solid rgba(139,92,246,0.3); border-top: 3px solid var(--accent-purple); border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite;"></div>
    </div>
  `;

  if (det.narrative) {
    renderNarrative(det.narrative);
  } else {
    fetch(`${API_BASE}/api/narrative`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(det)
    })
    .then(res => res.json())
    .then(narrative => {
      det.narrative = narrative;
      renderNarrative(narrative);
    })
    .catch(err => {
      container.innerHTML = `<div class="text-red" style="padding:1rem;text-align:center;">Failed to generate AI narrative: ${err}</div>`;
    });
  }
}

function renderNarrative(n) {
  const container = document.getElementById("narrative-panel-content");
  if (!container || !n) return;

  const actionItems = n.actions.map(act => `<li style="margin-left:1.15rem; font-size:0.78rem; color:var(--text-secondary); margin-top:0.25rem;">${act}</li>`).join("");

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:0.75rem; animation: toastIn 0.35s ease;">
      <div style="font-size:0.88rem; font-weight:700; color:var(--text-primary); border-left: 3px solid var(--accent-purple); padding-left: 8px; line-height: 1.4;">
        ${n.headline}
      </div>
      <div style="font-size:0.8rem; color:var(--text-secondary); line-height:1.55; background:rgba(255,255,255,0.02); padding:10px 12px; border-radius:var(--radius-sm); border:1px solid rgba(255,255,255,0.04);">
        ${n.summary}
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
        <div class="metric-card" style="padding:8px 12px;">
          <div class="metric-name" style="font-size:0.7rem;">MITRE ATT&CK REFERENCE</div>
          <div class="metric-value text-blue" style="font-size:1rem; font-weight:700; margin: 2px 0;">${n.mitre.id}</div>
          <div class="metric-model" style="font-size:0.68rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${n.mitre.name} (${n.mitre.tactic})</div>
        </div>
        <div class="metric-card" style="padding:8px 12px;">
          <div class="metric-name" style="font-size:0.7rem;">ATTACKER GEOLOCATION</div>
          <div class="metric-value text-yellow" style="font-size:1.0rem; font-weight:700; margin: 2px 0;">
            ${n.src_geo.flag} ${n.src_geo.country}
          </div>
          <div class="metric-model" style="font-size:0.68rem;">${n.src_geo.city} (${n.src_geo.lat}, ${n.src_geo.lon})</div>
        </div>
      </div>
      <div>
        <div style="font-size:0.7rem; font-weight:700; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.25rem;">
          Recommended Incident Actions:
        </div>
        <ul style="list-style-type: square; padding-left: 0.5rem;">
          ${actionItems}
        </ul>
      </div>
    </div>
  `;
}


// ── UNIQUE FEATURE 3: ATTACKER BEHAVIORAL PROFILER ──────────────────────────────
async function fetchAttackerProfiles() {
  const container = document.getElementById("adversary-profiles-list");
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/api/attacker-profile`);
    const data = await res.json();

    if (!data.profiles || data.profiles.length === 0) {
      container.innerHTML = `
        <div style="text-align:center;color:var(--text-secondary);font-size:0.8rem;padding:2rem;">
          🕵️ No advanced threat actors geolocated or profiled yet.
        </div>
      `;
      return;
    }

    container.innerHTML = data.profiles.map(p => {
      const barColor = p.risk_score > 75 ? "var(--sev-critical)" : p.risk_score > 45 ? "var(--sev-high)" : "var(--accent-yellow)";
      return `
        <div class="metric-card" style="padding: 10px 14px; display:flex; flex-direction:column; gap:0.4rem; border-color: rgba(255,255,255,0.03);">
          <div style="display:flex; justify-content:between; align-items:center; width:100%;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; font-weight:700; color:var(--text-primary);">${p.src_ip}</span>
            <span style="font-size:0.7rem; background:${barColor}22; color:${barColor}; border:1px solid ${barColor}55; padding:1px 8px; border-radius:999px; font-weight:700;">
              ${p.risk_level}
            </span>
          </div>
          <div style="font-size:0.74rem; color:var(--text-secondary);">
            <strong>Kill-Chain Prediction:</strong> ${p.prediction_text} (Next stage: <span class="text-red">${p.predicted_next.toUpperCase()}</span>)
          </div>
          <div>
            <div style="display:flex; justify-content:between; font-size:0.65rem; color:var(--text-muted); margin-bottom: 2px;">
              <span>Threat Level Score</span>
              <span>${p.risk_score}/100</span>
            </div>
            <div class="progress-track" style="height:5px;">
              <div class="progress-fill" style="width:${p.risk_score}%; background:${barColor}; height:100%;"></div>
            </div>
          </div>
        </div>
      `;
    }).join("");
  } catch (_) {}
}


// ── UNIQUE FEATURE 4: ADVERSARIAL SIMULATOR ────────────────────────────────────
async function triggerAdversarialSimulation() {
  const btn = document.getElementById("btn-simulate-adv");
  if (btn) {
    btn.setAttribute("disabled", "");
    btn.textContent = "⚙️ Simulating Evasion...";
  }

  try {
    const res = await fetch(`${API_BASE}/api/adversarial`);
    const d = await res.json();
    renderAdversarialReport(d);
    showToast("Adversarial simulation complete", "info");
  } catch (_) {
    showToast("Simulation failed", "alert");
  } finally {
    if (btn) {
      btn.removeAttribute("disabled");
      btn.textContent = "⚡ Trigger Simulation Evasion";
    }
  }
}

async function fetchAdversarialReport() {
  try {
    const res = await fetch(`${API_BASE}/api/adversarial`);
    const d = await res.json();
    renderAdversarialReport(d);
  } catch (_) {}
}

function renderAdversarialReport(d) {
  const resilience = document.getElementById("adv-ensemble-resilience");
  const dnnEvasion = document.getElementById("adv-dnn-evasion");
  const rfEvasion  = document.getElementById("adv-rf-evasion");
  const dnnLbl     = document.getElementById("adv-dnn-acc-lbl");
  const rfLbl      = document.getElementById("adv-rf-acc-lbl");
  const verdict    = document.getElementById("adv-verdict-box");

  if (resilience) resilience.textContent = d.ensemble_resilience + "%";
  if (dnnEvasion)  dnnEvasion.textContent  = d.dnn.evasion_rate + "%";
  if (rfEvasion)   rfEvasion.textContent   = d.rf.evasion_rate + "%";

  if (dnnLbl) dnnLbl.textContent = `DNN Acc: ${d.dnn.original_accuracy}% -> ${d.dnn.adversarial_accuracy}%`;
  if (rfLbl)  rfLbl.textContent  = `RF Acc: ${d.rf.original_accuracy}% -> ${d.rf.adversarial_accuracy}%`;

  if (verdict) {
    verdict.textContent = d.verdict;
    if (d.ensemble_evasion_rate > 30) {
      verdict.style.background = "rgba(239, 68, 68, 0.1)";
      verdict.style.borderColor = "rgba(239, 68, 68, 0.25)";
      verdict.style.color = "var(--sev-critical)";
    } else {
      verdict.style.background = "rgba(16, 185, 129, 0.1)";
      verdict.style.borderColor = "rgba(16, 185, 129, 0.25)";
      verdict.style.color = "var(--accent-green)";
    }
  }
}


dom.alertFilter?.addEventListener("change", e=>{ state.filter=e.target.value; rebuildAlertFeed(); });
dom.btnStart?.addEventListener("click", startCapture);
dom.btnStop?.addEventListener("click",  stopCapture);
document.getElementById("btn-simulate-adv")?.addEventListener("click", triggerAdversarialSimulation);

document.addEventListener("DOMContentLoaded", ()=>{
  initTrafficChart();
  initDonutChart();
  initThreatGauge();
  initSHAPChart();
  initAnomalyGauge();
  initThreatMap();
  fetchInitialData();
  fetchAEStats();
  connectSocket();
  // Refresh SHAP data every 30 seconds
  setInterval(fetchSHAPImportance, 30000);
  // Refresh Attacker Profiles every 15 seconds
  setInterval(fetchAttackerProfiles, 15000);
});

