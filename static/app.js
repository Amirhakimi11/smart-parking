const LABEL = {available:"Available", busy:"Busy", almost_full:"Almost full", full:"Full", closed:"Closed"};
let data = null, scope = "ALL", chart = null, openFloor = null;
const $ = (id) => document.getElementById(id);
const hhmm = (iso) => iso.slice(11, 16);
const pct = (r) => Math.round(r * 100);

async function getJSON(url){ const r = await fetch(url); if(!r.ok) throw new Error(url); return r.json(); }

function addMinutes(iso, m){
  const d = new Date(iso); d.setMinutes(d.getMinutes() + m);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}`;
}

function scopeData(){
  if(!data) return null;
  if(scope === "ALL") return {name:"All levels", ...data.totals};
  const f = data.floors.find(f => f.floor === scope);
  return f ? f : {name:"All levels", ...data.totals};
}

function renderBoard(){
  const t = data.totals;
  $("site").textContent = data.site;
  $("totalFree").textContent = t.available;
  $("totalLbl").textContent = `of ${t.capacity} bays`;
  const f60 = t.forecast["60"];
  $("forecastLine").innerHTML = `In 60 minutes about <b>${f60.available} bays</b> will be free (${pct(f60.rate)}% full).`;
  $("rows").innerHTML = `<div class="head"><span>Level</span><span>Free now</span><span>In 60 min</span></div>` +
    data.floors.map(f => {
      const fullNow = f.available === 0;
      return `<div class="row ${f.level}">
        <div class="nm">${f.name}<small>${LABEL[f.level]} · ${pct(f.rate)}% full</small></div>
        <div class="free ${fullNow ? "full-txt" : ""}">${fullNow ? "FULL" : f.available}</div>
        <div class="next">${f.forecast["60"].available}<small>${LABEL[f.level_60].toLowerCase()}</small></div>
      </div>`;
    }).join("");
  const rec = data.recommendation;
  const tip = $("tip");
  if(rec){
    tip.hidden = false;
    tip.innerHTML = `<b>Best level to try:</b> ${rec.name}. ${rec.available_now} bays free now, about ${rec.available_30} expected in 30 minutes.`;
  } else { tip.hidden = false; tip.textContent = "All levels are forecast to be full. Consider public transport or a nearby car park."; }
  const v = data.vision || {};
  $("camBadge").textContent = v.mode === "yolo" ? "Live YOLOv8 inference" : "Simulated feed";
  $("camBadge").className = "badge " + (v.mode === "yolo" ? "live" : "sim");
  $("camCount").textContent = `${v.total - v.occupied} free of ${v.total} bays in view · ${v.fps} fps`;
}

function renderScope(){
  const items = [["ALL","All levels"], ...data.floors.map(f => [f.floor, f.name])];
  $("scope").innerHTML = items.map(([k, n]) => `<button class="chip" data-k="${k}" aria-pressed="${k === scope}">${n}</button>`).join("");
  $("scope").querySelectorAll("button").forEach(b => b.onclick = () => { scope = b.dataset.k; renderScope(); drawChart(); });
}

async function drawChart(){
  if(!data) return;
  const h = await getJSON(`/api/history?scope=${scope}&hours=6`);
  const sd = scopeData();
  const cap = h.capacity || sd.capacity;
  const labels = h.points.map(p => hhmm(p.t));
  const hist = h.points.map(p => pct(p.occupied / cap));
  const fcLabels = [15, 30, 45, 60].map(m => addMinutes(data.updated_at, m));
  const fc = [15, 30, 45, 60].map(m => pct(sd.forecast[String(m)].rate));
  const allLabels = labels.concat(fcLabels);
  const histData = hist.concat(fcLabels.map(() => null));
  const fcData = labels.map(() => null).concat(fc);
  if(hist.length) fcData[labels.length - 1] = hist[hist.length - 1];
  const cfg = {
    type: "line",
    data: { labels: allLabels, datasets: [
      {label: "Measured", data: histData, borderColor: "#17212B", backgroundColor: "#17212B", borderWidth: 2, pointRadius: 0, tension: .25},
      {label: "AI forecast", data: fcData, borderColor: "#E8412C", backgroundColor: "#E8412C", borderWidth: 3, borderDash: [6, 5], pointRadius: 4, tension: .25}
    ]},
    options: { responsive: true, maintainAspectRatio: false, animation: false,
      scales: { y: {min: 0, max: 100, ticks: {callback: v => v + "%"}, title: {display: true, text: "Bays occupied"}},
                x: {ticks: {maxTicksLimit: 8}} },
      plugins: { legend: {position: "bottom"}, tooltip: {callbacks: {label: c => `${c.dataset.label}: ${c.parsed.y}%`}} } }
  };
  if(chart){ chart.data = cfg.data; chart.update(); } else { chart = new Chart($("chart"), cfg); }
  $("chartHint").textContent = `${sd.name}: ${sd.forecast["60"].available} of ${sd.capacity} bays expected free in 60 minutes.`;
}

function renderDrawer(){
  $("drawerBody").innerHTML = data.floors.map(f => {
    const isOpen = openFloor === f.floor;
    const zones = isOpen ? `<div class="zones"><table>
      <tr><th>Zone</th><th>Capacity</th><th>Taken</th><th>Free</th><th>Status</th></tr>
      ${f.zones.map(z => `<tr><td>${z.name}</td><td>${z.capacity}</td><td>${z.occupied}</td><td><b>${z.available}</b></td>
        <td><span class="pill ${z.level}">${LABEL[z.level]}</span></td></tr>
        <tr><td colspan="5" style="border:0;padding-top:0;color:#5B6873;font-size:12px">In 60 min: ${z.forecast["60"].available} free (${LABEL[z.level_60].toLowerCase()})</td></tr>`).join("")}
    </table></div>` : "";
    return `<div class="floor">
      <button aria-expanded="${isOpen}" data-f="${f.floor}">
        <span class="fn">${f.name}</span><span class="pill ${f.level}">${LABEL[f.level]}</span>
        <span class="fs">${f.available} free of ${f.capacity} bays · ${f.zones.length} zones</span><span class="fs">${pct(f.rate)}% full</span>
        <span class="bar"><i class="${f.level}" style="width:${pct(f.rate)}%"></i></span>
      </button>${zones}</div>`;
  }).join("");
  $("drawerBody").querySelectorAll(".floor>button").forEach(b => b.onclick = () => {
    openFloor = openFloor === b.dataset.f ? null : b.dataset.f;
    scope = b.dataset.f; renderDrawer(); renderScope(); drawChart();
  });
}

function setDrawer(show){
  $("drawer").hidden = !show; $("bg").hidden = !show;
  if(show){ renderDrawer(); $("closeLevels").focus(); } else { $("openLevels").focus(); }
}
$("openLevels").onclick = () => setDrawer(true);
$("closeLevels").onclick = () => setDrawer(false);
$("bg").onclick = () => setDrawer(false);
document.addEventListener("keydown", e => { if(e.key === "Escape" && !$("drawer").hidden) setDrawer(false); });

async function refresh(first){
  try{
    data = await getJSON("/api/overview");
    renderBoard();
    if(first){ renderScope(); }
    if(!$("drawer").hidden) renderDrawer();
    if(first) drawChart();
  }catch(e){ $("forecastLine").textContent = "Cannot reach the server. Retrying…"; }
}
function tick(){ $("clock").textContent = new Date().toLocaleTimeString("en-MY", {hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "Asia/Kuala_Lumpur"}); }
tick(); setInterval(tick, 20000);
refresh(true);
setInterval(() => refresh(false), 10000);
setInterval(() => { renderScope(); drawChart(); }, 60000);
