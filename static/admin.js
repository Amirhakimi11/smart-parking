const $ = (id) => document.getElementById(id);
const LABEL = {available:"Available", busy:"Busy", almost_full:"Almost full", full:"Full", closed:"Closed"};
let token = sessionStorage.getItem("adm") || "", acc = null, S = null;
const pct = (r) => Math.round(r * 100);

async function api(path, opts = {}){
  const r = await fetch(path, {...opts, headers: {"Content-Type": "application/json", Authorization: "Bearer " + token, ...(opts.headers || {})}});
  if(r.status === 401){ signOut(); throw new Error("auth"); }
  if(!r.ok){ throw new Error((await r.json().catch(() => ({}))).detail || r.statusText); }
  return r;
}
function signOut(){ token = ""; sessionStorage.removeItem("adm"); $("app").hidden = true; $("loginBox").hidden = false; $("logout").hidden = true; }

$("loginBtn").onclick = async () => {
  $("err").textContent = "";
  try{
    const r = await fetch("/api/admin/login", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({password: $("pw").value})});
    if(!r.ok) throw new Error("Wrong password. Try again.");
    token = (await r.json()).token; sessionStorage.setItem("adm", token); start();
  }catch(e){ $("err").textContent = e.message; }
};
$("pw").addEventListener("keydown", e => { if(e.key === "Enter") $("loginBtn").click(); });
$("logout").onclick = signOut;

function render(){
  const t = S.totals, f60 = t.forecast["60"];
  $("site").textContent = S.site + " · Operations";
  $("kOcc").textContent = pct(t.rate) + "%";
  $("kFree").textContent = t.available;
  $("kF60").textContent = pct(f60.rate) + "%";
  $("kAlerts").textContent = S.alerts.length;
  $("alerts").innerHTML = S.alerts.length ? S.alerts.map(a => `<div class="alert">${a.message}</div>`).join("") : `<p class="ok-note">No alerts. Every open zone is expected to stay below 97% for the next hour.</p>`;
  $("scen").innerHTML = ["normal", "peak", "event", "quiet"].map(n =>
    `<button class="btn ghost small" data-n="${n}" aria-pressed="${S.scenario === n}">${{normal:"Normal day", peak:"Peak hour", event:"Mall event", quiet:"Quiet day"}[n]}</button>`).join("");
  $("scen").querySelectorAll("button").forEach(b => b.onclick = async () => { await api("/api/admin/scenario", {method: "POST", body: JSON.stringify({name: b.dataset.n, minutes: 10})}); load(); });

  const rows = S.floors.flatMap(f => f.zones);
  const flat = Object.fromEntries(S.zones_flat.map(z => [z.id, z]));
  $("zones").innerHTML = `<tr><th>Zone</th><th>Source</th><th>Capacity</th><th>Taken</th><th>Free</th><th>Status</th><th>In 60 min</th><th>Open</th></tr>` +
    rows.map(z => `<tr>
      <td><b>${z.name}</b></td><td>${z.source === "yolov8" ? "YOLOv8 camera" : "Simulated sensor"}</td>
      <td><input class="num" type="number" min="1" max="2000" value="${flat[z.id].capacity}" data-cap="${z.id}" aria-label="Capacity of ${z.name}"></td>
      <td>${z.occupied}</td><td>${z.available}</td><td><span class="pill ${z.level}">${LABEL[z.level]}</span></td>
      <td><span class="pill ${z.level_60}">${LABEL[z.level_60]}</span> ${z.forecast["60"].available} free</td>
      <td><button class="toggle ${z.is_open ? "open" : "shut"}" data-open="${z.id}" aria-pressed="${z.is_open}">${z.is_open ? "Open" : "Closed"}</button></td></tr>`).join("");
  $("zones").querySelectorAll("[data-cap]").forEach(i => i.onchange = async () => { await api("/api/admin/zones/" + i.dataset.cap, {method: "PATCH", body: JSON.stringify({capacity: +i.value})}); load(); });
  $("zones").querySelectorAll("[data-open]").forEach(b => b.onclick = async () => { await api("/api/admin/zones/" + b.dataset.open, {method: "PATCH", body: JSON.stringify({is_open: b.getAttribute("aria-pressed") !== "true"})}); load(); });

  const m = S.metrics || {};
  const row = (a, b) => `<div class="metric"><span>${a}</span><b>${b}</b></div>`;
  $("metrics").innerHTML = m.mae_bays ? [
    row("Model", m.model),
    row("Average error (all horizons)", m.mae_bays.toFixed(2) + " bays"),
    row("Average error at +60 min", m.mae_bays_60min.toFixed(2) + " bays"),
    row("“Nothing changes” baseline error", m.baseline_mae_bays.toFixed(2) + " bays"),
    row("Improvement over baseline", m.improvement_vs_baseline_pct + "%"),
    row("Test period", m.test_days + " days · " + m.test_rows.toLocaleString() + " rows")].join("") : "<p class='ok-note'>No model yet.</p>";
  const v = S.vision || {};
  $("health").innerHTML = [
    row("Database", S.db === "postgresql" ? "Cloud PostgreSQL" : "Local SQLite"),
    row("Readings stored", S.readings_total.toLocaleString()),
    row("Camera mode", v.mode === "yolo" ? "Live YOLOv8" : "Simulated feed"),
    row("Camera speed", (v.fps || 0) + " fps"),
    row("Last update", (S.updated_at || "").replace("T", " "))].join("");

  const a = S.accuracy;
  if(a && a.points && a.points.length){
    $("accHint").textContent = `Last 6 hours · ${a.n} predictions · average error ${a.mae_bays} bays (${a.mae_pct}% of zone capacity).`;
    const labels = a.points.map(p => p.t.slice(11, 16));
    const cfg = {type: "line", data: {labels, datasets: [
      {label: "Actual", data: a.points.map(p => p.actual), borderColor: "#17212B", borderWidth: 2, pointRadius: 0, tension: .25},
      {label: "Predicted 60 min earlier", data: a.points.map(p => p.pred), borderColor: "#E8412C", borderWidth: 2, borderDash: [6, 4], pointRadius: 0, tension: .25}]},
      options: {responsive: true, maintainAspectRatio: false, animation: false, scales: {y: {title: {display: true, text: "Bays occupied"}}, x: {ticks: {maxTicksLimit: 8}}}, plugins: {legend: {position: "bottom"}}}};
    if(acc){ acc.data = cfg.data; acc.update(); } else acc = new Chart($("accChart"), cfg);
  }
}

async function load(){ S = await (await api("/api/admin/summary")).json(); render(); }

$("csv").onclick = async () => {
  const blob = await (await api("/api/admin/export.csv?hours=24")).blob();
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "parking_readings.csv"; a.click();
};

function start(){ $("loginBox").hidden = true; $("app").hidden = false; $("logout").hidden = false; load().catch(() => {}); }
if(token) start(); else signOut();
setInterval(() => { if(token) load().catch(() => {}); }, 15000);
