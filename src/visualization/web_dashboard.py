"""
Self-Contained Live Web Dashboard Server for Bitcoin Price Predictor (1 FPS).

Provides an interactive real-time visualizer accessible via browser or Snowflake iframe:
- Real-time animated canvas chart: Actual Price (solid green/cyan), Predicted Path (dotted orange).
- Real-time Accuracy Bar & Stability Gauges.
- Random day market selector across 2020 - 2026 regimes.
- 1 FPS real-time replay loop with play/pause and fast-forward controls.
- Pure zero-external-dependency self-contained HTML5/Canvas rendering.
"""

import os
import sys
import json
import time
import random
import datetime
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional

try:
    from ..simulator.market_simulator import MarketSimulator
    from ..analyser.market_analyser import MarketAnalyser
    from ..predictor.model import HybridPredictor
    from ..predictor.online_trainer import OnlineRewardTrainer
except ImportError:
    from simulator.market_simulator import MarketSimulator
    from analyser.market_analyser import MarketAnalyser
    from predictor.model import HybridPredictor
    from predictor.online_trainer import OnlineRewardTrainer


class GlobalDashboardState:
    """Thread-safe global state for live streaming dashboard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = True
        self.fps = 1.0
        self.speed_multiplier = 1.0
        self.current_day_str = "2024-03-15"

        # Initialize core components
        self.simulator = MarketSimulator(random_day=True, start_year=2020, end_year=2026)
        self.analyser = MarketAnalyser()
        self.predictor = HybridPredictor(device="cpu", horizon_mins=25)
        self.trainer = OnlineRewardTrainer(self.predictor)

        # Ring buffers for web client
        self.history_len = 300
        self.hist_timestamps = []
        self.hist_prices = []
        self.latest_tick = None
        self.latest_prediction = []
        self.latest_ground_truth = []
        self.latest_eval = None

    def seek_random_day(self, year_start: int = 2020, year_end: int = 2026):
        with self.lock:
            self.analyser.reset()
            sampled_dt = self.simulator.seek_to_random_window(window_seconds=86400, start_year=year_start, end_year=year_end)
            self.current_day_str = sampled_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            self.hist_timestamps.clear()
            self.hist_prices.clear()
            self.latest_tick = None
            self.latest_prediction = []
            self.latest_ground_truth = []
            self.latest_eval = None

    def tick_step(self):
        with self.lock:
            if not self.simulator.has_next():
                self.simulator.seek_to_random_window()

            future_gt = self.simulator.get_future_ground_truth(horizon_seconds=1500)
            tick = self.simulator.next_tick()
            if tick is None:
                return

            features = self.analyser.process_tick(tick)
            past_12h = self.simulator.get_past_window_prices(window_seconds=43200)

            pred_trajectory = self.predictor.predict_25min_trajectory(
                current_price=tick.price,
                features=features.feature_vector,
                past_12h_prices=past_12h
            )

            eval_res = self.trainer.evaluate_step(
                timestamp=tick.timestamp,
                current_price=tick.price,
                predicted_trajectory=pred_trajectory,
                future_ground_truth=future_gt,
                update_model=True
            )

            self.hist_timestamps.append(tick.timestamp)
            self.hist_prices.append(tick.price)
            if len(self.hist_timestamps) > self.history_len:
                self.hist_timestamps.pop(0)
                self.hist_prices.pop(0)

            # Subsample 1500 future seconds to 50 visualization points for lightweight JSON transport
            step = 30  # every 30 seconds
            self.latest_prediction = [round(float(p), 2) for p in pred_trajectory[::step]]
            self.latest_ground_truth = [round(float(p), 2) for p in future_gt[::step]]
            self.latest_tick = tick.to_dict()
            self.latest_eval = {
                "timestamp": eval_res.timestamp,
                "current_price": eval_res.current_price,
                "pred_final_price": eval_res.pred_final_price,
                "true_final_price": eval_res.true_final_price,
                "mae_dollars": eval_res.mae_dollars,
                "rmse_dollars": eval_res.rmse_dollars,
                "directional_accuracy": eval_res.directional_accuracy,
                "accuracy_percentage": eval_res.accuracy_percentage,
                "stability_score": eval_res.stability_score,
                "realtime_reward": eval_res.realtime_reward,
                "summary": self.trainer.get_summary_stats(),
                "day_info": self.current_day_str,
            }

    def get_payload(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "history_prices": self.hist_prices,
                "history_timestamps": self.hist_timestamps,
                "predicted_trajectory": self.latest_prediction,
                "ground_truth_trajectory": self.latest_ground_truth,
                "eval": self.latest_eval,
                "is_running": self.is_running,
                "day_str": self.current_day_str,
            }


GLOBAL_STATE = GlobalDashboardState()


def background_simulation_loop():
    """Background thread advancing the market simulation at 1 FPS."""
    while True:
        if GLOBAL_STATE.is_running:
            GLOBAL_STATE.tick_step()
        sleep_sec = 1.0 / max(0.1, GLOBAL_STATE.fps * GLOBAL_STATE.speed_multiplier)
        time.sleep(sleep_sec)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bitcoin Real-Time 25-Min Price Predictor</title>
<style>
  :root {
    --bg-dark: #0a0e17;
    --card-bg: #131c2e;
    --card-border: #1f2d48;
    --accent-cyan: #00e5ff;
    --accent-green: #00e676;
    --accent-orange: #ff9100;
    --accent-red: #ff5252;
    --text-main: #f0f4f8;
    --text-muted: #8ca0ba;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
  body { background: var(--bg-dark); color: var(--text-main); min-height: 100vh; padding: 18px; }
  .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; border-bottom: 1px solid var(--card-border); padding-bottom: 12px; }
  .title-group h1 { font-size: 20px; color: var(--accent-cyan); display: flex; align-items: center; gap: 8px; }
  .title-group p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
  .controls { display: flex; gap: 8px; align-items: center; }
  button { background: #1a2844; color: #fff; border: 1px solid #2d4370; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; transition: all 0.2s; }
  button:hover { background: #263b63; border-color: var(--accent-cyan); }
  button.active { background: #00b4d8; color: #000; font-weight: bold; }
  
  .grid-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .metric-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 12px 14px; position: relative; overflow: hidden; }
  .metric-card::before { content: ""; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--accent-cyan); }
  .metric-card.green::before { background: var(--accent-green); }
  .metric-card.orange::before { background: var(--accent-orange); }
  .metric-card.red::before { background: var(--accent-red); }
  .metric-title { font-size: 11px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }
  .metric-val { font-size: 20px; font-weight: bold; margin-top: 4px; color: #fff; }
  .metric-sub { font-size: 11px; color: var(--text-muted); margin-top: 3px; }

  .chart-container { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 14px; margin-bottom: 16px; position: relative; }
  .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .legend { display: flex; gap: 14px; font-size: 12px; }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  canvas { width: 100%; height: 340px; display: block; }

  .accuracy-panel { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 14px; }
  .bar-row { margin-bottom: 12px; }
  .bar-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: var(--text-muted); }
  .bar-bg { width: 100%; height: 10px; background: #0c121e; border-radius: 5px; overflow: hidden; border: 1px solid #1a253a; }
  .bar-fill { height: 100%; width: 50%; transition: width 0.3s ease, background-color 0.3s; }
  
  .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  .badge-success { background: rgba(0, 230, 118, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }
  .badge-danger { background: rgba(255, 82, 82, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }
</style>
</head>
<body>

<div class="header">
  <div class="title-group">
    <h1><span>⚡</span> BTC/USD 25-Min Real-Time Price Predictor</h1>
    <p id="sub-header">Replaying second-by-second historical market regime • Continuous Real-Time Reward Training</p>
  </div>
  <div class="controls">
    <button id="btn-random" onclick="pickRandomDay()">🎲 Random Day (2020-2026)</button>
    <button id="btn-toggle" onclick="togglePlay()">⏸ Pause</button>
    <button onclick="setSpeed(1.0)" id="spd-1" class="active">1x (1 FPS)</button>
    <button onclick="setSpeed(5.0)" id="spd-5">5x</button>
    <button onclick="setSpeed(20.0)" id="spd-20">20x</button>
  </div>
</div>

<div class="grid-metrics">
  <div class="metric-card">
    <div class="metric-title">Current Market Price</div>
    <div class="metric-val" id="val-price">$0.00</div>
    <div class="metric-sub" id="val-time">--:--:-- UTC</div>
  </div>
  <div class="metric-card orange">
    <div class="metric-title">25-Min Predicted Price</div>
    <div class="metric-val" id="val-pred">$0.00</div>
    <div class="metric-sub" id="val-pred-delta">+0.00%</div>
  </div>
  <div class="metric-card green">
    <div class="metric-title">Directional Win Rate</div>
    <div class="metric-val" id="val-winrate">0.0%</div>
    <div class="metric-sub" id="val-match-status"><span class="badge badge-success">SYNCED</span></div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Mean Absolute Error (MAE)</div>
    <div class="metric-val" id="val-mae">$0.00</div>
    <div class="metric-sub" id="val-rmse">RMSE: $0.00</div>
  </div>
  <div class="metric-card green">
    <div class="metric-title">Real-Time Online Reward</div>
    <div class="metric-val" id="val-reward">+0.000</div>
    <div class="metric-sub">Continuous sec-by-sec policy</div>
  </div>
</div>

<div class="chart-container">
  <div class="chart-header">
    <span style="font-size:13px; font-weight:600; color:#b0bec5;" id="chart-regime-title">Live Second-by-Second Stream & 25-Min Forecast Trajectory</span>
    <div class="legend">
      <div class="legend-item"><span class="dot" style="background:var(--accent-green)"></span> Actual Market Price</div>
      <div class="legend-item"><span class="dot" style="background:var(--accent-orange)"></span> 25-Min Predicted (Dotted)</div>
      <div class="legend-item"><span class="dot" style="background:#2979ff"></span> Ground Truth (Stored Data)</div>
    </div>
  </div>
  <canvas id="marketCanvas"></canvas>
</div>

<div class="accuracy-panel">
  <div class="bar-row">
    <div class="bar-label">
      <span>Trajectory Accuracy Match Score (Price Proximity & Direction)</span>
      <strong id="bar-acc-val">50.0%</strong>
    </div>
    <div class="bar-bg">
      <div id="bar-acc-fill" class="bar-fill" style="width: 50%; background: var(--accent-cyan);"></div>
    </div>
  </div>
  <div class="bar-row" style="margin-bottom:0;">
    <div class="bar-label">
      <span>Prediction Stability & Smoothness Score (Kalman Regularization)</span>
      <strong id="bar-stab-val">100.0%</strong>
    </div>
    <div class="bar-bg">
      <div id="bar-stab-fill" class="bar-fill" style="width: 100%; background: var(--accent-green);"></div>
    </div>
  </div>
</div>

<script>
  let isRunning = true;
  let canvas = document.getElementById("marketCanvas");
  let ctx = canvas.getContext("2d");

  function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth - 28;
    canvas.height = 340;
  }
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  function pickRandomDay() {
    fetch("/api/random_day").then(() => fetchUpdate());
  }

  function togglePlay() {
    fetch("/api/toggle_play").then(res => res.json()).then(data => {
      isRunning = data.is_running;
      document.getElementById("btn-toggle").innerText = isRunning ? "⏸ Pause" : "▶ Resume";
    });
  }

  function setSpeed(spd) {
    fetch("/api/set_speed?speed=" + spd).then(() => {
      document.querySelectorAll("[id^='spd-']").forEach(b => b.classList.remove("active"));
      if(spd === 1.0) document.getElementById("spd-1").classList.add("active");
      if(spd === 5.0) document.getElementById("spd-5").classList.add("active");
      if(spd === 20.0) document.getElementById("spd-20").classList.add("active");
    });
  }

  function renderChart(histPrices, predPrices, gtPrices) {
    if (!histPrices || histPrices.length === 0) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let allPrices = [...histPrices, ...(predPrices || []), ...(gtPrices || [])];
    let minP = Math.min(...allPrices);
    let maxP = Math.max(...allPrices);
    let pad = (maxP - minP) * 0.1 || 10;
    minP -= pad;
    maxP += pad;

    let totalPoints = histPrices.length + (predPrices ? predPrices.length : 0);
    let currentX = (histPrices.length / totalPoints) * canvas.width;

    function getY(p) {
      return canvas.height - ((p - minP) / (maxP - minP)) * (canvas.height - 40) - 20;
    }

    // Grid lines
    ctx.strokeStyle = "#1a253a";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
      let y = (canvas.height / 5) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();

      let pVal = maxP - (i / 5) * (maxP - minP);
      ctx.fillStyle = "#5c7090";
      ctx.font = "10px sans-serif";
      ctx.fillText("$" + pVal.toFixed(2), 6, y - 4);
    }

    // Vertical line separating history and future forecast
    ctx.strokeStyle = "rgba(0, 229, 255, 0.3)";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(currentX, 0);
    ctx.lineTo(currentX, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#00e5ff";
    ctx.font = "10px sans-serif";
    ctx.fillText("NOW (t=0)", currentX - 25, 14);

    // 1. Draw Historical Prices (Solid Green)
    ctx.strokeStyle = "#00e676";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < histPrices.length; i++) {
      let x = (i / totalPoints) * canvas.width;
      let y = getY(histPrices[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    let lastHistY = getY(histPrices[histPrices.length - 1]);

    // 2. Draw Ground Truth Future (Blue Dashed)
    if (gtPrices && gtPrices.length > 0) {
      ctx.strokeStyle = "rgba(41, 121, 255, 0.7)";
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(currentX, lastHistY);
      for (let i = 0; i < gtPrices.length; i++) {
        let x = ((histPrices.length + i) / totalPoints) * canvas.width;
        let y = getY(gtPrices[i]);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 3. Draw Model Predicted 25-Min Future (Dotted Orange)
    if (predPrices && predPrices.length > 0) {
      ctx.strokeStyle = "#ff9100";
      ctx.setLineDash([2, 4]);
      ctx.lineWidth = 3.0;
      ctx.beginPath();
      ctx.moveTo(currentX, lastHistY);
      for (let i = 0; i < predPrices.length; i++) {
        let x = ((histPrices.length + i) / totalPoints) * canvas.width;
        let y = getY(predPrices[i]);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Glow on latest prediction point
      let lastPredX = ((histPrices.length + predPrices.length - 1) / totalPoints) * canvas.width;
      let lastPredY = getY(predPrices[predPrices.length - 1]);
      ctx.fillStyle = "#ff9100";
      ctx.beginPath();
      ctx.arc(lastPredX, lastPredY, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function fetchUpdate() {
    fetch("/api/data").then(r => r.json()).then(data => {
      if (!data.eval) return;
      let ev = data.eval;
      
      document.getElementById("val-price").innerText = "$" + ev.current_price.toLocaleString(undefined, {minimumFractionDigits: 2});
      let dt = new Date(ev.timestamp * 1000);
      document.getElementById("val-time").innerText = dt.toISOString().replace("T", " ").substring(0, 19) + " UTC";

      document.getElementById("val-pred").innerText = "$" + ev.pred_final_price.toLocaleString(undefined, {minimumFractionDigits: 2});
      let diffPct = ((ev.pred_final_price - ev.current_price) / ev.current_price) * 100;
      document.getElementById("val-pred-delta").innerText = (diffPct >= 0 ? "+" : "") + diffPct.toFixed(2) + "% (" + (diffPct >= 0 ? "▲ BULLISH" : "▼ BEARISH") + ")";
      document.getElementById("val-pred-delta").style.color = diffPct >= 0 ? "var(--accent-green)" : "var(--accent-red)";

      document.getElementById("val-winrate").innerText = ev.summary.directional_accuracy_pct + "%";
      document.getElementById("val-match-status").innerHTML = ev.directional_accuracy
        ? '<span class="badge badge-success">✓ PREDICTION MATCH</span>'
        : '<span class="badge badge-danger">✗ DIVERGENT</span>';

      document.getElementById("val-mae").innerText = "$" + ev.mae_dollars.toFixed(2);
      document.getElementById("val-rmse").innerText = "RMSE: $" + ev.rmse_dollars.toFixed(2);
      document.getElementById("val-reward").innerText = (ev.realtime_reward >= 0 ? "+" : "") + ev.realtime_reward.toFixed(3);
      document.getElementById("val-reward").style.color = ev.realtime_reward >= 0 ? "var(--accent-green)" : "var(--accent-red)";

      document.getElementById("chart-regime-title").innerText = "Market Regime: " + data.day_str + " (1 FPS Real-Time Stream)";

      // Bars
      document.getElementById("bar-acc-val").innerText = ev.accuracy_percentage.toFixed(1) + "%";
      document.getElementById("bar-acc-fill").style.width = Math.min(100, Math.max(5, ev.accuracy_percentage)) + "%";
      document.getElementById("bar-acc-fill").style.background = ev.accuracy_percentage >= 60 ? "var(--accent-green)" : (ev.accuracy_percentage >= 40 ? "var(--accent-cyan)" : "var(--accent-red)");

      document.getElementById("bar-stab-val").innerText = ev.stability_score.toFixed(1) + "%";
      document.getElementById("bar-stab-fill").style.width = Math.min(100, Math.max(5, ev.stability_score)) + "%";

      renderChart(data.history_prices, data.predicted_trajectory, data.ground_truth_trajectory);
    });
  }

  // 1 FPS polling loop (1000ms)
  setInterval(fetchUpdate, 1000);
  fetchUpdate();
</script>
</body>
</html>
"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for live web visualizer."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif path == "/api/data":
            payload = GLOBAL_STATE.get_payload()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif path == "/api/random_day":
            GLOBAL_STATE.seek_random_day()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif path == "/api/toggle_play":
            GLOBAL_STATE.is_running = not GLOBAL_STATE.is_running
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"is_running": GLOBAL_STATE.is_running}).encode("utf-8"))
        elif path == "/api/set_speed":
            qs = parse_qs(parsed.query)
            spd = float(qs.get("speed", [1.0])[0])
            GLOBAL_STATE.speed_multiplier = max(0.1, min(100.0, spd))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence standard HTTP logs to avoid clutter
        pass


def run_dashboard_server(host: str = "0.0.0.0", port: int = 8000):
    """Starts the web dashboard background thread and HTTP server."""
    t_sim = threading.Thread(target=background_simulation_loop, daemon=True)
    t_sim.start()

    server = HTTPServer((host, port), DashboardRequestHandler)
    print(f"Bitcoin Predictor Live Web Dashboard running at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    run_dashboard_server(port=8000)
