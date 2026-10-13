"""Script to generate interactive_input.html and interactive_output.html."""

import json
from pathlib import Path
import numpy as np

# Load processed data
data_dir = Path("data/processed/plasticity")
train = np.load(data_dir / "train.npz")

die_boundary = train["die_boundary"]  # (900, 101, 2)
coords = train["coords"]              # (900, 101, 31, 20, 2)
displacements = train["displacements"] # (900, 101, 31, 20, 2)

out_dir = Path("results/DatasetAnalysis")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. BUILD INTERACTIVE INPUT HTML
# -------------------------------------------------------------
num_input_samples = 50
input_samples = []
for i in range(num_input_samples):
    x_pts = die_boundary[i, :, 0].tolist()
    y_pts = die_boundary[i, :, 1].tolist()
    input_samples.append({
        "id": i,
        "x": [round(v, 3) for v in x_pts],
        "y": [round(v, 3) for v in y_pts],
        "min_y": round(float(np.min(y_pts)), 3),
        "max_y": round(float(np.max(y_pts)), 3),
        "ptp_y": round(float(np.ptp(y_pts)), 3),
        "mean_y": round(float(np.mean(y_pts)), 3),
    })

input_json_str = json.dumps(input_samples)

input_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Input Parameter Explorer - 2D Plasticity Benchmark</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #334155;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #475569;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }
        .header {
            max-width: 1200px;
            margin: 0 auto 24px auto;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
        }
        .header h1 { font-size: 24px; color: var(--accent); margin-bottom: 6px; }
        .header p { color: var(--text-secondary); font-size: 14px; }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 24px;
        }
        @media (max-width: 900px) {
            .container { grid-template-columns: 1fr; }
        }
        
        .sidebar {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
            height: fit-content;
        }
        .control-group { margin-bottom: 20px; }
        .control-group label {
            display: block;
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        select, input[type="range"] {
            width: 100%;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            outline: none;
        }
        .stats-card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 14px;
            margin-top: 16px;
            font-size: 13px;
        }
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .stats-row:last-child { border: none; margin: 0; padding: 0; }
        .stats-label { color: var(--text-secondary); }
        .stats-value { font-weight: bold; color: var(--accent); }

        .main-panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--border);
        }
        .canvas-container {
            background: #090d16;
            border-radius: 8px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            margin-bottom: 20px;
        }
        canvas { display: block; width: 100%; height: 420px; cursor: crosshair; }
        
        .tooltip {
            position: absolute;
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid var(--accent);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            color: #fff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        
        .explanation-box {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 18px;
            margin-top: 20px;
            font-size: 14px;
            border-left: 4px solid var(--accent);
        }
        .explanation-box h3 { font-size: 15px; margin-bottom: 8px; color: var(--accent); }
        .explanation-box ul { padding-left: 20px; color: var(--text-secondary); font-size: 13.5px; }
        .explanation-box li { margin-bottom: 6px; }
        
        .btn-group { display: flex; gap: 8px; margin-top: 10px; }
        .btn {
            flex: 1;
            padding: 8px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .btn:hover { background: var(--accent); color: #000; }
        .btn.active { background: var(--accent); color: #000; font-weight: bold; }
        
        /* Physical Parameter Table Styles */
        .param-table-card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            border-left: 4px solid var(--accent);
        }
        .param-table-card h3 {
            font-size: 15px;
            margin-bottom: 6px;
            color: var(--accent);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }
        .provenance-badge {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .source-note {
            font-size: 12.5px;
            color: var(--text-secondary);
            margin-bottom: 14px;
            line-height: 1.5;
        }
        .source-note strong {
            color: #f59e0b;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 8px;
        }
        .param-table th, .param-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .param-table th {
            background: rgba(11, 15, 25, 0.5);
            color: var(--text-secondary);
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .param-table tr:hover td {
            background: rgba(255, 255, 255, 0.03);
        }
        .param-table td.param-name {
            font-weight: 600;
            color: var(--text-primary);
        }
        .param-table td.param-val {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: var(--accent);
            font-weight: 500;
        }
        .param-table td.param-role {
            color: var(--text-secondary);
        }
    </style>
</head>
<body>

<div class="header">
    <h1>📐 2D Plasticity Benchmark: Input Parameter Explorer</h1>
    <p>Understand the single physical input parameter that governs each simulation: the rigid impacting die boundary geometry S_d(x).</p>
</div>

<div class="container">
    <div class="sidebar">
        <div class="control-group">
            <label for="sampleSelect">Select Simulation Sample</label>
            <select id="sampleSelect"></select>
        </div>

        <div class="control-group">
            <label>Sample Index Slider (0 - 49)</label>
            <input type="range" id="sampleSlider" min="0" max="49" value="0">
        </div>

        <div class="control-group">
            <label>Visualization Mode</label>
            <div class="btn-group">
                <button class="btn active" id="btnSingle">Single Sample</button>
                <button class="btn" id="btnOverlay">Overlay 10 Samples</button>
            </div>
        </div>

        <div class="control-group">
            <label>Show Metal Block</label>
            <div class="btn-group">
                <button class="btn active" id="btnShowBlock">Yes (Underneath)</button>
                <button class="btn" id="btnHideBlock">Die Only</button>
            </div>
        </div>

        <div class="stats-card">
            <div class="stats-row">
                <span class="stats-label">Sample ID:</span>
                <span class="stats-value" id="statId">#0</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Number of Points:</span>
                <span class="stats-value">101 nodes</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Span in X:</span>
                <span class="stats-value">50.00 mm [-49.65, 0.35]</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Min Elevation:</span>
                <span class="stats-value" id="statMin">14.91 mm</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Max Elevation:</span>
                <span class="stats-value" id="statMax">23.50 mm</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Height Variation (Δy):</span>
                <span class="stats-value" id="statPtp">8.59 mm</span>
            </div>
        </div>
    </div>

    <div class="main-panel">
        <div class="canvas-container">
            <canvas id="plotCanvas" width="800" height="420"></canvas>
            <div class="tooltip" id="tooltip"></div>
        </div>

        <div class="explanation-box">
            <h3>💡 What is this Input Parameter?</h3>
            <ul>
                <li><strong>What it is:</strong> The raw 2D input tensor has shape <code>(987, 101)</code>. Each row contains 101 height points describing the physical shape of a rigid die S_d(x) across the top width of the metal workpiece.</li>
                <li><strong>How it was generated:</strong> In the Geo-FNO study, the authors randomly selected 8 control points across the 50 mm span and connected them with <em>piecewise cubic Hermite splines</em>.</li>
                <li><strong>Why is it the only input?</strong> All material properties (Young's modulus E = 200 GPa, yield strength = 70 MPa), block dimensions (50 mm x 15 mm), and die speed (v = 3 m/s) are completely constant. The neural surrogate's goal is to learn the <strong>Geometry-to-Deformation Operator</strong>: mapping this die contour directly to the resulting 20-step internal plastic deformation.</li>
            </ul>
        </div>

        <div class="param-table-card">
            <h3>
                <span>⚙️ Underlying Simulation & Physical Parameters</span>
                <span class="provenance-badge">Source: Geo-FNO Paper (Not in .mat)</span>
            </h3>
            <p class="source-note">
                <strong>Important Provenance Note:</strong> These constitutive parameters and simulation boundary conditions are defined in the original benchmark paper (<em>"Fourier Neural Operator with Learned Deformations for PDEs on General Geometries"</em>, Li et al., NeurIPS 2022 / JMLR 2023, arXiv:2207.05209). They are <strong>not</strong> stored as header metadata inside the raw <code>plas_N987_T20.mat</code> file itself (the <code>.mat</code> file stores only the resulting numerical tensors <code>input</code> and <code>output</code>).
            </p>
            <div style="overflow-x: auto;">
                <table class="param-table">
                    <thead>
                        <tr>
                            <th>Physical Parameter</th>
                            <th>Value</th>
                            <th>Role in Simulation</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="param-name">Material Constitutive Law</td>
                            <td class="param-val">J₂ Elasto-plastic (von Mises)</td>
                            <td class="param-role">Isotropic hardening plasticity</td>
                        </tr>
                        <tr>
                            <td class="param-name">Young's Modulus (E)</td>
                            <td class="param-val">200 GPa</td>
                            <td class="param-role">Elastic stiffness (steel-like)</td>
                        </tr>
                        <tr>
                            <td class="param-name">Initial Yield Stress (σ_Y)</td>
                            <td class="param-val">70 MPa</td>
                            <td class="param-role">Threshold where permanent plastic deformation begins</td>
                        </tr>
                        <tr>
                            <td class="param-name">Poisson's Ratio (ν)</td>
                            <td class="param-val">0.3</td>
                            <td class="param-role">Transverse strain contraction ratio</td>
                        </tr>
                        <tr>
                            <td class="param-name">Mass Density (ρ)</td>
                            <td class="param-val">7850 kg/m³</td>
                            <td class="param-role">Structural mass for dynamic equilibrium</td>
                        </tr>
                        <tr>
                            <td class="param-name">Workpiece Geometry</td>
                            <td class="param-val">50.0 mm × 15.0 mm</td>
                            <td class="param-role">Rectangular undeformed domain</td>
                        </tr>
                        <tr>
                            <td class="param-name">FEA Discretization</td>
                            <td class="param-val">3,000 CPS4R elements</td>
                            <td class="param-role">4-node bilinear plane stress/strain with reduced integration</td>
                        </tr>
                        <tr>
                            <td class="param-name">Mesh Dimensions</td>
                            <td class="param-val">100 × 30 elements</td>
                            <td class="param-role">Exactly (100 + 1) × (30 + 1) = 101 × 31 = 3, 131 nodes</td>
                        </tr>
                        <tr>
                            <td class="param-name">Boundary Conditions</td>
                            <td class="param-val">Clamped base (u_y = 0)</td>
                            <td class="param-role">Workpiece rests on a rigid horizontal floor</td>
                        </tr>
                        <tr>
                            <td class="param-name">Die Speed / Indentation</td>
                            <td class="param-val">v = 3.0 m/s downward</td>
                            <td class="param-role">Rigid, frictionless die pressed into the top boundary</td>
                        </tr>
                        <tr>
                            <td class="param-name">Time Discretization</td>
                            <td class="param-val">T = 20 uniform increments</td>
                            <td class="param-role">t ∈ {t₀, t₁, …, t₁₉}</td>
                        </tr>
                        <tr>
                            <td class="param-name">Total Runs Generated</td>
                            <td class="param-val">N = 987 simulations</td>
                            <td class="param-role">900 training, 7 validation, 80 testing</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<script>
const SAMPLES = __INPUT_JSON__;

let currentSampleIdx = 0;
let overlayMode = false;
let showBlock = true;

const select = document.getElementById("sampleSelect");
const slider = document.getElementById("sampleSlider");
const canvas = document.getElementById("plotCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

// Populate dropdown
SAMPLES.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = `Simulation Sample #${s.id} (Δy = ${s.ptp_y} mm)`;
    select.appendChild(opt);
});

function updateStats(s) {
    document.getElementById("statId").textContent = `#${s.id}`;
    document.getElementById("statMin").textContent = `${s.min_y} mm`;
    document.getElementById("statMax").textContent = `${s.max_y} mm`;
    document.getElementById("statPtp").textContent = `${s.ptp_y} mm`;
}

function render() {
    const s = SAMPLES[currentSampleIdx];
    updateStats(s);

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 60, padR = 30, padT = 40, padB = 60;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const minX = -52, maxX = 2;
    const minY = -2, maxY = 28;

    function toScreenX(x) { return padL + ((x - minX) / (maxX - minX)) * plotW; }
    function toScreenY(y) { return padT + (1 - (y - minY) / (maxY - minY)) * plotH; }

    // 1. Draw Grid
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    for (let x = -50; x <= 0; x += 10) {
        const sx = toScreenX(x);
        ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, h - padB); ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = "11px sans-serif";
        ctx.fillText(`${x}`, sx - 8, h - padB + 16);
    }
    for (let y = 0; y <= 25; y += 5) {
        const sy = toScreenY(y);
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(w - padR, sy); ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = "11px sans-serif";
        ctx.fillText(`${y}`, padL - 28, sy + 4);
    }

    // Axis Labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px sans-serif";
    ctx.fillText("Horizontal Position x [mm]", w / 2 - 60, h - 18);
    ctx.save();
    ctx.translate(16, h / 2 + 40);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Elevation y [mm]", 0, 0);
    ctx.restore();

    // 2. Draw undeformed metal workpiece block underneath (50 mm x 15 mm)
    if (showBlock) {
        const bx1 = toScreenX(-49.65);
        const bx2 = toScreenX(0.35);
        const byTop = toScreenY(15.0);
        const byBot = toScreenY(0.0);

        ctx.fillStyle = "rgba(71, 85, 105, 0.25)";
        ctx.fillRect(bx1, byTop, bx2 - bx1, byBot - byTop);

        ctx.strokeStyle = "#64748b";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(bx1, byTop, bx2 - bx1, byBot - byTop);

        ctx.fillStyle = "#94a3b8";
        ctx.font = "bold 11px sans-serif";
        ctx.fillText("Workpiece Material Block (L = 50 mm, H = 15 mm)", bx1 + 10, byTop + 24);

        // Clamped bottom constraint indicator
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(bx1, byBot);
        ctx.lineTo(bx2, byBot);
        ctx.stroke();
        ctx.fillStyle = "#ef4444";
        ctx.font = "10px sans-serif";
        ctx.fillText("CLAMPED BOUNDARY (y = 0)", bx1 + 10, byBot + 14);
    }

    // 3. Draw Die Profile Curves
    if (overlayMode) {
        const overlayColors = ["#38bdf8", "#f43f5e", "#10b981", "#fbbf24", "#a855f7", "#ec4899", "#3b82f6", "#14b8a6", "#f97316", "#eab308"];
        for (let i = 0; i < 10; i++) {
            const sample = SAMPLES[i];
            ctx.beginPath();
            ctx.strokeStyle = overlayColors[i % overlayColors.length];
            ctx.lineWidth = (i === currentSampleIdx) ? 3.0 : 1.2;
            ctx.globalAlpha = (i === currentSampleIdx) ? 1.0 : 0.45;
            for (let j = 0; j < sample.x.length; j++) {
                const sx = toScreenX(sample.x[j]);
                const sy = toScreenY(sample.y[j]);
                if (j === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
            }
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;
    } else {
        ctx.beginPath();
        ctx.moveTo(toScreenX(s.x[0]), toScreenY(s.y[0]));
        for (let j = 1; j < s.x.length; j++) {
            ctx.lineTo(toScreenX(s.x[j]), toScreenY(s.y[j]));
        }
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 3.0;
        ctx.stroke();

        ctx.lineTo(toScreenX(s.x[s.x.length - 1]), toScreenY(27));
        ctx.lineTo(toScreenX(s.x[0]), toScreenY(27));
        ctx.closePath();
        ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
        ctx.fill();

        for (let j = 0; j < s.x.length; j += 5) {
            const sx = toScreenX(s.x[j]);
            const sy = toScreenY(s.y[j]);
            ctx.beginPath();
            ctx.arc(sx, sy, 3, 0, Math.PI * 2);
            ctx.fillStyle = "#0284c7";
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1;
            ctx.stroke();
        }

        const midIdx = Math.floor(s.x.length / 2);
        const ax = toScreenX(s.x[midIdx]);
        const ay = toScreenY(s.y[midIdx]) - 30;
        ctx.strokeStyle = "#f59e0b";
        ctx.fillStyle = "#f59e0b";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(ax, ay - 25);
        ctx.lineTo(ax, ay);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ax - 6, ay - 8);
        ctx.lineTo(ax + 6, ay - 8);
        ctx.lineTo(ax, ay + 2);
        ctx.closePath();
        ctx.fill();

        ctx.font = "bold 11px sans-serif";
        ctx.fillText("Velocity v = 3 m/s ↓", ax - 50, ay - 32);
    }
}

select.addEventListener("change", (e) => {
    currentSampleIdx = parseInt(e.target.value);
    slider.value = currentSampleIdx;
    render();
});
slider.addEventListener("input", (e) => {
    currentSampleIdx = parseInt(e.target.value);
    select.value = currentSampleIdx;
    render();
});

document.getElementById("btnSingle").addEventListener("click", () => {
    overlayMode = false;
    document.getElementById("btnSingle").classList.add("active");
    document.getElementById("btnOverlay").classList.remove("active");
    render();
});
document.getElementById("btnOverlay").addEventListener("click", () => {
    overlayMode = true;
    document.getElementById("btnOverlay").classList.add("active");
    document.getElementById("btnSingle").classList.remove("active");
    render();
});
document.getElementById("btnShowBlock").addEventListener("click", () => {
    showBlock = true;
    document.getElementById("btnShowBlock").classList.add("active");
    document.getElementById("btnHideBlock").classList.remove("active");
    render();
});
document.getElementById("btnHideBlock").addEventListener("click", () => {
    showBlock = false;
    document.getElementById("btnHideBlock").classList.add("active");
    document.getElementById("btnShowBlock").classList.remove("active");
    render();
});

canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    const padL = 60, padR = 30, padT = 40, padB = 60;
    const minX = -52, maxX = 2;
    const minY = -2, maxY = 28;
    const plotW = canvas.width - padL - padR;
    const plotH = canvas.height - padT - padB;

    if (mx >= padL && mx <= canvas.width - padR && my >= padT && my <= canvas.height - padB) {
        const xVal = minX + ((mx - padL) / plotW) * (maxX - minX);
        const yVal = minY + (1 - (my - padT) / plotH) * (maxY - minY);
        tooltip.style.display = "block";
        tooltip.style.left = `${e.clientX - rect.left + 15}px`;
        tooltip.style.top = `${e.clientY - rect.top + 15}px`;
        tooltip.innerHTML = `<strong>Position:</strong> x = ${xVal.toFixed(2)} mm<br><strong>Elevation:</strong> y = ${yVal.toFixed(2)} mm`;
    } else {
        tooltip.style.display = "none";
    }
});
canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });

render();
</script>

</body>
</html>
"""

final_input_html = input_template.replace("__INPUT_JSON__", input_json_str)
with open(out_dir / "interactive_input.html", "w", encoding="utf-8") as f:
    f.write(final_input_html)

print("Saved interactive_input.html!")

# -------------------------------------------------------------
# 2. BUILD INTERACTIVE OUTPUT HTML
# -------------------------------------------------------------
chosen_sample_ids = [0, 1, 2, 5]
trajectory_samples = []

for sid in chosen_sample_ids:
    s_coords = coords[sid]  # (101, 31, 20, 2)
    s_disp = displacements[sid]  # (101, 31, 20, 2)
    
    timesteps_data = []
    for t in range(20):
        x = [round(float(v), 2) for v in s_coords[:, :, t, 0].flatten()]
        y = [round(float(v), 2) for v in s_coords[:, :, t, 1].flatten()]
        ux = [round(float(v), 3) for v in s_disp[:, :, t, 0].flatten()]
        uy = [round(float(v), 3) for v in s_disp[:, :, t, 1].flatten()]
        mag = [round(float(np.sqrt(u**2 + v**2)), 3) for u, v in zip(ux, uy)]
        
        timesteps_data.append({
            "t_step": t,
            "t_norm": round(t / 19.0, 2),
            "x": x,
            "y": y,
            "ux": ux,
            "uy": uy,
            "mag": mag,
            "max_mag": round(float(np.max(mag)), 3),
            "min_uy": round(float(np.min(uy)), 3),
        })
        
    trajectory_samples.append({
        "sample_id": sid,
        "nx": 101,
        "ny": 31,
        "num_steps": 20,
        "timesteps": timesteps_data,
        "die_x": [round(float(v), 2) for v in die_boundary[sid, :, 0]],
        "die_y": [round(float(v), 2) for v in die_boundary[sid, :, 1]],
        "max_indent": round(float(np.abs(np.min(s_disp[:, :, 19, 1]))), 3),
    })

output_json_str = json.dumps(trajectory_samples)

output_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Output Parameter Explorer - 2D Plasticity Benchmark</title>
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #161f30;
            --bg-card: #223048;
            --accent: #38bdf8;
            --accent-warm: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 24px;
            line-height: 1.5;
        }
        .header {
            max-width: 1300px;
            margin: 0 auto 20px auto;
            border-bottom: 1px solid var(--border);
            padding-bottom: 14px;
        }
        .header h1 { font-size: 24px; color: var(--accent); margin-bottom: 6px; }
        .header p { color: var(--text-secondary); font-size: 14px; }
        
        .container {
            max-width: 1300px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 24px;
        }
        @media (max-width: 950px) {
            .container { grid-template-columns: 1fr; }
        }
        
        .sidebar {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
            height: fit-content;
        }
        .control-group { margin-bottom: 18px; }
        .control-group label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        select, input[type="range"] {
            width: 100%;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            outline: none;
        }
        
        .btn-group {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .btn {
            flex: 1 1 calc(50% - 6px);
            padding: 8px 10px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 12.5px;
            text-align: center;
            transition: all 0.2s;
        }
        .btn:hover { background: #2d3e5c; }
        .btn.active {
            background: var(--accent);
            color: #000;
            font-weight: bold;
            border-color: var(--accent);
        }

        .anim-controls {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        .anim-btn {
            flex: 1;
            padding: 10px;
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            font-size: 13px;
        }
        .anim-btn:hover { background: #1d4ed8; }
        .anim-btn.pause { background: #ef4444; }

        .stats-card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 14px;
            margin-top: 16px;
            font-size: 13px;
        }
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .stats-row:last-child { border: none; margin: 0; padding: 0; }
        .stats-label { color: var(--text-secondary); }
        .stats-value { font-weight: bold; color: var(--accent); }

        .main-panel {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid var(--border);
        }
        
        .canvas-card {
            background: #060911;
            border-radius: 8px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            margin-bottom: 16px;
        }
        canvas { display: block; width: 100%; height: 500px; cursor: crosshair; }

        .legend-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card);
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 13px;
            margin-bottom: 18px;
        }
        .gradient-box {
            flex: 1;
            height: 14px;
            margin: 0 16px;
            border-radius: 3px;
            background: linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725);
        }

        .tooltip {
            position: absolute;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid var(--accent);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            color: #fff;
            box-shadow: 0 4px 14px rgba(0,0,0,0.5);
            z-index: 10;
        }
        
        .tensor-explainer {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 18px;
            font-size: 13.5px;
            border-left: 4px solid var(--accent);
        }
        .tensor-explainer h3 { font-size: 15px; margin-bottom: 6px; color: var(--accent); }
        .tensor-explainer code { background: #0b0f19; padding: 2px 6px; border-radius: 4px; color: #38bdf8; }
        .tensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }
        .tensor-cell {
            background: rgba(11, 15, 25, 0.6);
            padding: 10px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.06);
        }
        .tensor-cell strong { color: var(--accent); display: block; font-size: 12.5px; }
        
        /* Physical Parameter Table Styles */
        .param-table-card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            border-left: 4px solid var(--accent);
        }
        .param-table-card h3 {
            font-size: 15px;
            margin-bottom: 6px;
            color: var(--accent);
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }
        .provenance-badge {
            display: inline-block;
            font-size: 11px;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            padding: 3px 8px;
            border-radius: 4px;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .source-note {
            font-size: 12.5px;
            color: var(--text-secondary);
            margin-bottom: 14px;
            line-height: 1.5;
        }
        .source-note strong {
            color: #f59e0b;
        }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 8px;
        }
        .param-table th, .param-table td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .param-table th {
            background: rgba(11, 15, 25, 0.5);
            color: var(--text-secondary);
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .param-table tr:hover td {
            background: rgba(255, 255, 255, 0.03);
        }
        .param-table td.param-name {
            font-weight: 600;
            color: var(--text-primary);
        }
        .param-table td.param-val {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: var(--accent);
            font-weight: 500;
        }
        .param-table td.param-role {
            color: var(--text-secondary);
        }
    </style>
</head>
<body>

<div class="header">
    <h1>🔬 2D Plasticity Benchmark: Output Parameter Explorer</h1>
    <p>Explore the 5D output tensor <code>(987, 101, 31, 20, 4)</code>: spatiotemporal deformed mesh grids and displacement fields.</p>
</div>

<div class="container">
    <div class="sidebar">
        <div class="control-group">
            <label for="sampleSelect">Select Sample</label>
            <select id="sampleSelect">
                <option value="0">Simulation Sample #0</option>
                <option value="1">Simulation Sample #1</option>
                <option value="2">Simulation Sample #2</option>
                <option value="3">Simulation Sample #5</option>
            </select>
        </div>

        <div class="control-group">
            <label>Temporal Rollout: Step t = <span id="lblStep">0</span> / 19 (<span id="lblTNorm">t* = 0.00</span>)</label>
            <input type="range" id="timeSlider" min="0" max="19" value="0">
            <div class="anim-controls">
                <button class="anim-btn" id="btnPlay">▶ Play Rollout</button>
                <button class="btn" id="btnReset">Reset (t=0)</button>
            </div>
        </div>

        <div class="control-group">
            <label>Physical Channel to Display</label>
            <div class="btn-group">
                <button class="btn active" data-channel="mag">Total Displ. |u|</button>
                <button class="btn" data-channel="uy">Vertical u_y (Indentation)</button>
                <button class="btn" data-channel="ux">Horizontal u_x (Flow)</button>
                <button class="btn" data-channel="mesh">Mesh Wireframe (X, Y)</button>
            </div>
        </div>

        <div class="control-group">
            <label>Overlay Options</label>
            <div class="btn-group">
                <button class="btn active" id="toggleWire">Wireframe Grid</button>
                <button class="btn active" id="toggleVectors">Flow Arrows</button>
                <button class="btn active" id="toggleDie">Die Contour (Mold)</button>
            </div>
        </div>

        <div class="stats-card">
            <div class="stats-row">
                <span class="stats-label">Time Step:</span>
                <span class="stats-value" id="statStep">0 / 19</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Max Displ. |u|:</span>
                <span class="stats-value" id="statMaxMag">0.000 mm</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Max Indentation (uy):</span>
                <span class="stats-value" id="statMinUy">0.000 mm</span>
            </div>
            <div class="stats-row">
                <span class="stats-label">Mesh Nodes:</span>
                <span class="stats-value">101 × 31 = 3,131</span>
            </div>
        </div>
    </div>

    <div class="main-panel">
        <div class="legend-bar">
            <span id="cbarMin">0.00 mm</span>
            <div class="gradient-box" id="cbarGradient"></div>
            <span id="cbarMax">6.00 mm</span>
            <span style="margin-left: 14px; font-weight: bold; color: var(--accent);" id="cbarLabel">|u| [mm]</span>
        </div>

        <div class="canvas-card">
            <canvas id="viewCanvas" width="900" height="500"></canvas>
            <div class="tooltip" id="tooltip"></div>
        </div>

        <div class="tensor-explainer">
            <h3>📊 Understanding the 5D Output Tensor Structure</h3>
            <p>The raw simulation solution is stored as a 5-dimensional array: <code>output[sample, x_node, y_node, time, channel]</code>.</p>
            <div class="tensor-grid">
                <div class="tensor-cell">
                    <strong>Dim 0: Sample (987)</strong>
                    Independent simulation runs, each with a different die geometry.
                </div>
                <div class="tensor-cell">
                    <strong>Dim 1 & 2: Mesh (101 x 31)</strong>
                    3,131 spatial quadrilateral nodes across the 50x15 mm domain.
                </div>
                <div class="tensor-cell">
                    <strong>Dim 3: Time (20)</strong>
                    Temporal rollout steps tracking elastoplastic deformation.
                </div>
                <div class="tensor-cell">
                    <strong>Dim 4: Channels (4)</strong>
                    Ch 0: Deformed X<br>
                    Ch 1: Deformed Y<br>
                    Ch 2: Horizontal Disp u_x<br>
                    Ch 3: Vertical Disp u_y
                </div>
            </div>
        </div>

        <div class="param-table-card">
            <h3>
                <span>⚙️ Underlying Simulation & Physical Parameters</span>
                <span class="provenance-badge">Source: Geo-FNO Paper (Not in .mat)</span>
            </h3>
            <p class="source-note">
                <strong>Important Provenance Note:</strong> These constitutive parameters and simulation boundary conditions are defined in the original benchmark paper (<em>"Fourier Neural Operator with Learned Deformations for PDEs on General Geometries"</em>, Li et al., NeurIPS 2022 / JMLR 2023, arXiv:2207.05209). They are <strong>not</strong> stored as header metadata inside the raw <code>plas_N987_T20.mat</code> file itself (the <code>.mat</code> file stores only the resulting numerical tensors <code>input</code> and <code>output</code>).
            </p>
            <div style="overflow-x: auto;">
                <table class="param-table">
                    <thead>
                        <tr>
                            <th>Physical Parameter</th>
                            <th>Value</th>
                            <th>Role in Simulation</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="param-name">Material Constitutive Law</td>
                            <td class="param-val">J₂ Elasto-plastic (von Mises)</td>
                            <td class="param-role">Isotropic hardening plasticity</td>
                        </tr>
                        <tr>
                            <td class="param-name">Young's Modulus (E)</td>
                            <td class="param-val">200 GPa</td>
                            <td class="param-role">Elastic stiffness (steel-like)</td>
                        </tr>
                        <tr>
                            <td class="param-name">Initial Yield Stress (σ_Y)</td>
                            <td class="param-val">70 MPa</td>
                            <td class="param-role">Threshold where permanent plastic deformation begins</td>
                        </tr>
                        <tr>
                            <td class="param-name">Poisson's Ratio (ν)</td>
                            <td class="param-val">0.3</td>
                            <td class="param-role">Transverse strain contraction ratio</td>
                        </tr>
                        <tr>
                            <td class="param-name">Mass Density (ρ)</td>
                            <td class="param-val">7850 kg/m³</td>
                            <td class="param-role">Structural mass for dynamic equilibrium</td>
                        </tr>
                        <tr>
                            <td class="param-name">Workpiece Geometry</td>
                            <td class="param-val">50.0 mm × 15.0 mm</td>
                            <td class="param-role">Rectangular undeformed domain</td>
                        </tr>
                        <tr>
                            <td class="param-name">FEA Discretization</td>
                            <td class="param-val">3,000 CPS4R elements</td>
                            <td class="param-role">4-node bilinear plane stress/strain with reduced integration</td>
                        </tr>
                        <tr>
                            <td class="param-name">Mesh Dimensions</td>
                            <td class="param-val">100 × 30 elements</td>
                            <td class="param-role">Exactly (100 + 1) × (30 + 1) = 101 × 31 = 3, 131 nodes</td>
                        </tr>
                        <tr>
                            <td class="param-name">Boundary Conditions</td>
                            <td class="param-val">Clamped base (u_y = 0)</td>
                            <td class="param-role">Workpiece rests on a rigid horizontal floor</td>
                        </tr>
                        <tr>
                            <td class="param-name">Die Speed / Indentation</td>
                            <td class="param-val">v = 3.0 m/s downward</td>
                            <td class="param-role">Rigid, frictionless die pressed into the top boundary</td>
                        </tr>
                        <tr>
                            <td class="param-name">Time Discretization</td>
                            <td class="param-val">T = 20 uniform increments</td>
                            <td class="param-role">t ∈ {t₀, t₁, …, t₁₉}</td>
                        </tr>
                        <tr>
                            <td class="param-name">Total Runs Generated</td>
                            <td class="param-val">N = 987 simulations</td>
                            <td class="param-role">900 training, 7 validation, 80 testing</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<script>
const DATA = __OUTPUT_JSON__;

let currentSampleIdx = 0;
let currentTimeStep = 0;
let currentChannel = "mag";
let showWire = true;
let showVectors = true;
let showDie = true;
let isPlaying = false;
let animTimer = null;

const canvas = document.getElementById("viewCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

const sampleSelect = document.getElementById("sampleSelect");
const timeSlider = document.getElementById("timeSlider");
const btnPlay = document.getElementById("btnPlay");
const btnReset = document.getElementById("btnReset");

function colormapViridis(t) {
    const c0 = [68, 1, 84], c1 = [59, 82, 139], c2 = [33, 145, 140], c3 = [94, 201, 98], c4 = [253, 231, 37];
    if (t < 0.25) return interpolateColor(c0, c1, t / 0.25);
    if (t < 0.5) return interpolateColor(c1, c2, (t - 0.25) / 0.25);
    if (t < 0.75) return interpolateColor(c2, c3, (t - 0.5) / 0.25);
    return interpolateColor(c3, c4, (t - 0.75) / 0.25);
}

function colormapCoolwarm(t) {
    const cBlue = [59, 76, 192], cMid = [221, 221, 221], cRed = [180, 4, 38];
    if (t < 0.5) return interpolateColor(cBlue, cMid, t / 0.5);
    return interpolateColor(cMid, cRed, (t - 0.5) / 0.5);
}

function interpolateColor(c1, c2, f) {
    f = Math.max(0, Math.min(1, f));
    const r = Math.round(c1[0] + (c2[0] - c1[0]) * f);
    const g = Math.round(c1[1] + (c2[1] - c1[1]) * f);
    const b = Math.round(c1[2] + (c2[2] - c1[2]) * f);
    return `rgb(${r},${g},${b})`;
}

function render() {
    const sample = DATA[currentSampleIdx];
    const stepData = sample.timesteps[currentTimeStep];

    document.getElementById("lblStep").textContent = currentTimeStep;
    document.getElementById("lblTNorm").textContent = `t* = ${stepData.t_norm.toFixed(2)}`;
    document.getElementById("statStep").textContent = `${currentTimeStep} / 19`;
    document.getElementById("statMaxMag").textContent = `${stepData.max_mag.toFixed(3)} mm`;
    document.getElementById("statMinUy").textContent = `${stepData.min_uy.toFixed(3)} mm`;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 60, padR = 40, padT = 40, padB = 60;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const minX = -52, maxX = 2;
    const minY = -2, maxY = 25;

    function toScreenX(x) { return padL + ((x - minX) / (maxX - minX)) * plotW; }
    function toScreenY(y) { return padT + (1 - (y - minY) / (maxY - minY)) * plotH; }

    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    for (let x = -50; x <= 0; x += 10) {
        const sx = toScreenX(x);
        ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, h - padB); ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = "11px sans-serif";
        ctx.fillText(`${x}`, sx - 8, h - padB + 16);
    }
    for (let y = 0; y <= 25; y += 5) {
        const sy = toScreenY(y);
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(w - padR, sy); ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.font = "11px sans-serif";
        ctx.fillText(`${y}`, padL - 26, sy + 4);
    }

    ctx.fillStyle = "#94a3b8";
    ctx.font = "12px sans-serif";
    ctx.fillText("Deformed x-coordinate [mm]", w / 2 - 70, h - 18);
    ctx.save();
    ctx.translate(16, h / 2 + 30);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Deformed y-coordinate [mm]", 0, 0);
    ctx.restore();

    const nx = sample.nx;
    const ny = sample.ny;
    const xs = stepData.x;
    const ys = stepData.y;
    const uxs = stepData.ux;
    const uys = stepData.uy;
    const mags = stepData.mag;

    let fieldValues = mags;
    let minVal = 0.0, maxVal = 6.0;
    let colorFn = colormapViridis;

    if (currentChannel === "uy") {
        fieldValues = uys;
        minVal = -6.0; maxVal = 0.5;
        colorFn = colormapCoolwarm;
        document.getElementById("cbarMin").textContent = "-6.0 mm";
        document.getElementById("cbarMax").textContent = "0.0 mm";
        document.getElementById("cbarLabel").textContent = "Vertical Disp. uy [mm]";
        document.getElementById("cbarGradient").style.background = "linear-gradient(to right, #3b4cc0, #dddddd, #b40426)";
    } else if (currentChannel === "ux") {
        fieldValues = uxs;
        minVal = -2.0; maxVal = 2.0;
        colorFn = colormapCoolwarm;
        document.getElementById("cbarMin").textContent = "-2.0 mm";
        document.getElementById("cbarMax").textContent = "+2.0 mm";
        document.getElementById("cbarLabel").textContent = "Horizontal Disp. ux [mm]";
        document.getElementById("cbarGradient").style.background = "linear-gradient(to right, #3b4cc0, #dddddd, #b40426)";
    } else {
        fieldValues = mags;
        minVal = 0.0; maxVal = 6.0;
        colorFn = colormapViridis;
        document.getElementById("cbarMin").textContent = "0.00 mm";
        document.getElementById("cbarMax").textContent = "6.00 mm";
        document.getElementById("cbarLabel").textContent = "Total |u| [mm]";
        document.getElementById("cbarGradient").style.background = "linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)";
    }

    for (let i = 0; i < nx - 1; i++) {
        for (let j = 0; j < ny - 1; j++) {
            const idx00 = i * ny + j;
            const idx10 = (i + 1) * ny + j;
            const idx11 = (i + 1) * ny + (j + 1);
            const idx01 = i * ny + (j + 1);

            const p00x = toScreenX(xs[idx00]), p00y = toScreenY(ys[idx00]);
            const p10x = toScreenX(xs[idx10]), p10y = toScreenY(ys[idx10]);
            const p11x = toScreenX(xs[idx11]), p11y = toScreenY(ys[idx11]);
            const p01x = toScreenX(xs[idx01]), p01y = toScreenY(ys[idx01]);

            if (currentChannel !== "mesh") {
                const avgVal = (fieldValues[idx00] + fieldValues[idx10] + fieldValues[idx11] + fieldValues[idx01]) / 4;
                const normVal = (avgVal - minVal) / (maxVal - minVal);
                ctx.fillStyle = colorFn(normVal);

                ctx.beginPath();
                ctx.moveTo(p00x, p00y);
                ctx.lineTo(p10x, p10y);
                ctx.lineTo(p11x, p11y);
                ctx.lineTo(p01x, p01y);
                ctx.closePath();
                ctx.fill();
            }

            if (showWire || currentChannel === "mesh") {
                if (i % 2 === 0 && j % 2 === 0) {
                    ctx.strokeStyle = (currentChannel === "mesh") ? "#38bdf8" : "rgba(0,0,0,0.25)";
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    const botIdx0 = 0 * ny + (ny - 1);
    const botIdxN = (nx - 1) * ny + (ny - 1);
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(toScreenX(xs[botIdx0]), toScreenY(ys[botIdx0]));
    ctx.lineTo(toScreenX(xs[botIdxN]), toScreenY(ys[botIdxN]));
    ctx.stroke();

    if (showVectors && currentChannel !== "mesh") {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
        ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
        ctx.lineWidth = 1.2;

        const strideX = 6;
        const strideY = 4;
        for (let i = 2; i < nx - 2; i += strideX) {
            for (let j = 2; j < ny - 2; j += strideY) {
                const idx = i * ny + j;
                const px = toScreenX(xs[idx]);
                const py = toScreenY(ys[idx]);
                const ux = uxs[idx];
                const uy = uys[idx];
                const mag = mags[idx];

                if (mag > 0.05) {
                    const scale = 14;
                    const tipX = px + ux * scale;
                    const tipY = py - uy * scale;

                    ctx.beginPath();
                    ctx.moveTo(px, py);
                    ctx.lineTo(tipX, tipY);
                    ctx.stroke();

                    const angle = Math.atan2(tipY - py, tipX - px);
                    ctx.beginPath();
                    ctx.moveTo(tipX, tipY);
                    ctx.lineTo(tipX - 5 * Math.cos(angle - Math.PI / 6), tipY - 5 * Math.sin(angle - Math.PI / 6));
                    ctx.lineTo(tipX - 5 * Math.cos(angle + Math.PI / 6), tipY - 5 * Math.sin(angle + Math.PI / 6));
                    ctx.closePath();
                    ctx.fill();
                }
            }
        }
    }

    // 5. Draw Rigid Die Contour (Mold)
    if (showDie && sample.die_x && sample.die_y) {
        const indentOffset = (currentTimeStep / 19.0) * (sample.max_indent || 5.7);
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 3.0;
        ctx.beginPath();
        for (let j = 0; j < sample.die_x.length; j++) {
            const dx = toScreenX(sample.die_x[j]);
            const dy = toScreenY(sample.die_y[j] - indentOffset);
            if (j === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy);
        }
        ctx.stroke();

        // Shaded die body above contour
        ctx.lineTo(toScreenX(sample.die_x[sample.die_x.length - 1]), toScreenY(25));
        ctx.lineTo(toScreenX(sample.die_x[0]), toScreenY(25));
        ctx.closePath();
        ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
        ctx.fill();

        // Tool label
        ctx.fillStyle = "#f59e0b";
        ctx.font = "bold 11px sans-serif";
        const midIdx = Math.floor(sample.die_x.length / 2);
        const lx = toScreenX(sample.die_x[midIdx]) - 55;
        const ly = toScreenY(sample.die_y[midIdx] - indentOffset) - 10;
        ctx.fillText("Rigid Die S_d(x) (Mold) ↓", lx, ly);
    }
}

function playRollout() {
    if (!isPlaying) {
        isPlaying = true;
        btnPlay.textContent = "⏸ Pause";
        btnPlay.classList.add("pause");
        animTimer = setInterval(() => {
            currentTimeStep = (currentTimeStep + 1) % 20;
            timeSlider.value = currentTimeStep;
            render();
        }, 250);
    } else {
        isPlaying = false;
        btnPlay.textContent = "▶ Play Rollout";
        btnPlay.classList.remove("pause");
        clearInterval(animTimer);
    }
}

btnPlay.addEventListener("click", playRollout);
btnReset.addEventListener("click", () => {
    if (isPlaying) playRollout();
    currentTimeStep = 0;
    timeSlider.value = 0;
    render();
});

timeSlider.addEventListener("input", (e) => {
    if (isPlaying) playRollout();
    currentTimeStep = parseInt(e.target.value);
    render();
});

sampleSelect.addEventListener("change", (e) => {
    currentSampleIdx = parseInt(e.target.value);
    render();
});

document.querySelectorAll("[data-channel]").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll("[data-channel]").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentChannel = btn.getAttribute("data-channel");
        render();
    });
});

document.getElementById("toggleWire").addEventListener("click", function() {
    showWire = !showWire;
    this.classList.toggle("active", showWire);
    render();
});
document.getElementById("toggleVectors").addEventListener("click", function() {
    showVectors = !showVectors;
    this.classList.toggle("active", showVectors);
    render();
});
document.getElementById("toggleDie").addEventListener("click", function() {
    showDie = !showDie;
    this.classList.toggle("active", showDie);
    render();
});

canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    const padL = 60, padR = 40, padT = 40, padB = 60;
    const minX = -52, maxX = 2;
    const minY = -2, maxY = 25;
    const plotW = canvas.width - padL - padR;
    const plotH = canvas.height - padT - padB;

    if (mx >= padL && mx <= canvas.width - padR && my >= padT && my <= canvas.height - padB) {
        const sample = DATA[currentSampleIdx];
        const step = sample.timesteps[currentTimeStep];
        const nx = sample.nx, ny = sample.ny;

        let bestDist = Infinity;
        let bestIdx = -1;
        for (let i = 0; i < nx; i += 2) {
            for (let j = 0; j < ny; j += 2) {
                const idx = i * ny + j;
                const sx = padL + ((step.x[idx] - minX) / (maxX - minX)) * plotW;
                const sy = padT + (1 - (step.y[idx] - minY) / (maxY - minY)) * plotH;
                const d = Math.hypot(mx - sx, my - sy);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = idx;
                }
            }
        }

        if (bestDist < 25) {
            tooltip.style.display = "block";
            tooltip.style.left = `${e.clientX - rect.left + 15}px`;
            tooltip.style.top = `${e.clientY - rect.top + 15}px`;
            tooltip.innerHTML = `<strong>Node Coordinates:</strong> (${step.x[bestIdx].toFixed(2)}, ${step.y[bestIdx].toFixed(2)}) mm<br>` +
                                `<strong>Total |u|:</strong> ${step.mag[bestIdx].toFixed(3)} mm<br>` +
                                `<strong>Horizontal ux:</strong> ${step.ux[bestIdx].toFixed(3)} mm<br>` +
                                `<strong>Vertical uy:</strong> ${step.uy[bestIdx].toFixed(3)} mm`;
        } else {
            tooltip.style.display = "none";
        }
    } else {
        tooltip.style.display = "none";
    }
});
canvas.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });

render();
</script>

</body>
</html>
"""

final_output_html = output_template.replace("__OUTPUT_JSON__", output_json_str)
with open(out_dir / "interactive_output.html", "w", encoding="utf-8") as f:
    f.write(final_output_html)

print("Saved interactive_output.html!")
