"""Script to generate a 3-slide interactive dataset presentation deck:
Slide 1: Dataset Architecture, Tensor Dimensions, Relations, and Physical Parameters
Slide 2: Input Parameter Explorer: Rigid Die Profile S_d(x)
Slide 3: Output Solution Explorer: 5D Spatiotemporal Rollout
"""

import json
from pathlib import Path
import numpy as np

# Load processed data
data_dir = Path("data/processed/ds_plasticity_nochange")
train = np.load(data_dir / "train.npz")

die_boundary = train["die_boundary"]   # (900, 101, 2)
coords = train["coords"]               # (900, 101, 31, 20, 2)
displacements = train["displacements"] # (900, 101, 31, 20, 2)

out_dir = Path("results/DatasetAnalysis/ds_plasticity_nochange")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Prepare 50 input samples
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

# 2. Prepare 4 detailed trajectory rollout samples
chosen_sample_ids = [0, 1, 2, 5]
trajectory_samples = []

for sid in chosen_sample_ids:
    s_coords = coords[sid]        # (101, 31, 20, 2)
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

presentation_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>2D Plasticity Benchmark: Dataset Analysis Deck</title>
    <style>
        :root {
            --bg-primary: #0a0f1d;
            --bg-secondary: #141c2e;
            --bg-card: #1f2b42;
            --accent: #38bdf8;
            --accent-warm: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border: #334155;
            --success: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        
        /* Navigation Bar */
        .slide-nav-bar {
            background: rgba(20, 28, 46, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
            padding: 12px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        .nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .nav-brand h2 {
            font-size: 16px;
            font-weight: 700;
            color: var(--accent);
            letter-spacing: 0.5px;
        }
        .nav-brand span.badge {
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent);
            border: 1px solid rgba(56, 189, 248, 0.3);
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        
        .slide-pills {
            display: flex;
            gap: 8px;
        }
        .slide-pill {
            background: var(--bg-card);
            color: var(--text-secondary);
            border: 1px solid var(--border);
            padding: 7px 16px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .slide-pill:hover {
            background: #2a3b5c;
            color: #fff;
        }
        .slide-pill.active {
            background: var(--accent);
            color: #000;
            border-color: var(--accent);
            font-weight: 700;
        }
        
        .nav-controls {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .nav-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 7px 15px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
        }
        .nav-btn:hover { background: #2a3b5c; }
        .nav-btn.primary {
            background: #2563eb;
            border-color: #3b82f6;
            color: #fff;
        }
        .nav-btn.primary:hover { background: #1d4ed8; }

        /* Progress Bar */
        .progress-bar-container {
            width: 100%;
            height: 3px;
            background: rgba(255, 255, 255, 0.05);
            position: sticky;
            top: 57px;
            z-index: 999;
        }
        .progress-bar {
            height: 100%;
            background: var(--accent);
            width: 33.33%;
            transition: width 0.3s ease;
        }

        /* Slide Viewport */
        .slide-container {
            flex: 1;
            padding: 28px;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }
        .slide {
            display: none;
            animation: fadeIn 0.25s ease forwards;
        }
        .slide.active { display: block; }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Slide Header */
        .slide-header {
            margin-bottom: 22px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 14px;
        }
        .slide-header h1 {
            font-size: 25px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 6px;
        }
        .slide-header h1 span.highlight { color: var(--accent); }
        .slide-header p {
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
        }

        /* Slide 1 Layout */
        .grid-2col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 22px;
            margin-bottom: 22px;
        }
        @media (max-width: 950px) {
            .grid-2col, .split-layout { grid-template-columns: 1fr !important; }
        }
        .info-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 22px;
        }
        .info-card h3 {
            font-size: 16px;
            color: var(--accent);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .info-card p {
            color: var(--text-secondary);
            font-size: 13.5px;
            line-height: 1.6;
            margin-bottom: 12px;
        }

        .tensor-box {
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .tensor-box-title {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        .tensor-shape-tag {
            font-family: ui-monospace, monospace;
            font-size: 16px;
            font-weight: 700;
            color: var(--accent);
            display: inline-block;
            margin-bottom: 4px;
        }
        .tensor-desc {
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.5;
        }

        /* Stats List */
        .stats-card {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 14px;
            font-size: 13px;
        }
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 7px;
            padding-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .stats-row:last-child { border: none; margin: 0; padding: 0; }
        .stats-label { color: var(--text-secondary); }
        .stats-value { font-weight: bold; color: var(--accent); }

        /* Physical Table */
        .param-table-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 22px;
            border-left: 4px solid var(--accent);
        }
        .param-table-card h3 {
            font-size: 16px;
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
        .source-note strong { color: #f59e0b; }
        .param-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 8px;
        }
        .param-table th, .param-table td {
            padding: 9px 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .param-table th {
            background: rgba(11, 15, 25, 0.6);
            color: var(--text-secondary);
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .param-table tr:hover td { background: rgba(255, 255, 255, 0.03); }
        .param-table td.param-name { font-weight: 600; color: var(--text-primary); }
        .param-table td.param-val {
            font-family: ui-monospace, monospace;
            color: var(--accent);
            font-weight: 500;
        }
        .param-table td.param-role { color: var(--text-secondary); }

        /* Tensor Structure Cards (Slide 2 & 3) */
        .tensor-explainer {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 18px;
            border-left: 4px solid var(--accent);
        }
        .tensor-explainer h3 {
            font-size: 15px;
            margin-bottom: 6px;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .tensor-explainer p {
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            line-height: 1.5;
        }
        .tensor-explainer code {
            background: #0b0f19;
            padding: 2px 7px;
            border-radius: 4px;
            color: #38bdf8;
            font-family: monospace;
            font-size: 12.5px;
        }
        .tensor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }
        .tensor-cell {
            background: rgba(11, 15, 25, 0.6);
            padding: 10px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 12.5px;
            line-height: 1.45;
            color: var(--text-secondary);
        }
        .tensor-cell strong {
            color: var(--accent);
            display: block;
            font-size: 13px;
            margin-bottom: 4px;
        }

        /* Interactive Slides Layout (Slide 2 & 3) */
        .split-layout {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 22px;
        }
        .sidebar {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid var(--border);
            height: fit-content;
        }
        .main-card {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 22px;
            border: 1px solid var(--border);
        }
        .canvas-container {
            background: #060911;
            border-radius: 8px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
            margin-bottom: 14px;
        }
        canvas { display: block; width: 100%; cursor: crosshair; }

        /* Controls */
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
            font-size: 13.5px;
            outline: none;
        }
        .btn-group { display: flex; gap: 6px; flex-wrap: wrap; }
        .btn {
            flex: 1 1 calc(50% - 6px);
            padding: 8px 10px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
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
        .anim-controls { display: flex; gap: 8px; margin-top: 10px; }
        .anim-btn {
            flex: 1;
            padding: 9px;
            background: #2563eb;
            color: #fff;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            font-size: 12.5px;
        }
        .anim-btn:hover { background: #1d4ed8; }
        .anim-btn.pause { background: #ef4444; }

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

        .legend-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-card);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 12.5px;
            margin-bottom: 14px;
        }
        .gradient-box {
            flex: 1;
            height: 12px;
            margin: 0 16px;
            border-radius: 3px;
            background: linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725);
        }

        /* Footer */
        .footer-helper {
            background: rgba(15, 23, 42, 0.85);
            border-top: 1px solid var(--border);
            padding: 8px 28px;
            font-size: 12px;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: auto;
        }
        .footer-helper kbd {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: var(--accent);
        }
    </style>
</head>
<body>

<!-- Navigation Bar -->
<div class="slide-nav-bar">
    <div class="nav-brand">
        <h2>🔬 2D Plasticity Benchmark</h2>
        <span class="badge">Dataset Analysis</span>
    </div>

    <div class="slide-pills">
        <button class="slide-pill active" data-slide="0"><span>1</span> Dataset Architecture & Specs</button>
        <button class="slide-pill" data-slide="1"><span>2</span> Input Parameter (Die S_d)</button>
        <button class="slide-pill" data-slide="2"><span>3</span> Output Rollout (Mesh & Fields)</button>
    </div>

    <div class="nav-controls">
        <button class="nav-btn" id="btnPrev">◀ Previous</button>
        <button class="nav-btn primary" id="btnNext">Next ▶</button>
        <button class="nav-btn" id="btnFullscreen">⛶ Fullscreen</button>
    </div>
</div>

<div class="progress-bar-container">
    <div class="progress-bar" id="progressBar"></div>
</div>

<!-- Slide Container -->
<div class="slide-container">

    <!-- ==================== SLIDE 1: DATASET ARCHITECTURE, TENSORS & SPECS ==================== -->
    <div class="slide active" id="slide-0">
        <div class="slide-header">
            <h1>Dataset Architecture, <span class="highlight">Tensor Dimensions & Physical Specifications</span></h1>
            <p>Comprehensive breakdown of the 2D Plasticity benchmark dataset (<code>plas_N987_T20.mat</code>), including tensor shapes, input/output relationships, and underlying finite element parameters.</p>
        </div>

        <!-- 2 Columns: Tensors & Relations on left, Partitions & Scale on right -->
        <div class="grid-2col">
            <div class="info-card">
                <h3>📐 Tensor Dimensions & Input-Output Mapping</h3>
                
                <div class="tensor-box">
                    <div class="tensor-box-title">Input Tensor Dimension</div>
                    <div class="tensor-shape-tag">input : (987, 101)</div>
                    <div class="tensor-desc">
                        987 independent simulation runs × 101 discrete horizontal coordinates. Each row defines the 1D elevation profile $S_d(x)$ of the impacting rigid die across the top surface of the workpiece.
                    </div>
                </div>

                <div class="tensor-box">
                    <div class="tensor-box-title">Output Tensor Dimension</div>
                    <div class="tensor-shape-tag">output : (987, 101, 31, 20, 4)</div>
                    <div class="tensor-desc">
                        987 runs × $101$ x-nodes × $31$ y-nodes × $20$ temporal increments × $4$ solution channels:
                        <br>• Ch 0: Deformed coordinate $X$ [mm]
                        <br>• Ch 1: Deformed coordinate $Y$ [mm]
                        <br>• Ch 2: Horizontal displacement $u_x$ [mm] (material flow)
                        <br>• Ch 3: Vertical displacement $u_y &le; 0$ [mm] (indentation depth)
                    </div>
                </div>

                <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.5; background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.15); padding: 10px; border-radius: 6px;">
                    🔗 <strong>How They Are Related:</strong> The only independent variable varied across simulations is the rigid die shape $S_d(x)$. When pressed vertically downward into the standardized flat block at $v = 3\text{ m/s}$, the tool geometry directly dictates the spatiotemporal elastoplastic displacement field $\mathbf{u}(x, y, t)$ across all 3,131 nodes.
                </div>
            </div>

            <div class="info-card">
                <h3>📦 Dataset Architecture & Scale</h3>
                <p>
                    The benchmark dataset consists of <strong>987 high-fidelity ABAQUS simulations</strong> partitioned into clean, isolated partitions:
                </p>
                <div class="stats-card">
                    <div class="stats-row">
                        <span class="stats-label">Total Finite Element Runs:</span>
                        <span class="stats-value">N = 987 simulations</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Training Partition:</span>
                        <span class="stats-value">900 samples (1,373.95 MB, train.npz)</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Validation / Test Partitions:</span>
                        <span class="stats-value">7 val (10.81 MB) / 80 test (122.06 MB)</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Spatial Quadrilateral Discretization:</span>
                        <span class="stats-value">100 × 30 elements = 101 × 31 (3,131 nodes)</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Temporal Sequence Length:</span>
                        <span class="stats-value">T = 20 increments (t* ∈ [0, 1])</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Solution Channels per Node:</span>
                        <span class="stats-value">4 (Deformed X, Y, Disp u_x, Disp u_y)</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Kinematic Invariance Verified:</span>
                        <span class="stats-value">||X₀ - (x(t) - u(t))|| ≤ 1.7 × 10⁻⁵ mm</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Full Width Physical Specifications Table -->
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
                            <td class="param-role">Isotropic hardening plasticity model</td>
                        </tr>
                        <tr>
                            <td class="param-name">Young's Modulus (E)</td>
                            <td class="param-val">200 GPa</td>
                            <td class="param-role">Elastic stiffness (steel-like material)</td>
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
                            <td class="param-role">Structural mass density for dynamic equilibrium</td>
                        </tr>
                        <tr>
                            <td class="param-name">Workpiece Geometry</td>
                            <td class="param-val">50.0 mm × 15.0 mm</td>
                            <td class="param-role">Rectangular undeformed domain (flat top, height 15 mm)</td>
                        </tr>
                        <tr>
                            <td class="param-name">FEA Discretization</td>
                            <td class="param-val">3,000 CPS4R elements</td>
                            <td class="param-role">4-node bilinear plane strain elements with reduced integration</td>
                        </tr>
                        <tr>
                            <td class="param-name">Mesh Dimensions</td>
                            <td class="param-val">100 × 30 elements</td>
                            <td class="param-role">Exactly (100 + 1) × (30 + 1) = 101 × 31 = 3,131 nodes</td>
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
                            <td class="param-role">Recorded stages t ∈ {t₀, t₁, …, t₁₉}</td>
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

    <!-- ==================== SLIDE 2: INPUT PARAMETER EXPLORER ==================== -->
    <div class="slide" id="slide-1">
        <div class="slide-header">
            <h1>Input Parameter: <span class="highlight">The Rigid Impacting Die Profile S_d(x)</span></h1>
            <p>Explore the 1D parameterized tool elevation curves across 50 simulation runs. The die contour is aligned from $X = -49.65\text{ mm}$ (left) to $+0.35\text{ mm}$ (right).</p>
        </div>

        <div class="split-layout">
            <div class="sidebar">
                <div class="control-group">
                    <label for="inpSelect">Select Simulation Sample</label>
                    <select id="inpSelect"></select>
                </div>
                <div class="control-group">
                    <label>Sample Slider (0 - 49)</label>
                    <input type="range" id="inpSlider" min="0" max="49" value="0">
                </div>
                <div class="control-group">
                    <label>View Mode</label>
                    <div class="btn-group">
                        <button class="btn active" id="btnInpSingle">Single Sample</button>
                        <button class="btn" id="btnInpOverlay">Overlay 10 Runs</button>
                    </div>
                </div>
                <div class="control-group">
                    <label>Workpiece Context</label>
                    <div class="btn-group">
                        <button class="btn active" id="btnInpBlock">Show Block</button>
                        <button class="btn" id="btnInpHide">Die Only</button>
                    </div>
                </div>
                <div class="stats-card">
                    <div class="stats-row">
                        <span class="stats-label">Domain Span X:</span>
                        <span class="stats-value">50.0 mm [-49.65, 0.35]</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Min Elevation:</span>
                        <span class="stats-value" id="inpMinY">14.91 mm</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Max Elevation:</span>
                        <span class="stats-value" id="inpMaxY">23.50 mm</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Height Variation (Δy):</span>
                        <span class="stats-value" id="inpPtpY">8.59 mm</span>
                    </div>
                </div>
            </div>

            <div class="main-card">
                <div class="canvas-container">
                    <canvas id="inpCanvas" width="900" height="440"></canvas>
                    <div class="tooltip" id="inpTooltip"></div>
                </div>
                <div style="font-size: 13.5px; color: var(--text-secondary); line-height: 1.6; margin-bottom: 12px;">
                    💡 <strong>Physical Meaning:</strong> In the Geo-FNO study, the authors generated each tool curve using 8 randomized control points connected by <em>piecewise cubic Hermite splines (PCHIP)</em> across the 50 mm workpiece width. The die is moved vertically downward at $v = 3\text{ m/s}$. The teeth dip down to contact the flat workpiece ($y = 15\text{ mm}$).
                </div>
                <div class="tensor-explainer">
                    <h3>📊 Understanding the 2D Input Tensor Structure</h3>
                    <p>The raw tool geometry is stored as a 2-dimensional array: <code>input[sample, x_node]</code>.</p>
                    <div class="tensor-grid">
                        <div class="tensor-cell">
                            <strong>Dim 0: Sample (987)</strong>
                            Independent simulation runs, each with a unique randomized rigid die geometry.
                        </div>
                        <div class="tensor-cell">
                            <strong>Dim 1: Die Contour X (101)</strong>
                            101 discrete horizontal coordinates spanning the 50.0 mm workpiece width.
                        </div>
                        <div class="tensor-cell">
                            <strong>Elevation: S_d(x) [mm]</strong>
                            1D profile y generated with 8 randomized control points via PCHIP cubic splines.
                        </div>
                        <div class="tensor-cell">
                            <strong>Physical Boundary Role</strong>
                            Rigid kinematic tool pressed downward at v = 3.0 m/s into the flat workpiece top (y = 15 mm).
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- ==================== SLIDE 3: OUTPUT ROLLOUT EXPLORER ==================== -->
    <div class="slide" id="slide-2">
        <div class="slide-header">
            <h1>Output Solution: <span class="highlight">5D Tensor Spatiotemporal Rollout</span></h1>
            <p>Explore the full-field solution tensor <code>(987, 101, 31, 20, 4)</code>: spatiotemporal deformed quadrilateral grids and displacement fields.</p>
        </div>

        <div class="split-layout">
            <div class="sidebar">
                <div class="control-group">
                    <label for="outSelect">Select Sample</label>
                    <select id="outSelect">
                        <option value="0">Simulation Sample #0</option>
                        <option value="1">Simulation Sample #1</option>
                        <option value="2">Simulation Sample #2</option>
                        <option value="3">Simulation Sample #5</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Rollout Step: <span id="outLblStep" style="color:var(--accent);">0</span> / 19 (<span id="outLblTNorm">t* = 0.00</span>)</label>
                    <input type="range" id="outTimeSlider" min="0" max="19" value="0">
                    <div class="anim-controls">
                        <button class="anim-btn" id="outBtnPlay">▶ Play Rollout</button>
                        <button class="btn" id="outBtnReset">Reset (t=0)</button>
                    </div>
                </div>
                <div class="control-group">
                    <label>Solution Field</label>
                    <div class="btn-group">
                        <button class="btn active" data-field="mag">|u| Magnitude</button>
                        <button class="btn" data-field="uy">Vertical u_y</button>
                        <button class="btn" data-field="ux">Horizontal u_x</button>
                        <button class="btn" data-field="mesh">Deformed Mesh</button>
                    </div>
                </div>
                <div class="control-group">
                    <label>Visual Overlays</label>
                    <div class="btn-group">
                        <button class="btn active" id="outToggleWire">Wireframe</button>
                        <button class="btn active" id="outToggleVectors">Flow Arrows</button>
                        <button class="btn active" id="outToggleDie">Die (Mold)</button>
                    </div>
                </div>
                <div class="stats-card">
                    <div class="stats-row">
                        <span class="stats-label">Max Displ. |u|:</span>
                        <span class="stats-value" id="outStatMaxMag">0.000 mm</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Max Indentation (uy):</span>
                        <span class="stats-value" id="outStatMinUy">0.000 mm</span>
                    </div>
                    <div class="stats-row">
                        <span class="stats-label">Mesh Nodes:</span>
                        <span class="stats-value">101 × 31 = 3,131</span>
                    </div>
                </div>
            </div>

            <div class="main-card">
                <div class="legend-bar">
                    <span id="outCbarMin">0.00 mm</span>
                    <div class="gradient-box" id="outCbarGrad"></div>
                    <span id="outCbarMax">6.00 mm</span>
                    <span style="font-weight:bold; color:var(--accent);" id="outCbarLbl">|u| [mm]</span>
                </div>
                <div class="canvas-container">
                    <canvas id="outCanvas" width="900" height="500"></canvas>
                    <div class="tooltip" id="outTooltip"></div>
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
            </div>
        </div>
    </div>

</div>

<!-- Footer Helper -->
<div class="footer-helper">
    <div>
        <span>Keyboard Navigation: <kbd>◀</kbd> / <kbd>▶</kbd> or <kbd>Space</kbd> to change slides • <kbd>F11</kbd> for Fullscreen Presentation</span>
    </div>
    <div>
        <span>Guided Research: 2D Plasticity Benchmark Dataset Analysis</span>
    </div>
</div>

<script>
const SAMPLES = __INPUT_JSON__;
const DATA = __OUTPUT_JSON__;

let currentSlide = 0;
const totalSlides = 3;

function goToSlide(idx) {
    if (idx < 0 || idx >= totalSlides) return;
    currentSlide = idx;
    
    document.querySelectorAll(".slide").forEach((s, i) => {
        s.classList.toggle("active", i === currentSlide);
    });
    document.querySelectorAll(".slide-pill").forEach((p, i) => {
        p.classList.toggle("active", i === currentSlide);
    });
    
    document.getElementById("progressBar").style.width = `${((currentSlide + 1) / totalSlides) * 100}%`;
    document.getElementById("btnPrev").disabled = (currentSlide === 0);
    document.getElementById("btnNext").textContent = (currentSlide === totalSlides - 1) ? "Restart ↺" : "Next ▶";

    if (currentSlide === 1) renderInp();
    if (currentSlide === 2) renderOut();
}

document.querySelectorAll(".slide-pill").forEach(p => {
    p.addEventListener("click", () => goToSlide(parseInt(p.getAttribute("data-slide"))));
});
document.getElementById("btnPrev").addEventListener("click", () => goToSlide(currentSlide - 1));
document.getElementById("btnNext").addEventListener("click", () => {
    if (currentSlide === totalSlides - 1) goToSlide(0); else goToSlide(currentSlide + 1);
});
document.getElementById("btnFullscreen").addEventListener("click", () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
});

window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") {
        if (e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {
            e.preventDefault();
            if (currentSlide < totalSlides - 1) goToSlide(currentSlide + 1);
        }
    } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        if (e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") {
            e.preventDefault();
            if (currentSlide > 0) goToSlide(currentSlide - 1);
        }
    } else if (e.key === "Home") {
        goToSlide(0);
    } else if (e.key === "End") {
        goToSlide(totalSlides - 1);
    }
});

// ==========================================
// COLORMAP HELPERS
// ==========================================
function interpolateColor(c1, c2, f) {
    f = Math.max(0, Math.min(1, f));
    return `rgb(${Math.round(c1[0] + (c2[0] - c1[0]) * f)},${Math.round(c1[1] + (c2[1] - c1[1]) * f)},${Math.round(c1[2] + (c2[2] - c1[2]) * f)})`;
}
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

// ==========================================
// SLIDE 2: INPUT CANVAS LOGIC
// ==========================================
let inpCurrentIdx = 0;
let inpOverlay = false;
let inpShowBlock = true;

const inpCanvas = document.getElementById("inpCanvas");
const inpCtx = inpCanvas.getContext("2d");
const inpSelect = document.getElementById("inpSelect");
const inpSlider = document.getElementById("inpSlider");
const inpTooltip = document.getElementById("inpTooltip");

SAMPLES.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s.id;
    opt.textContent = `Die Sample #${s.id} (min y = ${s.min_y} mm)`;
    inpSelect.appendChild(opt);
});

function renderInp() {
    const s = SAMPLES[inpCurrentIdx];
    document.getElementById("inpMinY").textContent = `${s.min_y.toFixed(2)} mm`;
    document.getElementById("inpMaxY").textContent = `${s.max_y.toFixed(2)} mm`;
    document.getElementById("inpPtpY").textContent = `${s.ptp_y.toFixed(2)} mm`;

    const w = inpCanvas.width, h = inpCanvas.height;
    inpCtx.clearRect(0, 0, w, h);

    const padL = 60, padR = 40, padT = 40, padB = 60;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const minX = -52, maxX = 2, minY = -2, maxY = 28;

    function toScreenX(x) { return padL + ((x - minX) / (maxX - minX)) * plotW; }
    function toScreenY(y) { return padT + (1 - (y - minY) / (maxY - minY)) * plotH; }

    inpCtx.strokeStyle = "#1e293b";
    inpCtx.lineWidth = 1;
    for (let x = -50; x <= 0; x += 10) {
        const sx = toScreenX(x);
        inpCtx.beginPath(); inpCtx.moveTo(sx, padT); inpCtx.lineTo(sx, h - padB); inpCtx.stroke();
        inpCtx.fillStyle = "#64748b"; inpCtx.font = "11px sans-serif";
        inpCtx.fillText(`${x}`, sx - 8, h - padB + 16);
    }
    for (let y = 0; y <= 25; y += 5) {
        const sy = toScreenY(y);
        inpCtx.beginPath(); inpCtx.moveTo(padL, sy); inpCtx.lineTo(w - padR, sy); inpCtx.stroke();
        inpCtx.fillStyle = "#64748b"; inpCtx.font = "11px sans-serif";
        inpCtx.fillText(`${y}`, padL - 26, sy + 4);
    }

    inpCtx.fillStyle = "#94a3b8"; inpCtx.font = "12px sans-serif";
    inpCtx.fillText("Horizontal Position x [mm]", w / 2 - 70, h - 18);
    inpCtx.save(); inpCtx.translate(16, h / 2 + 30); inpCtx.rotate(-Math.PI / 2);
    inpCtx.fillText("Elevation y [mm]", 0, 0); inpCtx.restore();

    if (inpShowBlock) {
        const bx1 = toScreenX(-49.65), bx2 = toScreenX(0.35);
        const byTop = toScreenY(14.9), byBot = toScreenY(-0.1);
        inpCtx.fillStyle = "rgba(56, 189, 248, 0.05)";
        inpCtx.fillRect(bx1, byTop, bx2 - bx1, byBot - byTop);
        inpCtx.strokeStyle = "#475569"; inpCtx.lineWidth = 1.5;
        inpCtx.strokeRect(bx1, byTop, bx2 - bx1, byBot - byTop);
        inpCtx.fillStyle = "#94a3b8"; inpCtx.font = "bold 11px sans-serif";
        inpCtx.fillText("Undeformed Workpiece (50 × 15 mm)", bx1 + 14, byTop + 24);

        inpCtx.strokeStyle = "#ef4444"; inpCtx.lineWidth = 3;
        inpCtx.beginPath(); inpCtx.moveTo(bx1, byBot); inpCtx.lineTo(bx2, byBot); inpCtx.stroke();
        inpCtx.fillStyle = "#ef4444"; inpCtx.font = "10px sans-serif";
        inpCtx.fillText("CLAMPED BASE (y = 0)", bx1 + 10, byBot + 14);
    }

    if (inpOverlay) {
        const colors = ["#38bdf8", "#f43f5e", "#10b981", "#fbbf24", "#a855f7", "#ec4899", "#3b82f6", "#14b8a6", "#f97316", "#eab308"];
        for (let i = 0; i < 10; i++) {
            const smp = SAMPLES[i];
            inpCtx.beginPath();
            inpCtx.strokeStyle = colors[i % colors.length];
            inpCtx.lineWidth = (i === inpCurrentIdx) ? 3.0 : 1.2;
            inpCtx.globalAlpha = (i === inpCurrentIdx) ? 1.0 : 0.45;
            for (let j = 0; j < smp.x.length; j++) {
                const sx = toScreenX(smp.x[j]), sy = toScreenY(smp.y[j]);
                if (j === 0) inpCtx.moveTo(sx, sy); else inpCtx.lineTo(sx, sy);
            }
            inpCtx.stroke();
        }
        inpCtx.globalAlpha = 1.0;
    } else {
        inpCtx.beginPath();
        inpCtx.moveTo(toScreenX(s.x[0]), toScreenY(s.y[0]));
        for (let j = 1; j < s.x.length; j++) inpCtx.lineTo(toScreenX(s.x[j]), toScreenY(s.y[j]));
        inpCtx.strokeStyle = "#38bdf8"; inpCtx.lineWidth = 3.0; inpCtx.stroke();

        inpCtx.lineTo(toScreenX(s.x[s.x.length - 1]), toScreenY(27));
        inpCtx.lineTo(toScreenX(s.x[0]), toScreenY(27));
        inpCtx.closePath();
        inpCtx.fillStyle = "rgba(56, 189, 248, 0.12)"; inpCtx.fill();

        for (let j = 0; j < s.x.length; j += 5) {
            const sx = toScreenX(s.x[j]), sy = toScreenY(s.y[j]);
            inpCtx.beginPath(); inpCtx.arc(sx, sy, 3, 0, Math.PI * 2);
            inpCtx.fillStyle = "#fff"; inpCtx.fill();
            inpCtx.strokeStyle = "#38bdf8"; inpCtx.lineWidth = 1.5; inpCtx.stroke();
        }

        const ax = toScreenX(-25), ay = toScreenY(20);
        inpCtx.strokeStyle = "#f59e0b"; inpCtx.fillStyle = "#f59e0b"; inpCtx.lineWidth = 3;
        inpCtx.beginPath(); inpCtx.moveTo(ax, ay - 25); inpCtx.lineTo(ax, ay); inpCtx.stroke();
        inpCtx.beginPath(); inpCtx.moveTo(ax - 6, ay - 8); inpCtx.lineTo(ax + 6, ay - 8); inpCtx.lineTo(ax, ay + 2); inpCtx.closePath(); inpCtx.fill();
        inpCtx.font = "bold 11px sans-serif"; inpCtx.fillText("Velocity v = 3 m/s ↓", ax - 50, ay - 32);
    }
}

inpSelect.addEventListener("change", (e) => {
    inpCurrentIdx = parseInt(e.target.value);
    inpSlider.value = inpCurrentIdx;
    renderInp();
});
inpSlider.addEventListener("input", (e) => {
    inpCurrentIdx = parseInt(e.target.value);
    inpSelect.value = inpCurrentIdx;
    renderInp();
});
document.getElementById("btnInpSingle").addEventListener("click", () => {
    inpOverlay = false;
    document.getElementById("btnInpSingle").classList.add("active");
    document.getElementById("btnInpOverlay").classList.remove("active");
    renderInp();
});
document.getElementById("btnInpOverlay").addEventListener("click", () => {
    inpOverlay = true;
    document.getElementById("btnInpOverlay").classList.add("active");
    document.getElementById("btnInpSingle").classList.remove("active");
    renderInp();
});
document.getElementById("btnInpBlock").addEventListener("click", () => {
    inpShowBlock = true;
    document.getElementById("btnInpBlock").classList.add("active");
    document.getElementById("btnInpHide").classList.remove("active");
    renderInp();
});
document.getElementById("btnInpHide").addEventListener("click", () => {
    inpShowBlock = false;
    document.getElementById("btnInpHide").classList.add("active");
    document.getElementById("btnInpBlock").classList.remove("active");
    renderInp();
});

// ==========================================
// SLIDE 3: OUTPUT ROLLOUT CANVAS
// ==========================================
let outSampleIdx = 0;
let outTimeStep = 0;
let outField = "mag";
let outWire = true;
let outVectors = true;
let outShowDie = true;
let outIsPlaying = false;
let outTimer = null;

const outCanvas = document.getElementById("outCanvas");
const outCtx = outCanvas.getContext("2d");
const outSelect = document.getElementById("outSelect");
const outSlider = document.getElementById("outTimeSlider");
const outTooltip = document.getElementById("outTooltip");

function renderOut() {
    const smp = DATA[outSampleIdx];
    const step = smp.timesteps[outTimeStep];

    document.getElementById("outLblStep").textContent = outTimeStep;
    document.getElementById("outLblTNorm").textContent = `t* = ${step.t_norm.toFixed(2)}`;
    document.getElementById("outStatMaxMag").textContent = `${step.max_mag.toFixed(3)} mm`;
    document.getElementById("outStatMinUy").textContent = `${step.min_uy.toFixed(3)} mm`;

    const w = outCanvas.width, h = outCanvas.height;
    outCtx.clearRect(0, 0, w, h);

    const padL = 60, padR = 40, padT = 40, padB = 60;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const minX = -52, maxX = 2, minY = -2, maxY = 25;

    function toScreenX(x) { return padL + ((x - minX) / (maxX - minX)) * plotW; }
    function toScreenY(y) { return padT + (1 - (y - minY) / (maxY - minY)) * plotH; }

    outCtx.strokeStyle = "#1e293b"; outCtx.lineWidth = 1;
    for (let x = -50; x <= 0; x += 10) {
        const sx = toScreenX(x);
        outCtx.beginPath(); outCtx.moveTo(sx, padT); outCtx.lineTo(sx, h - padB); outCtx.stroke();
        outCtx.fillStyle = "#64748b"; outCtx.font = "11px sans-serif";
        outCtx.fillText(`${x}`, sx - 8, h - padB + 16);
    }
    for (let y = 0; y <= 25; y += 5) {
        const sy = toScreenY(y);
        outCtx.beginPath(); outCtx.moveTo(padL, sy); outCtx.lineTo(w - padR, sy); outCtx.stroke();
        outCtx.fillStyle = "#64748b"; outCtx.font = "11px sans-serif";
        outCtx.fillText(`${y}`, padL - 26, sy + 4);
    }

    outCtx.fillStyle = "#94a3b8"; outCtx.font = "12px sans-serif";
    outCtx.fillText("Deformed x-coordinate [mm]", w / 2 - 70, h - 18);
    outCtx.save(); outCtx.translate(16, h / 2 + 30); outCtx.rotate(-Math.PI / 2);
    outCtx.fillText("Deformed y-coordinate [mm]", 0, 0); outCtx.restore();

    const nx = smp.nx, ny = smp.ny;
    const xs = step.x, ys = step.y;
    const uxs = step.ux, uys = step.uy, mags = step.mag;

    let vMin = 0, vMax = 6.0;
    if (outField === "mag") { vMin = 0; vMax = 6.0; }
    else if (outField === "uy") { vMin = -6.0; vMax = 0.5; }
    else if (outField === "ux") { vMin = -2.5; vMax = 2.5; }

    document.getElementById("outCbarMin").textContent = `${vMin.toFixed(2)} mm`;
    document.getElementById("outCbarMax").textContent = `${vMax.toFixed(2)} mm`;

    for (let i = 0; i < nx - 1; i++) {
        for (let j = 0; j < ny - 1; j++) {
            const idx00 = i * ny + j;
            const idx10 = (i + 1) * ny + j;
            const idx11 = (i + 1) * ny + (j + 1);
            const idx01 = i * ny + (j + 1);

            let val = 0;
            if (outField === "mag") val = (mags[idx00] + mags[idx10] + mags[idx11] + mags[idx01]) / 4.0;
            else if (outField === "uy") val = (uys[idx00] + uys[idx10] + uys[idx11] + uys[idx01]) / 4.0;
            else if (outField === "ux") val = (uxs[idx00] + uxs[idx10] + uxs[idx11] + uxs[idx01]) / 4.0;

            const norm = (val - vMin) / (vMax - vMin || 1.0);
            outCtx.fillStyle = (outField === "ux") ? colormapCoolwarm(norm) : colormapViridis(norm);

            outCtx.beginPath();
            outCtx.moveTo(toScreenX(xs[idx00]), toScreenY(ys[idx00]));
            outCtx.lineTo(toScreenX(xs[idx10]), toScreenY(ys[idx10]));
            outCtx.lineTo(toScreenX(xs[idx11]), toScreenY(ys[idx11]));
            outCtx.lineTo(toScreenX(xs[idx01]), toScreenY(ys[idx01]));
            outCtx.closePath();
            outCtx.fill();

            if (outWire) {
                outCtx.strokeStyle = "rgba(0,0,0,0.25)";
                outCtx.lineWidth = 0.5;
                outCtx.stroke();
            }
        }
    }

    if (outVectors) {
        outCtx.strokeStyle = "rgba(255, 255, 255, 0.75)";
        outCtx.fillStyle = "rgba(255, 255, 255, 0.85)";
        outCtx.lineWidth = 1.2;
        for (let i = 2; i < nx - 2; i += 6) {
            for (let j = 2; j < ny - 2; j += 4) {
                const idx = i * ny + j;
                const px = toScreenX(xs[idx]), py = toScreenY(ys[idx]);
                const ux = uxs[idx], uy = uys[idx], mag = mags[idx];
                if (mag > 0.05) {
                    const scale = 14;
                    const tipX = px + ux * scale, tipY = py - uy * scale;
                    outCtx.beginPath(); outCtx.moveTo(px, py); outCtx.lineTo(tipX, tipY); outCtx.stroke();
                    const angle = Math.atan2(tipY - py, tipX - px);
                    outCtx.beginPath(); outCtx.moveTo(tipX, tipY);
                    outCtx.lineTo(tipX - 5 * Math.cos(angle - Math.PI / 6), tipY - 5 * Math.sin(angle - Math.PI / 6));
                    outCtx.lineTo(tipX - 5 * Math.cos(angle + Math.PI / 6), tipY - 5 * Math.sin(angle + Math.PI / 6));
                    outCtx.closePath(); outCtx.fill();
                }
            }
        }
    }

    if (outShowDie && smp.die_x && smp.die_y) {
        const indentOffset = (outTimeStep / 19.0) * (smp.max_indent || 5.7);
        outCtx.strokeStyle = "#f59e0b"; outCtx.lineWidth = 3.0;
        outCtx.beginPath();
        for (let j = 0; j < smp.die_x.length; j++) {
            const dx = toScreenX(smp.die_x[j]), dy = toScreenY(smp.die_y[j] - indentOffset);
            if (j === 0) outCtx.moveTo(dx, dy); else outCtx.lineTo(dx, dy);
        }
        outCtx.stroke();

        outCtx.lineTo(toScreenX(smp.die_x[smp.die_x.length - 1]), toScreenY(25));
        outCtx.lineTo(toScreenX(smp.die_x[0]), toScreenY(25));
        outCtx.closePath();
        outCtx.fillStyle = "rgba(245, 158, 11, 0.15)";
        outCtx.fill();

        outCtx.fillStyle = "#f59e0b"; outCtx.font = "bold 11px sans-serif";
        const midIdx = Math.floor(smp.die_x.length / 2);
        outCtx.fillText("Rigid Die S_d(x) (Mold) ↓", toScreenX(smp.die_x[midIdx]) - 55, toScreenY(smp.die_y[midIdx] - indentOffset) - 10);
    }
}

document.getElementById("outBtnPlay").addEventListener("click", () => {
    if (!outIsPlaying) {
        outIsPlaying = true;
        document.getElementById("outBtnPlay").textContent = "⏸ Pause";
        document.getElementById("outBtnPlay").classList.add("pause");
        outTimer = setInterval(() => {
            outTimeStep = (outTimeStep + 1) % 20;
            outSlider.value = outTimeStep;
            renderOut();
        }, 250);
    } else {
        outIsPlaying = false;
        document.getElementById("outBtnPlay").textContent = "▶ Play Rollout";
        document.getElementById("outBtnPlay").classList.remove("pause");
        clearInterval(outTimer);
    }
});
document.getElementById("outBtnReset").addEventListener("click", () => {
    if (outIsPlaying) document.getElementById("outBtnPlay").click();
    outTimeStep = 0; outSlider.value = 0; renderOut();
});
outSlider.addEventListener("input", (e) => {
    if (outIsPlaying) document.getElementById("outBtnPlay").click();
    outTimeStep = parseInt(e.target.value); renderOut();
});
outSelect.addEventListener("change", (e) => {
    outSampleIdx = parseInt(e.target.value); renderOut();
});
document.querySelectorAll("[data-field]").forEach(b => {
    b.addEventListener("click", () => {
        document.querySelectorAll("[data-field]").forEach(x => x.classList.remove("active"));
        b.classList.add("active");
        outField = b.getAttribute("data-field");
        renderOut();
    });
});
document.getElementById("outToggleWire").addEventListener("click", function() {
    outWire = !outWire; this.classList.toggle("active", outWire); renderOut();
});
document.getElementById("outToggleVectors").addEventListener("click", function() {
    outVectors = !outVectors; this.classList.toggle("active", outVectors); renderOut();
});
document.getElementById("outToggleDie").addEventListener("click", function() {
    outShowDie = !outShowDie; this.classList.toggle("active", outShowDie); renderOut();
});

// Initial load
goToSlide(0);
</script>
</body>
</html>
"""

final_html = presentation_template.replace("__INPUT_JSON__", input_json_str).replace("__OUTPUT_JSON__", output_json_str)

with open(out_dir / "interactive_presentation.html", "w", encoding="utf-8") as f:
    f.write(final_html)

print("Saved 3-slide interactive_presentation.html successfully!")
