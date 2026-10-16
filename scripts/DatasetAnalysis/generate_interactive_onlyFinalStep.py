"""Generate interactive HTML visualizations for ds_plasticity_onlyFinalStep.

Outputs:
- results/DatasetAnalysis/ds_plasticity_onlyFinalStep/interactive_input.html
- results/DatasetAnalysis/ds_plasticity_onlyFinalStep/interactive_output.html
- results/DatasetAnalysis/ds_plasticity_onlyFinalStep/interactive_presentation.html
"""

import argparse
import json
from pathlib import Path
import re
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate interactive HTML for onlyFinalStep dataset.")
    parser.add_argument("--data_dir", type=str, default="data/processed/ds_plasticity_onlyFinalStep")
    parser.add_argument("--output_dir", type=str, default="results/DatasetAnalysis/ds_plasticity_onlyFinalStep")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    train = np.load(data_dir / "train.npz")
    die_boundary = train["die_boundary"]       # (900, 101, 2)
    coords = train["coords"]                   # (900, 101, 31, 2)
    coords_0 = train["coords_initial"]         # (900, 101, 31, 2)
    displacements = train["displacements"]     # (900, 101, 31, 2)

    # 1. BUILD INPUT JSON (50 samples)
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

    # 2. BUILD OUTPUT TRAJECTORY JSON (4 samples: 0, 1, 2, 5)
    chosen_sample_ids = [0, 1, 2, 5]
    trajectory_samples = []

    for sid in chosen_sample_ids:
        s_x0 = coords_0[sid, :, :, 0]
        s_y0 = coords_0[sid, :, :, 1]
        s_xf = coords[sid, :, :, 0]
        s_yf = coords[sid, :, :, 1]
        s_uxf = displacements[sid, :, :, 0]
        s_uyf = displacements[sid, :, :, 1]

        timesteps_data = []
        for t in range(20):
            alpha = t / 19.0
            x_t = s_x0 + alpha * (s_xf - s_x0)
            y_t = s_y0 + alpha * (s_yf - s_y0)
            ux_t = alpha * s_uxf
            uy_t = alpha * s_uyf
            mag_t = np.sqrt(ux_t**2 + uy_t**2)

            x = [round(float(v), 2) for v in x_t.flatten()]
            y = [round(float(v), 2) for v in y_t.flatten()]
            ux = [round(float(v), 3) for v in ux_t.flatten()]
            uy = [round(float(v), 3) for v in uy_t.flatten()]
            mag = [round(float(v), 3) for v in mag_t.flatten()]

            timesteps_data.append({
                "t_step": t,
                "t_norm": round(alpha, 2),
                "x": x,
                "y": y,
                "ux": ux,
                "uy": uy,
                "mag": mag,
                "max_mag": round(float(np.max(mag_t)), 3),
                "min_uy": round(float(np.min(uy_t)), 3),
            })

        trajectory_samples.append({
            "sample_id": sid,
            "nx": 101,
            "ny": 31,
            "num_steps": 20,
            "timesteps": timesteps_data,
            "die_x": [round(float(v), 2) for v in die_boundary[sid, :, 0]],
            "die_y": [round(float(v), 2) for v in die_boundary[sid, :, 1]],
            "max_indent": round(float(np.abs(np.min(s_uyf))), 3),
        })

    output_json_str = json.dumps(trajectory_samples)

    # 3. EXTRACT TEMPLATES FROM BASE SCRIPTS
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / "generate_interactive_visualizations.py", "r", encoding="utf-8") as f:
        vis_code = f.read()

    with open(script_dir / "generate_interactive_presentation.py", "r", encoding="utf-8") as f:
        pres_code = f.read()

    m_in = re.search(r'input_template = """(.*?)"""', vis_code, re.DOTALL)
    m_out = re.search(r'output_template = """(.*?)"""', vis_code, re.DOTALL)
    m_pres = re.search(r'presentation_template = r?"""(.*?)"""', pres_code, re.DOTALL)

    if not (m_in and m_out and m_pres):
        raise RuntimeError("Failed to extract HTML templates from visualization scripts.")

    input_template = m_in.group(1)
    output_template = m_out.group(1)
    presentation_template = m_pres.group(1)

    # 4. CUSTOMIZE TEMPLATES FOR ds_plasticity_onlyFinalStep
    # Input template customizations
    input_template = input_template.replace(
        "<title>Input Parameter Explorer - 2D Plasticity Benchmark</title>",
        "<title>Input Parameter Explorer - 2D Plasticity Benchmark (Only Final Step)</title>"
    ).replace(
        "<h1>Input Parameter: <span class=\"highlight\">Rigid Die Profile S_d(x)</span></h1>",
        "<h1>Input Parameter: <span class=\"highlight\">Rigid Die Profile S_d(x) (Only Final Step)</span></h1>"
    ).replace(
        "T = 20 uniform increments",
        "T = 1 final increment (t = -1, steady state)"
    ).replace(
        "t ∈ {t₀, t₁, …, t₁₉}",
        "Static steady-state formulation (ConsiderOnlyFinalStepModeler)"
    )

    final_input_html = input_template.replace("__INPUT_JSON__", input_json_str)
    with open(out_dir / "interactive_input.html", "w", encoding="utf-8") as f:
        f.write(final_input_html)
    print(f"Saved: {out_dir / 'interactive_input.html'}")

    # Output template customizations
    output_template = output_template.replace(
        "<title>Output Parameter Explorer - 2D Plasticity Benchmark</title>",
        "<title>Output Solution Explorer - 2D Plasticity Benchmark (Only Final Step)</title>"
    ).replace(
        "<h1>Output Solution: <span class=\"highlight\">Spatiotemporal Mesh Deformation & Indentation</span></h1>",
        "<h1>Output Solution: <span class=\"highlight\">Static Final-Step Mesh Deformation & Indentation</span></h1>"
    ).replace(
        "T = 20 uniform increments",
        "T = 1 final increment (steady state)"
    ).replace(
        "t ∈ {t₀, t₁, …, t₁₉}",
        "Static steady-state formulation (ConsiderOnlyFinalStepModeler)"
    )

    final_output_html = output_template.replace("__OUTPUT_JSON__", output_json_str)
    with open(out_dir / "interactive_output.html", "w", encoding="utf-8") as f:
        f.write(final_output_html)
    print(f"Saved: {out_dir / 'interactive_output.html'}")

    # Presentation template customizations
    presentation_template = presentation_template.replace(
        "<title>2D Plasticity Benchmark: Dataset Analysis Deck</title>",
        "<title>2D Plasticity Benchmark (Only Final Step): Dataset Analysis Deck</title>"
    ).replace(
        "<h2>🔬 2D Plasticity Benchmark</h2>",
        "<h2>🔬 2D Plasticity Benchmark (Only Final Step)</h2>"
    ).replace(
        "output : (987, 101, 31, 20, 4)",
        "output : (987, 101, 31, 2) displacements / (987, 101, 31, 4) target_all"
    ).replace(
        "987 runs × $101$ x-nodes × $31$ y-nodes × $20$ temporal increments × $4$ solution channels:",
        "987 runs × $101$ x-nodes × $31$ y-nodes × static final step ($t=-1$) × 2 displacement channels (4 for target_all):"
    ).replace(
        "900 samples (1,373.95 MB, train.npz)",
        "900 samples (165.97 MB, train.npz)"
    ).replace(
        "7 val (10.81 MB) / 80 test (122.06 MB)",
        "7 val (1.31 MB) / 80 test (14.75 MB)"
    ).replace(
        "T = 20 increments (t* ∈ [0, 1])",
        "T = 1 static final increment (t = -1, steady state)"
    ).replace(
        "T = 20 uniform increments",
        "T = 1 static increment (steady-state final step)"
    ).replace(
        "Recorded stages t ∈ {t₀, t₁, …, t₁₉}",
        "Final stage t = -1 (temporal dimension reduced via ConsiderOnlyFinalStepModeler)"
    )

    final_pres_html = presentation_template.replace("__INPUT_JSON__", input_json_str).replace("__OUTPUT_JSON__", output_json_str)
    with open(out_dir / "interactive_presentation.html", "w", encoding="utf-8") as f:
        f.write(final_pres_html)
    print(f"Saved: {out_dir / 'interactive_presentation.html'}")

    print("\nAll 3 interactive HTML visualization artifacts created successfully!")


if __name__ == "__main__":
    main()
