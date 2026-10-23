# Surrogate Benchmark Evaluation Report

**Checkpoint:** `models/smart_temporal_experiment_2/best_model.pt`

---

### 1. Physical Accuracy Metrics (Real Physical Units: mm)
| Metric | Value | Reference Standard | Assessment |
| :--- | :--- | :--- | :--- |
| **Overall Relative L2 Error** | **11.45%** | < 5.0% | Adequate |
| **Surface Contact L2 Error** | **10.67%** | < 6.0% | Moderate |
| **Volume Interior L2 Error** | **11.72%** | < 5.0% | Moderate |
| **Maximum Pointwise Error (L_inf)** | **0.824 mm** | Peak displacement: ~6.5 mm | Peak localized deviation |
| **Sobolev H1 Semi-Norm Error** | **24.66%** | < 10.0% | Strain gradient fidelity |

---

### 2. Solid Mechanics Feasibility (Die Non-Penetration Constraint)
| Metric | Value | Constraint Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Max Die Penetration Depth** | **0.0000 mm** | < 0.05 mm | **PASS** |
| **Mean Die Penetration** | **0.0000 mm** | < 0.01 mm | **PASS** |
| **Contact Violation Rate** | **0.0%** of boundary nodes | < 5.0% | **PASS** |

---

### 3. Computational Performance vs. Numerical Solver (ABAQUS FEM)
| Hardware / Execution Metric | Evaluated Value |
| :--- | :--- |
| **Inference Latency per Sample** | **3.27 ms** |
| **Peak GPU VRAM Allocation** | **1442.1 MB** |
| **Trainable Parameters** | **2,607,044** |
| **Abaqus FEM Solver Runtime** | ~120.0 seconds / simulation |
| **Surrogate Speedup Factor** | **36,668x Faster** |

---

### 4. Training Convergence & Optimization KPIs
| Training Metric / KPI | Value | Reference / Status |
| :--- | :--- | :--- |
| **Total Completed Epochs** | **242** | Full training schedule executed |
| **Optimal Checkpoint Epoch** | **Epoch 239** | Lowest validation error checkpoint |
| **Initial Total Loss (Epoch 1)** | **1.1663e+00** (Train: 1.2683e+00) | Optimization starting point |
| **Optimal Validation Loss** | **5.9799e-02** | **94.9% Total Reduction** |
| **Final Epoch Total Loss** | Train: **5.4343e-02** | Val: **6.5072e-02** | Terminal training state |
| **Surface Contact Loss (Best)** | Train: **1.4426e-02** | Val: **1.3961e-02** | Boundary contact adherence |
| **Volume Interior Loss (Best)** | Train: **4.0412e-02** | Val: **4.5838e-02** | Continuum deformation fidelity |
| **Generalization Gap (Val - Train)** | **+4.9608e-03** | Generalization error spread |
| **Learning Rate Transition** | **1.50e-04 -> 1.43e-05** | Scheduled annealing |
| **Convergence Assessment** | **Strong Convergence (94.9% Error Drop, Peak at Epoch 239)** | Solid mechanics surrogate stability |

---

### 5. Benchmark Visualizations

#### A. Training Convergence Curves & Error per Epoch
![Training Convergence Curves](training_curves.png)

#### B. Deformed Mesh & Physical Displacement Verification Contours
![Surrogate Verification Contours](evaluation_contours.png)

#### C. Spatiotemporal Trajectory Evolution & Surrogate Accuracy
![evaluation_trajectory_sample00](evaluation_trajectory_sample00.png)

![evaluation_trajectory_sim907_sample00](evaluation_trajectory_sim907_sample00.png)

![evaluation_trajectory_sim908_sample01](evaluation_trajectory_sim908_sample01.png)

![evaluation_trajectory_sim909_sample02](evaluation_trajectory_sim909_sample02.png)

![evaluation_trajectory_sim910_sample03](evaluation_trajectory_sim910_sample03.png)


#### D. Verification Gallery Across Time Steps and Test Samples
- [evaluation_contours_sample00_sim907_step10_t0.526.png](evaluation_contours_sample00_sim907_step10_t0.526.png)
- [evaluation_contours_sample00_sim907_step19_t1.000.png](evaluation_contours_sample00_sim907_step19_t1.000.png)
- [evaluation_contours_sample00_step00_t0.000.png](evaluation_contours_sample00_step00_t0.000.png)
- [evaluation_contours_sample00_step05_t0.263.png](evaluation_contours_sample00_step05_t0.263.png)
- [evaluation_contours_sample00_step10_t0.526.png](evaluation_contours_sample00_step10_t0.526.png)
- [evaluation_contours_sample00_step15_t0.789.png](evaluation_contours_sample00_step15_t0.789.png)
- [evaluation_contours_sample00_step19_t1.000.png](evaluation_contours_sample00_step19_t1.000.png)
- [evaluation_contours_sample01_sim908_step10_t0.526.png](evaluation_contours_sample01_sim908_step10_t0.526.png)
- [evaluation_contours_sample01_sim908_step19_t1.000.png](evaluation_contours_sample01_sim908_step19_t1.000.png)
- [evaluation_contours_sample02_sim909_step10_t0.526.png](evaluation_contours_sample02_sim909_step10_t0.526.png)
- [evaluation_contours_sample02_sim909_step19_t1.000.png](evaluation_contours_sample02_sim909_step19_t1.000.png)
- [evaluation_contours_sample03_sim910_step10_t0.526.png](evaluation_contours_sample03_sim910_step10_t0.526.png)
- [evaluation_contours_sample03_sim910_step19_t1.000.png](evaluation_contours_sample03_sim910_step19_t1.000.png)
