# Surrogate Benchmark Evaluation Report

**Checkpoint:** `/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/models/LION/batch_size_16_fp32_swap_test/best_model.pt`

---

### 1. Physical Accuracy Metrics (Real Physical Units: mm)
| Metric | Value | Reference Standard | Assessment |
| :--- | :--- | :--- | :--- |
| **Pure Displacement Relative L2 Error** | **3.16%** | < 5.0% | Excellent |
| **Joint 4-Channel Relative L2 Error (Literature)** | **0.06%** | < 1.0% | Excellent |
| **Surface Contact L2 Error** | **3.37%** | < 6.0% | Strong Contact Adherence |
| **Volume Interior L2 Error** | **3.15%** | < 5.0% | Accurate Continuum Flow |
| **Maximum Pointwise Error (L_inf)** | **0.446 mm** | Peak displacement: ~6.5 mm | Peak localized deviation |
| **Sobolev H1 Semi-Norm Error** | **9.91%** | < 10.0% | Strain gradient fidelity |

---

### 2. Solid Mechanics Feasibility (Die Non-Penetration Constraint)
| Metric | Value | Constraint Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Max Die Penetration Depth** | **0.0002 mm** | < 0.05 mm | **PASS** |
| **Mean Die Penetration** | **0.0002 mm** | < 0.01 mm | **PASS** |
| **Contact Violation Rate** | **0.0%** of boundary nodes | < 5.0% | **PASS** |

---

### 3. Computational Performance vs. Numerical Solver (ABAQUS FEM)
| Hardware / Execution Metric | Evaluated Value |
| :--- | :--- |
| **Inference Latency per Sample** | **10.49 ms** |
| **Peak GPU VRAM Allocation** | **1540.4 MB** |
| **Trainable Parameters** | **11,380,676** |
| **Abaqus FEM Solver Runtime** | ~120.0 seconds / simulation |
| **Surrogate Speedup Factor** | **11,442x Faster** |

---

### 4. Training Convergence & Optimization KPIs
| Training Metric / KPI | Value | Reference / Status |
| :--- | :--- | :--- |
| **Total Completed Epochs** | **150** | Full training schedule executed |
| **Optimal Checkpoint Epoch** | **Epoch 149** | Lowest validation error checkpoint |
| **Initial Total Loss (Epoch 1)** | **1.1633e+00** (Train: 1.2927e+00) | Optimization starting point |
| **Optimal Validation Loss** | **1.3879e-02** | **98.8% Total Reduction** |
| **Final Epoch Total Loss** | Train: **1.2056e-02** | Val: **1.3884e-02** | Terminal training state |
| **Surface Contact Loss (Best)** | Train: **1.6940e-03** | Val: **1.9350e-03** | Boundary contact adherence |
| **Volume Interior Loss (Best)** | Train: **1.0388e-02** | Val: **1.1944e-02** | Continuum deformation fidelity |
| **Generalization Gap (Val - Train)** | **+1.7970e-03** | Generalization error spread |
| **Learning Rate Transition** | **1.60e-05 -> 1.00e-06** | Scheduled annealing |
| **Convergence Assessment** | **Strong Convergence (98.8% Error Drop, Peak at Epoch 149)** | Solid mechanics surrogate stability |

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
