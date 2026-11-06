# Análisis Post-Mortem: Divergencia Numérica (NaN Crash) en SMART Temporal

**Fecha del análisis:** 6 de Noviembre de 2026  
**Experimento analizado:** `models/smart_temporal_experiment_3`  
**Directorio de resultados:** `results/TrainingAnalysis/ds_plasticity_temporal/SMART/epochs_50_batch_1_lr_1e-4/smart_temporal_experiment_3`  
**Modelo:** SMART (Surrogate Model for Accelerated Residual Transformers)  
**Parámetros entrenables:** 11.380.676 (~11.38M)  

---

## 1. Resumen Ejecutivo

Durante la ejecución del entrenamiento del modelo subrogado **SMART Temporal** con 50 épocas, optimizador **Lion** (`lr=1e-4`, `warmup_cosine`) y tamaño de lote unitario (**`batch_size=1`**), el proceso completó las 50 épocas pero sufrió una **divergencia numérica irreversible hacia `NaN` (Not a Number)** en la **Época 5**.

A pesar del colapso numérico en las épocas 5–50:
1. **El mejor modelo está 100% a salvo y libre de corrupción**: el gestor de persistencia atómica guardó el checkpoint óptimo en la **Época 2** (`best_model.pt`) con un error relativo de validación de **`0.3217`** (reducción del 34.8%), con **0 NaNs** en sus pesos.
2. La evaluación sobre las 1.600 instancias de test del benchmark demostró una precisión relativa global del **`30.89%`**, una penetración nula contra la matriz rígida (**`0.0000 mm`**, 0% de violaciones de contacto) y una latencia de **`10.53 ms`** (una aceleración de **11.396x** respecto al solver FEM ABAQUS).

---

## 2. Cronología del Fallo Época por Época

Los datos registrados en `training_history.csv` revelan con exactitud la evolución del entrenamiento:

| Época | Fase del Scheduler | Learning Rate real | Loss Entrenamiento | Loss Validación | Estado del Checkpoint |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | Warmup lineal (ascenso) | `2.80e-05` | `0.6503` | `0.4938` | Convergencia inicial estable |
| **2** | Warmup lineal (ascenso) | `4.60e-05` | `0.4492` | **`0.3217`** | **Mejor modelo guardado (`best_model.pt`)** |
| **3** | Warmup lineal (ascenso) | `6.40e-05` | `0.4076` | `0.4192` | Entrenamiento estable |
| **4** | Warmup lineal (ascenso) | `8.20e-05` | `0.5056` | `0.3326` | Aumento de oscilaciones |
| **5** | **PICO MÁXIMO DEL WARMUP** | **`1.00e-04`** | **`nan`** | `0.5093` | 💥 **Desbordamiento a NaN en Train** |
| **6** | Cosine Annealing (descenso) | `9.98e-05` | `nan` | `nan` | Contaminación total en Val |
| **7 – 50** | Cosine Annealing (descenso) | `... -> 1e-06` | `nan` | `nan` | Estado estacionario en NaN |

---

## 3. Diagnóstico Matemático y Computacional: ¿Qué es el NaN y cómo se generó?

### 3.1. El concepto de `NaN` en coma flotante (IEEE 754)
`NaN` (*Not a Number*) no es un error de programación en Python, sino una condición de parada matemática del procesador que surge cuando se intenta ejecutar una operación indefinida:
$$\frac{0}{0} = \text{NaN}, \quad \frac{\infty}{\infty} = \text{NaN}, \quad \infty - \infty = \text{NaN}, \quad \sqrt{-x} = \text{NaN}$$

En una red neuronal profunda, el `NaN` actúa como un **virus de propagación instantánea**:
$$\text{Salida} = W \cdot x + b$$
Si un único elemento de una matriz se vuelve `NaN`, cualquier producto escalar o suma matricial que lo toque genera automáticamente `NaN`. En menos de una iteración, las 8 capas del transformer quedan completamente contaminadas.

### 3.2. El techo físico de FP16 (AMP - Automatic Mixed Precision)
Para acelerar el cómputo en la GPU NVIDIA RTX 3060, se utilizó `--use_amp` (precisión media de 16 bits).
* En el formato **FP16**, la mantisa y el exponente solo pueden codificar números hasta un valor máximo de **`65.504`**.
* Cualquier cálculo que supere `65.504` no se redondea con pérdida: la GPU lo sustituye inmediatamente por **`+Inf` (Infinito)**.

### 3.3. El crecimiento explosivo de la Atención (Softmax y $e^x$)
La arquitectura SMART contiene 8 bloques de encoder-decoder con atención multi-cabeza (`MultiheadAttention`). En cada capa de atención se proyectan los vectores de consulta ($Q$) y clave ($K$) y se calcula la matriz de atención mediante Softmax:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V = \frac{e^{Q K^T / \sqrt{d_k}}}{\sum e^{Q K^T / \sqrt{d_k}}} V$$

La función exponencial ($e^x$) crece de manera hiperbólica:
* $e^5 \approx 148$
* $e^8 \approx 2.980$
* $e^{10} \approx 22.026$
* $e^{11.1} \approx 66.170 \quad \longrightarrow \quad \mathbf{Supera\ el\ límite\ de\ 65.504\ de\ FP16!}$

Si por un paso de optimización abrupto las activaciones crecen moderadamente de forma que el producto escalar $Q K^T / \sqrt{d_k}$ alcanza valores de apenas $12$ o $15$, la GPU evalúa $e^{12} = \infty$. Al normalizar la Softmax:
$$\frac{\infty}{\infty} = \mathbf{NaN}$$

### 3.4. El escalado de gradientes en PyTorch (`GradScaler`)
En precisión mixta, PyTorch escala los gradientes multiplicándolos por un factor alto (típicamente $65.536$) para evitar que los gradientes diminutos desaparezcan (underflow).
$$\text{Gradiente escalado} = \text{Gradiente} \times 65.536$$
Si una muestra del dataset presenta un gradiente puntual de tan solo $1.2$ o $1.5$ (habitual en problemas elastoplásticos con contacto no lineal), el gradiente escalado supera $98.000$, desbordando FP16 a $\infty$. Al desescalar ($\infty / 65.536$), el resultado numérico es `NaN`.

---

## 4. El Factor Detonante: La interacción entre Lion, `batch_size=1` y el Pico del Warmup

¿Por qué los modelos entrenados previamente en este repositorio no sufrieron este problema? La respuesta radica en la combinación simultánea de tres decisiones de diseño:

### 4.1. La naturaleza no-adaptativa de Lion
A diferencia de AdamW (que divide el gradiente por su desviación estándar $\sqrt{v_t} + \epsilon$, reduciendo drásticamente el tamaño del paso si el gradiente es ruidoso), **Lion (Google Brain, 2023)** actualiza los parámetros basándose exclusivamente en el **signo del gradiente acumulado**:
$$\theta_t \leftarrow \theta_t - \eta \cdot \text{sign}\left(\beta_1 m_{t-1} + (1 - \beta_1) g_t\right)$$
Esto implica que **cada parámetro de la red se desplaza exactamente una distancia fija $\pm \eta$**, sin importar si el gradiente real era de magnitud $10^{-6}$ o de $10^{2}$.

### 4.2. Varianza extrema por ausencia de promediado (`batch_size = 1`)
En la literatura de optimizadores basados en signo (como Lion o SignSGD), se exige expresamente el uso de **tamaños de lote grandes (32, 64, 128 o superiores)**.  
* Con `batch_size = 32`, se promedian 32 geometrías y trayectorias distintas: el ruido se cancela y la dirección del signo es representativa del gradiente real del problema.
* Con `batch_size = 1`, cada simulación individual empuja los 11.38 millones de parámetros a su propio mínimo local.
* En 4 épocas completas, el modelo dio:
  $$4 \text{ épocas} \times 18.000 \text{ pasos/época} = \mathbf{72.000 \text{ pasos seguidos sin ningún promediado}}$$

### 4.3. La coincidencia matemática exacta con el Pico del Warmup
El scheduler `warmup_cosine` fue configurado para 50 épocas con un ratio de calentamiento del 10%:
$$\text{Épocas de Warmup} = 50 \times 0.10 = \mathbf{5 \text{ épocas}}$$
Durante las primeras 4 épocas, el learning rate efectivo era reducido:
* Época 1: $\eta = 2.8 \times 10^{-5}$
* Época 2: $\eta = 4.6 \times 10^{-5}$ (óptimo alcanzado)
* Época 3: $\eta = 6.4 \times 10^{-5}$
* Época 4: $\eta = 8.2 \times 10^{-5}$

Con estos valores moderados, el modelo amortiguó la varianza estocástica de las muestras individuales.  
Sin embargo, **justo en la Época 5 el scheduler alcanzó su pico máximo nominal de $\eta = 1.0 \times 10^{-4}$**.  
Con pasos de signo de tamaño máximo $\pm 10^{-4}$ repetidos 18.000 veces consecutivas sobre muestras aisladas, bastó una simulación con concentración de tensiones para generar una oscilación en las activaciones de atención, desencadenar el desbordamiento de FP16 y corromper la red con `NaN`.

---

## 5. Integridad del Modelo y Resultados Válidos

Afortunadamente, el sistema de persistencia modular de la base de código (`TorchCheckpointHandler`) implementa seguimiento continuo del mejor loss de validación:
* El archivo [`best_model.pt`](file:///home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/models/smart_temporal_experiment_3/best_model.pt) se escribió al terminar la **Época 2**, momento en el que el modelo no contenía ningún valor anómalo.
* **Comprobación de tensores:** Se ha verificado programáticamente que los 460 tensores de pesos en `best_model.pt` tienen `has_nan == False`.
* **Rendimiento validado en test (1.600 casos):**
  * Error Relativo L2 Global: **`30.89%`**
  * Error de Contacto Superficial: **`28.00%`**
  * Violaciones de contacto (penetración de troquel): **`0.0%` (0.0000 mm)**
  * Aceleración frente a FEM: **`11.396x`**

---

## 6. Recomendaciones Técnicas para Futuras Ejecuciones

Para entrenar modelos SMART de alta capacidad (11.38M parámetros) con estabilidad garantizada:

1. **Aumentar el tamaño de lote a `batch_size = 32` (o 16):**
   * Es el estándar recomendado para el optimizador Lion.
   * Reduce la varianza del gradiente a niveles seguros.
   * Disminuye el tiempo de 50 épocas de **24.8 horas a ~1.2 horas** al saturar adecuadamente los núcleos Tensor de la GPU RTX 3060.
2. **Si se requiere experimentar con `batch_size = 1`:**
   * Utilizar **`AdamW`** en lugar de Lion (AdamW escala cada paso dividiendo por la raíz de los gradientes al cuadrado, amortiguando picos extremos).
   * O bien reducir el learning rate de Lion a $\eta \le 3 \times 10^{-5}$ (evitando superar el umbral de estabilidad observado en la época 2).
3. **Mecanismos de parada temprana:**
   * Añadir el parámetro `--early-stopping-patience 5` para que, en caso de estancamiento o divergencia, el script corte la ejecución inmediatamente sin consumir horas innecesarias de GPU.
