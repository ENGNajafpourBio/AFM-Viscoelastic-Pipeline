# 🔬 Nanomechanical & Viscoelastic Analysis Pipeline for AFM Data

An enterprise-grade, clinical-research Python framework designed to extract nanoscale viscoelastic properties from Atomic Force Microscopy (AFM) stress-relaxation experiments. This pipeline quantifies the nanomechanical biomarker differences between healthy and virus-infected living cells.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary
In mechanobiology, quantifying cellular viscoelasticity is crucial for disease diagnostics. However, raw AFM force-displacement curves are heavily influenced by substrate artifacts and local minima traps during mathematical fitting. 

This repository solves these challenges by implementing an object-oriented **Dual-Stage Hybrid Optimization Pipeline** bundled with **Dimitriadis finite-thickness boundary corrections**, achieving highly stable, publication-ready biomechanical profiling.

### 📊 Pipeline Visual Output
Here is the comprehensive 6-panel clinical diagnostic dashboard generated directly by the OOP architecture:

![AFM Analysis Dashboard](output/afm_results_oop.jpg)

---

## ⚙️ Software Architecture & Design Patterns
The codebase was refactored from a monolithic script into a production-ready **Object-Oriented Programming (OOP)** structure, adhering to SOLID principles:

* **Registry Pattern (`viscoelastic_models.py`):** Implements an abstract base class (`ViscoelasticModel`) and a central registry. New biomechanical models (e.g., Maxwell, Burgers) can be integrated seamlessly without altering the core optimization engine.
* **Immutability with Dataclasses (`physics_engine.py`):** Utilizes `@dataclass(frozen=True)` for experimental geometry parameters, ensuring physics constants remain untampered during analytical runtime loop execution.
* **Separation of Concerns:** Isolated physics engines, mathematical models, and mathematical optimization layers. `main.py` purely orchestrates execution.

---

## 📐 Mathematical Framework & Physics Implementation

### 1. Finite-Thickness Boundary Correction
Standard Hertzian contact mechanics assume infinite sample depth. For thin, soft biological cells, the rigid underlying petri-dish substrate artificially inflates the measured Young's Modulus. This pipeline integrates the **Dimitriadis Correction** factor ($f_{\text{dim}}$) for a spherical indenter bonded to a rigid substrate:

$$f_{\text{dim}}(c) = 1 + 1.133c + 1.283c^2 + 0.769c^3 + 0.096c^4$$

Where $c = \frac{\sqrt{R \cdot \delta_0}}{h}$, $R$ is probe radius, $\delta_0$ is indentation depth, and $h$ is cell thickness. The physics engine automatically inverts the measured forces to reconstruct the true relaxation modulus $E^*(t)$.

### 2. Viscoelastic Models Implemented
* **Standard Linear Solid (SLS) / Maxwell:** Classic spring-dashpot framework.
* **Fractional Order Zener (FKV):** Implements fractional calculus (Springpot element) to model power-law relaxation behavior often observed in complex living cytoplasm:

$$E^*(t) = E_{\infty} + C \cdot t^{-\alpha}$$

---

## ⚡ Dual-Stage Hybrid Optimization Strategy
Standard gradient descent algorithms (like Levenberg-Marquardt alone) frequently fail or converge to sub-optimal local minima due to the highly non-linear, multi-modal loss surface of fractional order parameters.

To guarantee global convergence, this framework utilizes a **Two-Stage Hybrid Optimization Strategy**:
1.  **Stage 1 (Global):** Stochastic **Differential Evolution (DE)** explores the entire bounded parameter space to find a reliable global seed, bypassing local traps.
2.  **Stage 2 (Local):** The **Levenberg-Marquardt (LM) / Trust Region Reflective (TRF)** algorithm refines the DE seed with high numerical precision, computing standard errors via the Jacobian matrix covariance approximation.

---

## 🚀 Quick Start & Reproducibility

### Installation
```bash
git clone [https://github.com/YOUR_USERNAME/AFM-Viscoelastic-Pipeline.git](https://github.com/YOUR_USERNAME/AFM-Viscoelastic-Pipeline.git)
cd AFM-Viscoelastic-Pipeline
pip install -r requirements.txt