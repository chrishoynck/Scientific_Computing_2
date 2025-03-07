# **Pattern Formation and Aggregation in DLA & Gray-Scott**
By Bart Koedijk (15756785), Charlotte Koolen (15888592), & Chris Hoynck van Papendrecht (15340791)

This repository contains numerical experiments and implementations for simulating **Diffusion-Limited Aggregation (DLA)** and the **Gray-Scott** reaction-diffusion model. The focus is on understanding how different parameters and stochastic processes influence pattern formation.

## **Project Overview**
- **DLA (Diffusion-Limited Aggregation)**  
  - Concentration-based approach using iterative solvers (e.g., SOR).  
  - Monte Carlo approach with random walkers (Brownian motion).  
- **Gray-Scott Reaction-Diffusion Model**  
  - Exploring pattern formation under various parameter settings.  
  - Investigating the effects of added noise.

## **Main Components**
- **`main.ipynb`** – A Jupyter Notebook demonstrating the setup, simulation steps, and visualizations for both DLA and Gray-Scott models.
- **`src/solutions.py`** – Core implementations of the DLA (concentration-based and Monte Carlo) and Gray-Scott simulations.
- **`src/visualizations.py`** – Plotting utilities for visualizing cluster growth and reaction-diffusion patterns.
- **`data/`** – Stores output files (e.g., grid states) and any auxiliary data.
- **`plots/`** – Directory for saving generated figures and animations.
- **`README.md`** – This file, describing the project’s structure and usage.

## **Installation & Setup**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/chrishoynck/Scientific_Computing_2
   cd Scientific_Computing_2
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## **Usage**
1. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook main.ipynb
   ```
2. **Follow the notebook cells** to see how each simulation is set up, how parameters are varied, and to render results.

