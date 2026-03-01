## Overview
This repository contains code for (1) numerical calculation of displacement probability density functions (DPDF) for a two-state diffusion model, (2) Brownian Dynamics (BD) simulations, and (3) Self-Consistent Field (SCF) calculations.

---

# Project Structure

## `Two_State_model/`
- `Two_state_Gs.py`  
  Numerically calculates the **Displacement Probability Density Function (DPDF)** of the **two-state diffusion model**.

## `Simulation/`
- `sim_run.py`  
  Batch execution of **Brownian Dynamics (BD)** simulations.

- `sys_init_exist_di.py`, `sys_init_di.py`  
  Initialize simulation systems.

- `sim_reaction_di.py`  
  Main **Brownian Dynamics simulation** program.

- `delete_0xml_dcd.py`  
  Utility script to **remove temporary files** (`.xml`, `.dcd`).

- `analyse_function_raw_di.py`, `analyse_function_distri0.py`  
  Analyze simulation results.

### Directories
- `net/` — Stores **network configuration** files.  
- `particles/` — Stores **particle configuration** files.  
- `simulation/` — Stores **simulation output** files.

## `SCF/`
- `dat_file_init.py`  
  Generates required `.dat` initialization files for SCF simulations.

- `kernel_pureNet.cu`, `kernel_pureNet`  
  **SCF kernel** (CUDA source and executable) for **pure networks** (without nanoparticles or dynamic bonds).

- `kernel_M20_V5.cu`, `kernel_M20_V5`  
  **SCF kernel** (CUDA source and executable) for **networks containing nanoparticles and dynamic bonds**.

- `run_one_pos.py`  
  Run SCF calculation for a **single nanoparticle position**.

- `run_over_space.py`  
  Run SCF calculations for **multiple nanoparticle positions**, by calling `run_one_pos.py`.

---

## SCF Calculation Workflow
1. Run `kernel_pureNet` to simulate a **pure network**, obtaining a **steady-state crosslink density field**.  
2. Use this density field as the **initial distribution** for subsequent simulations with `kernel_M20_V5`, which includes **nanoparticles** and **dynamic bonds**.

---

## Available Equilibrated `.dat` Files
A collection of equilibrated `.dat` files is available for download:  
https://cloud.tsinghua.edu.cn/d/3cc96fd8e6c04a65935e/
