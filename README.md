Two_State_model/
│
├── Two_state_Gs.py
│   └── Numerically calculate the Displacement Probability Density Function (DPDF)
│       of the two-state diffusion model.
│
Simulation/
│
├── sim_run.py
│   └── Batch run of Brownian Dynamics (BD) simulations.
│
├── sys_init_exist_di.py
├── sys_init_di.py
│   └── Initialize simulation systems (particle and network configuration).
│
├── sim_reaction_di.py
│   └── Main BD simulation program.
│
├── delete_0xml_dcd.py
│   └── Utility to remove temporary simulation files (.xml, .dcd).
│
├── analyse_function_raw_di.py
├── analyse_function_distri0.py
│   └── Analyze simulation results.
│
├── net/
│   └── Store network configuration files.
│
├── particles/
│   └── Store particle configuration files.
│
└── simulation/
    └── Store simulation outputs.
│
SCF/
│
├── dat_file_init.py
│   └── Generate required `.dat` initialization files for SCF runs.
│
├── kernel_pureNet.cu
├── kernel_pureNet
│   └── SCF kernel (CUDA source and compiled executable) for **pure networks**
│       (without nanoparticles and dynamic bonds).
│
├── kernel_M20_V5.cu
├── kernel_M20_V5
│   └── SCF kernel (CUDA source and compiled executable) for **networks with
│       nanoparticles and dynamic bonds**.
│
├── run_one_pos.py
│   └── Run SCF simulation for a **single nanoparticle position**.
│
└── run_over_space.py
    └── Run SCF simulations for **multiple nanoparticle positions** by calling
        `run_one_pos.py`.
