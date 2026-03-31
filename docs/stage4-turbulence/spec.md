# Stage 4: Turbulence

## Overview

Stage 4 solves the gyrokinetic equations to compute turbulent transport. The primary outputs -- heat and particle fluxes -- are both optimization objectives (to minimize) AND direct transport inputs for Stage 5.

**JAX-first priority:** SPECTRAX-GK is the primary code (JAX-native, differentiable). GX and GENE are traditional alternatives added later.

**Position in pipeline:** Receives geometry from Stage 1/2. Runs in parallel with Stage 3 (Neoclassical). Outputs feed Stage 5 (Transport).

**Important coordination point:** The coupling between SPECTRAX-GK output and NEOPAX (Stage 5) is less mature than the GX-Trinity3D coupling. NEOPAX has turbulence-coupling utilities but the public examples focus on the neoclassical reduced model. The Stage 4 and 5 owners must coordinate on this interface.

Reference: `stellarator_workflow.tex`, Section 4.7; `stellarator_io_reference.tex`, Sections 3.9-3.10.

## Codes

### SPECTRAX-GK (Primary JAX)
- **Repository:** https://github.com/uwplasma/SPECTRAX-GK
- **Language:** Python/JAX
- **Role:** JAX-native gyrokinetic solver for differentiable turbulence calculations

### GX (Alternative)
- **Repository:** https://bitbucket.org/gyrokinetics/gx
- **Language:** Fortran/CUDA
- **Role:** GPU-native gyrokinetic code, mature coupling with Trinity3D

### GENE / GENE-3D (Alternative)
- **Website:** https://genecode.org
- **Language:** Fortran
- **Role:** High-fidelity grid-based Eulerian gyrokinetic code

### Installation & Platform

<!-- OWNER COMPLETES: Document installation instructions for SPECTRAX-GK (primary), including:
     - Python/JAX version requirements and GPU/TPU backend setup
     - pip/conda/pixi install steps
     - Known platform issues (e.g., JAX on macOS ARM, CUDA version constraints)
     - Any dependency conflicts with other StellaForge stages
     - For GX: build instructions (Fortran compiler, CUDA toolkit, MPI)
     - For GENE: license and access process, build system requirements
     - Verified platform matrix (OS, GPU, JAX version combinations that are tested) -->

## Input Specification

### SPECTRAX-GK Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| TOML config | file | Grid, geometry, physics toggles, time integration | User-specified |
| Geometry | analytic or `*.eik.nc` | Magnetic geometry (can be VMEC-derived) | Stage 1/2 |
| Species profiles | in config | Density, temperature, gradients per species | User-specified |
| Collisionality | in config | Collision parameters | User-specified |
| Beta | in config | Electromagnetic parameter | User-specified |

### GX Inputs (Alternative)

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `run_name.in` | Input file | Geometry, species, domain, time stepping, diagnostics, resolution | User-specified |
| VMEC geometry | via geometry module | Field-line geometry from wout | Stage 1 |
| `omega=true` | flag | Enable growth-rate diagnostics | Config |
| `fluxes=true` | flag | Enable flux diagnostics | Config |

### GENE Inputs (Alternative)

Installation-dependent. Key physics contract: geometry from VMEC/Boozer, species profiles/gradients, collisionality, electromagnetic parameters, numerical grid settings.

### Input Validation

<!-- OWNER COMPLETES: After running the code, validate the input tables above against actual code behavior. Document:
     - Which TOML config fields are required vs. optional for SPECTRAX-GK
     - Default values for optional fields (especially grid resolution, time step, physics toggles)
     - How geometry is loaded: analytic specification vs. eik.nc file path in config
     - Species profile format: exact key names, units, normalization conventions
     - Any fields the TeX spec missed or that are deprecated
     - Differences between SPECTRAX-GK and GX input conventions for the same physics
     - For GX: validate the .in file format and geometry module interface -->

## Output Specification

### SPECTRAX-GK Outputs

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `gamma` | scalar/array | Linear growth rate | Objective / screening |
| `omega` | scalar/array | Real frequency | Diagnostic |
| `gamma_t` | 1D array (time) | Growth rate time trace | Convergence check |
| `omega_t` | 1D array (time) | Frequency time trace | Convergence check |
| `Wg_t` | 1D array (time) | Free energy (g) trace | Diagnostic |
| `Wphi_t` | 1D array (time) | Free energy (phi) trace | Diagnostic |
| `Wapar_t` | 1D array (time) | Free energy (A_parallel) trace | Diagnostic |
| `heat_flux_t` | 1D array (time) | Heat flux time trace | **Transport input** |
| `particle_flux_t` | 1D array (time) | Particle flux time trace | **Transport input** |

Optional CSV output: time, growth rate, frequency, free energy, species-resolved heat and particle flux.

The natural downstream contract is the same as GX: turbulent heat and particle flux (steady-state values).

### GX Outputs (Alternative)

| File | Description |
|------|-------------|
| `run_name.out.nc` | Linear run output |
| `run_name.nc` | Nonlinear run output |
| `run_name.big.nc` | Saved field diagnostics |
| `run_name.restart.nc` | Restart data |

Key NetCDF groups: `Grids`, `Geometry`, `Diagnostics`, `Inputs`

| Field | Location | Description | Used As |
|-------|----------|-------------|---------|
| `ParticleFlux_st` | `Diagnostics/` | Particle flux (species, time) | **Transport input** (Trinity3D) |
| `HeatFlux_st` | `Diagnostics/` | Heat flux (species, time) | **Transport input** (Trinity3D) |
| `pflux` | `Fluxes/` | Particle flux (alternative location) | Transport input |
| `qflux` | `Fluxes/` | Heat flux (alternative location) | Transport input |
| `ParticleFlux_zst` | `Diagnostics/` | Zeta-resolved particle flux (stellarator) | Transport input |
| `HeatFlux_zst` | `Diagnostics/` | Zeta-resolved heat flux (stellarator) | Transport input |
| `omega_v_time` | `Special/` | Linear growth rate vs time | Screening |

GX spectral representation: Hermite-Laguerre velocity-space basis:

$$h_s = \sum_{\ell,m,k_x,k_y} \hat{h}_{s,\ell,m}(z,t)\, e^{i(k_x x + k_y y)} H_m\left(\frac{v_\parallel}{v_{ts}}\right) L_\ell\left(\frac{v_\perp^2}{v_{ts}^2}\right) F_{Ms}$$

### GENE Outputs (Alternative)

Installation-dependent filenames. Key outputs: linear growth rates, real frequencies, eigenfunctions, nonlinear species heat/particle fluxes, spectra, time histories.

### Subset Handed to Next Stage

For transport coupling, the critical handoff is the **turbulent flux vector** (steady-state heat and particle flux per species). For screening, only linear gamma and omega may be retained.

Trinity3D obtains flux Jacobians by rerunning GX on perturbed gradients and finite-differencing.

### Outputs Used as Objectives

- Linear gamma, omega: rapid screening
- Nonlinear heat flux, particle flux: high-fidelity design objectives
- Heat flux is BOTH an objective AND a transport input (dual-role output)

### Output Validation

<!-- OWNER COMPLETES: Run actual gyrokinetic calculations and verify the output fields listed above. Document:
     - Exact output format for SPECTRAX-GK: file type (HDF5, NetCDF, CSV, in-memory), field names, array shapes
     - Units and normalization conventions for gamma, omega, heat flux, particle flux
     - How to extract steady-state flux values from time traces (averaging window, convergence criteria)
     - For GX: verify NetCDF group structure and field names against actual output files
     - Any additional output fields not listed here
     - Shape discrepancies or differences between linear and nonlinear run outputs
     - How species-resolved outputs are indexed (species ordering convention) -->

## Governing Equations

Generic delta-f gyrokinetic equation:

$$\frac{\partial h_s}{\partial t} + v_\parallel \mathbf{b}\cdot\nabla h_s + \mathbf{v}_{Ds}\cdot\nabla h_s + \mathbf{v}_E\cdot\nabla h_s - C[h_s] = -\frac{Z_s e F_{Ms}}{T_s}\frac{\partial\langle\chi\rangle}{\partial t} - \mathbf{v}_\chi\cdot\nabla F_{Ms}$$

Closed by quasineutrality and (for electromagnetic calculations) appropriate field equations.

Reference: `stellarator_workflow.tex`, Section 4.7.

## Convergence & Validity

<!-- OWNER COMPLETES: Document the following after running the code:
     - What stellarator geometries are tested and known to work (e.g., Landreman-Paul QA/QH, W7-X, NCSX)
     - Linear vs. nonlinear convergence criteria: how to determine when a linear growth rate is converged, when nonlinear fluxes have reached a statistical steady state
     - Resolution requirements: velocity-space (Hermite/Laguerre modes for GX, equivalent for SPECTRAX-GK), spatial (kx, ky, z grids), and time step constraints
     - Known failure modes: geometries that cause numerical instability, parameter regimes that are problematic
     - Comparison between SPECTRAX-GK and GX results for benchmark cases (if available)
     - Cost estimates: typical wall-clock time for linear scans vs. nonlinear runs -->

## API Documentation

<!-- OWNER COMPLETES: Document the following:
     - Key entry-point functions for SPECTRAX-GK with full signatures (Python/JAX)
     - How to run a single linear calculation programmatically
     - How to run a nonlinear flux calculation programmatically
     - How to extract growth rates and fluxes from the output object
     - JAX differentiation: how to obtain gradients of gamma or flux with respect to inputs
     - For GX: Python wrapper interface (if any), or command-line invocation pattern
     - Configuration parameters and their effects on physics fidelity vs. cost -->

## Scripts & Workflows

<!-- OWNER COMPLETES: Provide the following:
     - How to run SPECTRAX-GK standalone (CLI and Python)
     - Example: linear growth rate scan over a range of ky values
     - Example: nonlinear flux calculation for a given equilibrium
     - How to convert Stage 1/2 output (wout or Boozer) into SPECTRAX-GK geometry input
     - Common debugging workflows (e.g., diagnosing non-convergence, energy conservation checks)
     - How to visualize growth rate spectra and flux time traces
     - For GX: equivalent standalone workflow examples -->

## W&B Tracking

<!-- OWNER COMPLETES: Set up and document:
     - W&B project: stellaforge-stage4-turbulence
     - What metrics to log: growth rates (gamma, omega) vs. ky, flux time traces, steady-state flux values, runtime, resolution parameters
     - Key dashboard panels: growth rate spectrum, flux convergence, cost vs. fidelity
     - Run naming convention
     - How to tag linear-only vs. nonlinear runs
     - Logging of input geometry metadata (which equilibrium, which flux surface) -->

## Container Specification (Phase 2)

<!-- OWNER COMPLETES: Define the following during Phase 2:
     - Base image and key dependencies (JAX version, GPU drivers, etc.)
     - Dockerfile entry point for SPECTRAX-GK
     - Expected volume mounts (input geometry dir, output dir, config dir)
     - Environment variables (e.g., JAX platform, GPU memory settings)
     - Resource requirements: GPU type/memory for nonlinear runs, CPU fallback for linear runs
     - For GX container: Fortran/CUDA build layer, MPI configuration
     - Multi-code container strategy: separate images per code or combined -->

## Tests (Phase 2)

<!-- OWNER COMPLETES: Write the following during Phase 2:
     - Unit tests for mathematical invariants (e.g., energy conservation in collisionless limit, flux-surface averaging properties)
     - Regression tests: known-good growth rates and fluxes for benchmark stellarator cases, compared with tolerances
     - Benchmark: compare SPECTRAX-GK results against published GX or GENE results for the same geometry
     - Integration test: Stage 1/2 geometry output can be loaded and produces valid growth rates
     - Integration test with Stage 5: SPECTRAX-GK flux output can be consumed by NEOPAX turbulence coupling (this is the critical cross-stage test -- coordinate with Stage 5 owner)
     - Acceptance criteria: definition of "done" for this stage (linear AND nonlinear capabilities) -->

## Claude Skills

<!-- OWNER COMPLETES: Create the following Claude skills:
     - Dev skill: how to run SPECTRAX-GK, interpret growth rates and fluxes, debug convergence, understand the delta-f gyrokinetic formulation, navigate the codebase
     - Operational skill: how to build the container, run the test suite, validate outputs against known-good results, set up GX as an alternative backend
     - Cross-stage skill: how to coordinate the Stage 4 -> Stage 5 handoff, especially the SPECTRAX-GK -> NEOPAX turbulence coupling interface -->
