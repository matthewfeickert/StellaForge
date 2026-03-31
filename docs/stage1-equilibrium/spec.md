# Stage 1: Equilibrium

## Overview

Stage 1 solves the three-dimensional ideal-MHD equilibrium problem, producing the magnetic field geometry and flux-surface profiles that all downstream stages depend on. This is the entry point of the forward-pass pipeline.

**Physics:** Given a plasma boundary shape and profile guesses (pressure, rotational transform or current), find the 3D magnetic equilibrium satisfying force balance: $\nabla p = \mathbf{J} \times \mathbf{B}$, $\nabla \cdot \mathbf{B} = 0$.

**Position in pipeline:** This stage has no upstream dependencies. Its output (`wout_*.nc`) is consumed by Stage 2 (Boozer Transform) and also directly by some turbulence and transport codes.

Reference: `stellarator_workflow/stellarator_workflow.tex`, Section 4.1 (VMEC++ and vmec_jax) and Section 4.2 (DESC).

## Codes

### vmec_jax (Primary JAX)
- **Repository:** https://github.com/uwplasma/vmec_jax
- **Language:** Python/JAX
- **Role:** JAX-native implementation providing differentiable equilibrium solving with wout-compatible output

### VMEC++ (C++ Alternative)
- **Repository:** https://github.com/proximafusion/vmecpp
- **Documentation:** https://proximafusion.github.io/vmecpp/
- **Language:** C++ with Python bindings
- **Role:** From-scratch C++ reimplementation of VMEC. Solves fixed- and free-boundary ideal-MHD equilibria. Preserves the standard `wout` downstream contract.

### DESC (Differentiable Alternative)
- **Repository:** https://github.com/PlasmaControl/DESC
- **Language:** Python/JAX
- **Role:** Differentiable pseudo-spectral equilibrium and optimization suite. Can replace VMEC++ as the equilibrium engine and also perform some downstream computations (Boozer transform, geometry objectives) internally.

### Installation & Platform

> **OWNER COMPLETES:** Document installation instructions for vmec_jax (primary), version requirements, known platform issues, and any dependency conflicts. Include setup for VMEC++ and DESC as secondary/alternative codes if applicable.

## Input Specification

### Physical Inputs

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `RBC(m,n)` | 2D array (float) | Boundary R cosine Fourier coefficients | User-specified |
| `ZBS(m,n)` | 2D array (float) | Boundary Z sine Fourier coefficients | User-specified |
| `AM` | 1D array (float) | Pressure profile coefficients | User-specified |
| `AM_AUX_*` | arrays (float) | Auxiliary pressure arrays (alternative to AM) | User-specified |
| `AI` | 1D array (float) | Rotational transform iota coefficients (if iota-prescribed) | User-specified |
| `AC` | 1D array (float) | Current profile coefficients (if current-prescribed) | User-specified |
| `AC_AUX_*` | arrays (float) | Auxiliary current arrays | User-specified |
| `PHIEDGE` | scalar (float) | Total toroidal flux (magnetic scale) | User-specified |

### Resolution & Solver Controls

| Field | Type | Description |
|-------|------|-------------|
| `NS` | int or array | Number of radial grid points (can be a multi-grid sequence) |
| `MPOL` | int | Maximum poloidal mode number |
| `NTOR` | int | Maximum toroidal mode number |
| `NITER` / `NITER_ARRAY` | int / array | Iteration budgets |
| `FTOL` / `FTOL_ARRAY` | float / array | Convergence tolerances |

### Input Formats
- **INDATA files:** Fortran-style text `input.NAME` format (vmec_jax and VMEC++)
- **JSON:** Programmatic input (VMEC++ only)
- **Python objects:** In-memory API (both VMEC++ and vmec_jax)
- **Hot restart:** Previous converged output state as initial guess

### Input Validation

> **OWNER COMPLETES:** After running the code, validate the input tables above against actual code behavior. Note any fields the TeX spec missed, any deprecated fields, and any code-specific quirks (e.g., fields that are silently ignored, default values that differ between codes, required vs. optional fields).

## Output Specification

### Primary Output: `wout_*.nc` (NetCDF)

#### Geometry Scalars

| Field | Type | Description | Used As |
|-------|------|-------------|---------|
| `aspect` | scalar (float) | Aspect ratio R/a | Objective |
| `Aminor_p` | scalar (float) | Minor radius | Geometry |
| `Rmajor_p` | scalar (float) | Major radius | Geometry |
| `volume_p` | scalar (float) | Plasma volume | Objective |
| `betatotal` | scalar (float) | Total plasma beta | Objective |
| `b0` | scalar (float) | Magnetic field on axis | Geometry |
| `volavgB` | scalar (float) | Volume-averaged |B| | Geometry |
| `fsqr` | scalar (float) | Force residual (radial) | QA signal |
| `fsqz` | scalar (float) | Force residual (vertical) | QA signal |
| `fsql` | scalar (float) | Force residual (lambda) | QA signal |

#### Radial Profiles

| Field | Type | Description |
|-------|------|-------------|
| `presf` | 1D array | Pressure on full mesh |
| `pres` | 1D array | Pressure on half mesh |
| `phi` | 1D array | Toroidal flux |
| `phipf` | 1D array | d(phi)/ds on full mesh |
| `chi` | 1D array | Poloidal flux |
| `chipf` | 1D array | d(chi)/ds on full mesh |
| `iotas` | 1D array | Rotational transform on half mesh |
| `iotaf` | 1D array | Rotational transform on full mesh |
| `q_factor` | 1D array | Safety factor (1/iota) |
| `jcuru` | 1D array | Poloidal current density |
| `jcurv` | 1D array | Toroidal current density |
| `buco` | 1D array | Covariant B_theta (Boozer I) |
| `bvco` | 1D array | Covariant B_zeta (Boozer G) |

#### Spectral Geometry

| Field | Type | Description |
|-------|------|-------------|
| `rmnc` | 2D array (ns x mnmax) | R cosine Fourier coefficients |
| `zmns` | 2D array (ns x mnmax) | Z sine Fourier coefficients |
| `lmns` | 2D array (ns x mnmax) | Lambda sine Fourier coefficients |
| `bmnc` | 2D array (ns x mnmax) | |B| cosine Fourier coefficients |
| `gmnc` | 2D array (ns x mnmax) | Jacobian sqrt(g) cosine coefficients |
| `bsubumnc` | 2D array (ns x mnmax) | B_theta cosine coefficients |
| `bsubvmnc` | 2D array (ns x mnmax) | B_zeta cosine coefficients |
| `bsubsmns` | 2D array (ns x mnmax) | B_s sine coefficients |
| `currumnc` | 2D array (ns x mnmax) | J_theta cosine coefficients |
| `currvmnc` | 2D array (ns x mnmax) | J_zeta cosine coefficients |

#### Python API Objects (vmec_jax / VMEC++)

| Object | Description |
|--------|-------------|
| `wout` | Full wout data structure |
| `threed1_volumetrics` | 3D volume integrals |
| `jxbout` | J x B force-balance diagnostics |
| `mercier` | Mercier stability criterion |

### Subset Handed to Next Stage

Stage 2 (BOOZ_XFORM / booz_xform_jax) needs the **full** equilibrium spectrum and profiles in `wout_*.nc`. GX, Trinity3D, and NEOPAX geometry readers also consume wout-level data for field-line geometry, rotational transform, and surface metrics.

### Outputs Used as Objectives

- Aspect ratio, volume, beta, target iota(s): direct design objectives
- Mercier criterion: stability objective
- Residuals `fsqr`, `fsqz`, `fsql`: QA convergence signals, not physics design objectives

### Output Validation

> **OWNER COMPLETES:** Run an actual equilibrium case and verify the output fields listed above exist in the resulting `wout_*.nc` file. Note any missing fields, shape discrepancies, additional fields not listed here, or differences between vmec_jax and VMEC++ outputs.

## Governing Equations

The equilibrium satisfies ideal-MHD force balance:

$$\nabla p = \mathbf{J} \times \mathbf{B}, \quad \nabla \cdot \mathbf{B} = 0, \quad \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}$$

VMEC++ finds the stationary point of the energy functional (Hirshman & Whitson 1983):

$$W = \frac{1}{(2\pi)^2} \int \left( \frac{B^2}{2} + \frac{p}{\gamma - 1} \right) dV$$

In VMEC++ flux coordinates with the stream function lambda:

$$u = \theta + \lambda(s, \theta, \zeta), \quad \frac{du}{d\zeta} = \iota(s)$$

The contravariant field components are:

$$B^\zeta = \frac{\Phi'(s) + \text{lamscale} \cdot \partial_\theta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

$$B^\theta = \frac{\chi'(s) - \text{lamscale} \cdot \partial_\zeta \lambda}{\text{signgs} \cdot \sqrt{g} \cdot 2\pi}$$

DESC solves the same physics in a pseudo-spectral formulation:

$$\mathbf{B} = \frac{\partial_\rho \psi}{2\pi\sqrt{g}} \left[ \left(\iota - \frac{\partial\lambda}{\partial\zeta}\right) \mathbf{e}_\theta + \left(1 + \frac{\partial\lambda}{\partial\theta}\right) \mathbf{e}_\zeta \right]$$

Reference: `stellarator_workflow.tex`, Sections 4.1-4.2.

## Convergence & Validity

> **OWNER COMPLETES:** Document the following after running the code:
> - What input configurations are known to converge (e.g., specific stellarator configs like Landreman-Paul QA/QH)
> - What configurations are known to fail or diverge
> - Convergence criteria and tolerances (what `fsqr`/`fsqz`/`fsql` values indicate convergence)
> - Known numerical edge cases (e.g., very high aspect ratio, free-boundary issues)
> - Multi-grid strategy: which NS sequences work well

## API Documentation

> **OWNER COMPLETES:** Document the following:
> - Key entry-point functions with full signatures (vmec_jax primary, VMEC++ secondary)
> - Configuration parameters and their effects
> - Programmatic usage examples (Python/JAX)
> - How to do a hot restart from a previous converged state
> - How to extract specific fields from the output object

## Scripts & Workflows

> **OWNER COMPLETES:** Provide the following:
> - How to run the code standalone (CLI and Python)
> - Example scripts with sample input data (e.g., a simple stellarator case)
> - Common debugging workflows (e.g., what to check when convergence fails)
> - How to visualize equilibrium results

## W&B Tracking

> **OWNER COMPLETES:** Set up and document:
> - W&B project: `stellaforge-stage1-equilibrium`
> - What metrics to log: convergence residuals (`fsqr`, `fsqz`, `fsql`) vs iteration, runtime, key geometry scalars (aspect, volume, beta), iota profile
> - Key dashboard panels
> - Run naming convention

## Container Specification (Phase 2)

> **OWNER COMPLETES:** Define the following during Phase 2:
> - Base image and key dependencies (JAX version, etc.)
> - Dockerfile entry point
> - Expected volume mounts (input dir, output dir)
> - Environment variables
> - Resource requirements (CPU/GPU, memory)

## Tests (Phase 2)

> **OWNER COMPLETES:** Write the following during Phase 2:
> - Unit tests for mathematical invariants (e.g., force balance residual should decrease monotonically)
> - Regression tests against known-good wout files (compare key fields with tolerances)
> - Integration test: output `wout_*.nc` can be read by booz_xform_jax (Stage 2)
> - Acceptance criteria: definition of "done" for this stage

## Claude Skills

> **OWNER COMPLETES:** Create the following Claude skills:
> - Dev skill: how to run vmec_jax, debug convergence failures, interpret wout fields, understand the physics
> - Operational skill: how to build the container, run the test suite, validate outputs against known-good results
