# MVP Pipeline: I/O Reference

```
 INDATA                   wout_*.nc                boozmn_*.nc
   |                          |                         |
   v                          v                         v
┌────────┐  NetCDF  ┌──────────────┐  NetCDF  ┌──────────────────────┐
│Stage 1 │ -------> │   Stage 2    │ -------> │      Stage 3         │
│vmec_jax│ wout_*.nc│booz_xform_jax│boozmn_*.nc│ SFINCS       monkes │
└────────┘          └──────────────┘          └───┬──────────────┬───┘
                          |                       |              |
                          |                       v              v
                          |              sfincsOutput.h5      D_ij.h5
                          |                  (HDF5)           (HDF5)
                          |                       |              |
                          |    ┌────────────┐     |              |
                          +--->│  Stage 4   │     |              |
                     wout_*.nc │ SPECTRAX-GK│     |              |
                               └─────┬──────┘     |              |
                                     |             |              |
                                flux (CSV)         |              |
                                     |             |              |
                                     v             v              v
                     ┌───────────────────────────────────────────────┐
                     │                  Stage 5                     │
                     │            Transport / Profiles              │
                     │  NEOPAX (wout + boozmn + D_ij + turb flux)   │
                     └──────────────────┬──────────────────────────┘
                                        |
                                        v
                                   profiles.h5
                              n(r), T(r), E_r(r), P_fus, Q
```

## MVP Test Data

Input configs, committed reference outputs, and runtime outputs under `mvp/`:

```
mvp/
├── stage1-equilibrium/     expected_input/ + expected_output/ + (runtime)input/ + (runtime)output/
├── stage2-boozer/          example.py + expected_output/ + (runtime)output/
├── stage3-neoclassical/    expected_input/ + expected_output/ + run_monkes.py + (runtime)input/ + (runtime)output/
├── stage4-turbulence/      expected_input/ + expected_output/ + (runtime)input/ + (runtime)output/
└── stage5-transport/       run_NEOPAX.py + expected_output/ + (runtime)output/
```

`expected_input/` and `expected_output/` hold the tracked reference configs and reference outputs. `input/` and `output/` are gitignored runtime locations -- the pipeline reads from `input/` and writes to `output/`. Use `pixi run initialize-example-inputs` to seed `input/` from `expected_input/` (optional; users may populate or modify `input/` directly). The task skips any stage whose `input/` dir is already populated. To re-seed a stage (e.g. after `expected_input/` changes upstream), wipe that stage's `input/` dir and re-run the task. Cross-stage configs reference upstream `output/`, so run stages in forward-chain order (`stage-1-vmec` first).

---

**Note:** All paths are relative to the repository root.

## Stage 1 -- Equilibrium

**Code:** vmec_jax

| Direction                     | Format                                    | Location                                                                        |
| ----------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| **In**                        | Fortran-style Text                        | `mvp/stage1-equilibrium/input/input.HSX_QHS_vacuum_ns201`              |
| **Out**                       | NetCDF `wout_*.nc` (similar to hdf5 file) | `mvp/stage1-equilibrium/output/wout_HSX_QHS_vacuum_ns201.nc`  |
| **Additional Out** (optional) | Text (terminal output)                    | `mvp/stage1-equilibrium/output/optional_terminal_output.vmec` |

> [!NOTE]
> `HSX_QHS_vacuum_ns201` is an example name. This can be changed. As can the entirety of the name `optional_terminal_output.vmec`.

### How to Install

From inside the `mvp/` directory

```
pixi install --environment stage-1-vmec
```

### How to Run

```
pixi run stage-1-vmec
```

> [!NOTE]
> Populate `stage1-equilibrium/input/` from the tracked `expected_input/` via `pixi run initialize-example-inputs` (optional) or manually before running.

---

## Stage 2 -- Boozer Transform

**Code:** booz_xform_jax

| Direction | Format               | Location                                                                          |
| --------- | -------------------- | --------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc`   | `mvp/stage1-equilibrium/output/wout_HSX_QHS_vacuum_ns201.nc`    |
| **Out**   | NetCDF `boozmn_*.nc` | `mvp/stage2-boozer/output/boozmn_HSX_QHS_vacuum_ns201.nc` |

> [!NOTE]
> Stage 2's driver reads wout from `stage1-equilibrium/output/`. Populate this directory by running `pixi run stage-1-vmec`, or by copying the reference wout from `stage1-equilibrium/expected_output/`.

### How to Install

```
pixi install --environment stage-2-booz
```

### How to Run

```
pixi run stage-2-booz
```

which is morally similar to

```python
import booz_xform_jax as bx
b=bx.Booz_xform()
b.read_wout("wout_HSX_QHS_vacuum_ns201.nc")
b.run()
b.write_boozmn("boozmn_HSX_QHS_vacuum_ns201.nc")
```

---

## Stage 3 -- Neoclassical (two parallel sub-stages)

**Code:** sfincs_jax

| Direction | Format                       | Location                                                                       |
| --------- | ---------------------------- | ------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`           | `mvp/stage1-equilibrium/output/wout_HSX_QHS_vacuum_ns201.nc` |
| **In**    | Fortran-style Text `input.*` | `mvp/stage3-neoclassical/input/input.HSX_QHS_vacuum_ns201`          |
| **Out**   | HDF5 `sfincsOutput.h5`       | `mvp/stage3-neoclassical/output/sfincsOutput.h5`           |

#### How to Install

```
pixi install --environment stage-3-sfincs
```

#### How to Run

```
pixi run stage-3-sfincs
```

> [!NOTE]
> The Stage 3 namelist reads wout from `stage1-equilibrium/output/`. Populate this directory by running `pixi run stage-1-vmec`, or by copying the reference wout from `stage1-equilibrium/expected_output/`.

> [!NOTE]
> Populate `stage3-neoclassical/input/` from the tracked `expected_input/` via `pixi run initialize-example-inputs` (optional) or manually before running.


**code:** Monkes

| Direction | Format               | Location                                                                                         |
| --------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **In**    | NetCDF `wout_*.nc`   | `mvp/stage1-equilibrium/output/wout_HSX_QHS_vacuum_ns201.nc`                   |
| **In**    | NetCDF `boozmn_*.nc` | `mvp/stage2-boozer/output/boozmn_HSX_QHS_vacuum_ns201.nc`                |
| **Out**   | HDF5 `D_ij.h5`       | `mvp/stage3-neoclassical/output/Monoenergetic_database_VMEC_s_coordinate_HSX.h5` |

> [!NOTE]
> The inputs come from both Stage 1 and Stage 2.

#### How to Install
```bash
git clone https://github.com/eduardolneto/monkes.git
cd monkes
pip install .
```

#### How to Run

Monkes is a little more involved. See `mvp/stage3-neoclassical/run_monkes.py`

We basically call it inside a python loop to use Monkes to generate a database across different radial positions, electric fields, and collisionality, but it uses the same 2 input files for the entire loop.

---

## Stage 4 -- Turbulence

**Code:** SPECTRAX-GK

| Direction | Format             | Location                                                                           |
| --------- | ------------------ | ---------------------------------------------------------------------------------- |
| **In**    | NetCDF `wout_*.nc` | `mvp/stage1-equilibrium/output/wout_HSX_QHS_vacuum_ns201.nc`     |
| **In**    | TOML config        | `mvp/stage4-turbulence/input/runtime_hsx_nonlinear_vmec_geometry.toml` |
| **Out**   | Summary `JSON`     | `mvp/stage4-turbulence/output/hsx_run.summary.json`           |
| **Out**   | `csv`              | `mvp/stage4-turbulence/output/hsx_run.diagnostics.csv`        |

> [!NOTE]
> The TOML's `vmec_file` points into `stage1-equilibrium/output/`. Populate this directory by running `pixi run stage-1-vmec`, or by copying the reference wout from `stage1-equilibrium/expected_output/`.

### How to Install

```
pixi install --environment stage-4-spectrax
```

### How to Run

```
pixi run stage-4-spectrax
```

> [!NOTE]
> Populate `stage4-turbulence/input/` from the tracked `expected_input/` via `pixi run initialize-example-inputs` (optional) or manually before running.

which executes something morally equivalent to

```
spectrax-gk run --config runtime_hsx_nonlinear_vmec_geometry.toml --out output/hsx_run
```

---

## Stage 5 -- Transport Solver

**Code:** NEOPAX

| Direction                                 | Format               | Location                                                                                         |
| ----------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **In**                                    | NetCDF `wout_*.nc`   | `mvp/stage1-equilibrium/output/wout_HSX_QHS_vacuum_ns201.nc`                   |
| **In**                                    | NetCDF `boozmn_*.nc` | `mvp/stage2-boozer/output/boozmn_HSX_QHS_vacuum_ns201.nc`                |
| **In** (Only needed if `monkes` is used.) | HDF5 `D_ij.h5`       | `mvp/stage3-neoclassical/output/Monoenergetic_database_VMEC_s_coordinate_HSX.h5` |
| **Out**                                   | HDF5 `profiles_*.h5` | `mvp/stage5-transport/output/NEOPAX_output.h5`                                   |
> [!NOTE]
> The inputs come from Stage 1, Stage 2, and Stage 3.

### How to Install

```
pixi install --environment stage-5-neopax
```

### How to Run

This is run inside a script. See `mvp/stage5-transport/run_NEOPAX.py` as a reference.

> [!NOTE]
> `NEOPAX`, being the final stage, has additional complexities and variabilities in the workflow.
>
> For example, if `monkes` is used, then the script using `NEOPAX` won't constantly loop since `monkes` provides a database and does the loop before reaching Stage 5.
>
> If `sfincs_jax` is used, then ideally, the script using NEOPAX runs a loop to optimize over different fluxes. While ideal, this is most computationally expensive.

---
