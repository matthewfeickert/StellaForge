# Potential Issues

## Stage 1 -- Equilibrium

- [ ] vmec/vmec_jax and DESC do not have directly compatible inputs; an adapter or input translation layer will be needed to support both implementations behind the same pipeline entry point
- [ ] vmec_jax only consumes a subset of the full VMEC INDATA file; need to document which fields are supported/ignored, or validate inputs to warn when unsupported fields are present
- [ ] DESC can output Boozer coordinates directly, so with the right flag/argument it can handle both Stage 1 and Stage 2; the pipeline should support this shortcut path

## Stage 2 -- Boozer Transform

- [ ] Future boundary condition optimization can be added as additional functions in Stage 2

## Stage 3 -- Neoclassical

- [ ] sfincs/sfincs_jax and MONKES do not have directly compatible inputs; same adapter/translation issue as Stage 1

## Stage 4 -- Turbulence

- [ ] SPECTRAX-GK/GX, and GENE likely do not have directly compatible inputs; same adapter/translation issue as Stages 1 and 3
