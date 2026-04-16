from pathlib import Path

file_path = Path(__file__).resolve()
input_file_dir = file_path.parent.parent / "stage1-equilibrium" / "output"
output_file_dir = file_path.parent / "output"
output_file_dir.mkdir(exist_ok=True)

import booz_xform_jax as bx

b = bx.Booz_xform()
b.read_wout(input_file_dir / "wout_HSX_QHS_vacuum_ns201.nc")
b.run()
b.write_boozmn(output_file_dir / "boozmn_HSX_QHS_vacuum_ns201.nc")
