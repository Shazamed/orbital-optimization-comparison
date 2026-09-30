import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[name] = "1"

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


from pathlib import Path
import numpy as np
from itertools import product, repeat

from qibochem.ansatz.ups import Ansatz_tUPS
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.scripts.script_utils import load_molecule

SCRIPT_DIR = Path(__file__).resolve().parent


oo_opt_methods = {
    "classical_oo-vqe_alt" : {
        'oo_layers': 0,
    },
    "quantum_oo-vqe" : {
        'oo_layers': 3,
    },
    "classical_oo-vqe_comb" : {
        'oo_layers': 0,
    },
} 


def run_opt(layers, method, molecule, gtol, runs=51):
    molecule_name = molecule
    num_active_e = 6
    num_active_o = 6
    oo_layers = oo_opt_methods[method]["oo_layers"]

    output_path = (
        SCRIPT_DIR / "data"
        / f"{molecule_name}-L{layers}-{method}-{gtol}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for trial in range(runs):
        # Fresh molecule: orbital optimization modifies it.
        mol = load_molecule(
            SCRIPT_DIR / "molecules" / f"{molecule_name}.xyz",
            num_active_e,
            num_active_o,
            orbitals="canonical",
        )

        if trial == 0:
            use_rand = False
        else:
            use_rand = True

        tups = Ansatz_tUPS(
            mol=mol,
            layers=layers,
            oo_layers=oo_layers,
            use_random_angles=use_rand,
            use_mp2_guess=False,
            use_projection=True,
            use_mat_mul=True,
            perfect_pair=True,
            ref_bitstring="110011001100",
            mo_perm=None,
            use_small_perturb_angles=False,
        )
        if method == "classical_oo-vqe_alt":
            tups.run_oo_vqe_alternating(options={"gtol": gtol}, loops=1000)
        elif method == "quantum_oo-vqe":
            tups.run_oo_vqe_quantum(options={"gtol": gtol})
        elif method == "classical_oo-vqe_comb":
            tups.run_oo_vqe_combined(options={"gtol": gtol})
        else:
            raise ValueError("Method not defined")
        
        tups.export_summary(
            molecule_name,
            basis="sto-3g",
            active_e=num_active_e,
            active_o=num_active_o,
            output_path=output_path,
        )

    print(
        f"{molecule_name}-L{layers}-{method}-{gtol} finished",
        flush=True,
    )

    return str(output_path)



if __name__ == "__main__":
    # methods_names = ["quantum_oo-vqe","classical_oo-vqe_alt"]
    methods_names = ["classical_oo-vqe_comb"]
    molecules_names = ["H6"]
    layers_range = range(1, 5)
    jobs = list(product(layers_range,methods_names,molecules_names))
    layers,methods,molecules = zip(*jobs)

    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=get_context("spawn"),
    ) as pool:
        for output_file in pool.map(run_opt,
                                    layers,
                                    methods,
                                    molecules,
                                    repeat(1e-5),
                                    repeat(51)
                                    ):
            print(f"Completed: {output_file}")
