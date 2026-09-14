from __future__ import annotations

import argparse
from pathlib import Path

from .visual_circuit import VisualCircuit
from .visual_diagnostics import format_diagnostic, run_motion_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the extracted MaleCNS visual motion circuit")
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument("--steps", default=12, type=int)
    args = parser.parse_args()

    circuit = VisualCircuit.load(args.circuit)
    if not circuit.has_spatial_mapping:
        raise SystemExit("circuit has no L1/L2 retinotopic mapping")

    print("MaleCNS controlled visual-motion diagnostic")
    print(f"neurons: {len(circuit.neurons)}")
    print(f"edges: {len(circuit.edges)}")
    print(f"spatial mode: {circuit.spatial_mode}")
    print()
    for diagnostic in run_motion_suite(circuit, steps=args.steps):
        print(format_diagnostic(diagnostic))
    print()
    print("T4/T5 order: a=front-to-back, b=back-to-front, c=up, d=down")
    print("These are circuit responses only; no BNB labels are used by this diagnostic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
