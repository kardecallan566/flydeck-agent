from __future__ import annotations

import argparse
from pathlib import Path

from .visual_circuit import VisualCircuitBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MaleCNS visual motion subnetwork")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--neurotransmitters", type=Path)
    parser.add_argument("--output", default=Path("data/malecns_visual.json"), type=Path)
    args = parser.parse_args()

    builder = VisualCircuitBuilder(args.annotations, args.weights, args.neurotransmitters)
    circuit = builder.build(args.output)
    print("MaleCNS visual circuit")
    print(f"neurons: {len(circuit.neurons)}")
    print(f"edges: {len(circuit.edges)}")
    print(f"ON input pools: {sum(bool(pool) for pool in circuit.on_inputs)}/4")
    print(f"OFF input pools: {sum(bool(pool) for pool in circuit.off_inputs)}/4")
    print(f"T4 outputs: {[len(pool) for pool in circuit.t4_outputs]}")
    print(f"T5 outputs: {[len(pool) for pool in circuit.t5_outputs]}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
