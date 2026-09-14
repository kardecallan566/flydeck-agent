from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .visual_circuit import VisualCircuitBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MaleCNS visual motion subnetwork")
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--neurotransmitters", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/malecns_visual.json"), type=Path)
    args = parser.parse_args()

    builder = VisualCircuitBuilder(args.annotations, args.weights, args.neurotransmitters)
    circuit = builder.build(args.output)
    counts = Counter(neuron.cell_type for neuron in circuit.neurons)
    print("MaleCNS visual motion circuit")
    print(f"neurons: {len(circuit.neurons)}")
    print(f"edges: {len(circuit.edges)}")
    print(f"L1 sensory neurons: {len(circuit.l1_inputs)}")
    print(f"L2 sensory neurons: {len(circuit.l2_inputs)}")
    print(f"T4 [a,b,c,d]: {[len(pool) for pool in circuit.t4_outputs]}")
    print(f"T5 [a,b,c,d]: {[len(pool) for pool in circuit.t5_outputs]}")
    print("cell types:")
    for cell_type, count in sorted(counts.items()):
        print(f"  {cell_type}: {count}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
