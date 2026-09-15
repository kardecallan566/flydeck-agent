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
    l1_l2 = [circuit.neurons[index] for index in circuit.l1_inputs + circuit.l2_inputs]
    print("MaleCNS visual motion circuit")
    print(f"neurons: {len(circuit.neurons)}")
    print(f"edges: {len(circuit.edges)}")
    print(f"spatial mapping: {circuit.spatial_mode}")
    print(f"L1 sensory neurons: {len(circuit.l1_inputs)}")
    print(f"L2 sensory neurons: {len(circuit.l2_inputs)}")
    if l1_l2:
        xs = [n.spatial_x for n in l1_l2 if n.spatial_x is not None]
        ys = [n.spatial_y for n in l1_l2 if n.spatial_y is not None]
        print(f"L1/L2 spatial coverage: {len(xs)}/{len(l1_l2)}")
        print(f"L1/L2 x range: {min(xs):.3f}..{max(xs):.3f}")
        print(f"L1/L2 y range: {min(ys):.3f}..{max(ys):.3f}")
    print(f"T4 [a,b,c,d]: {[len(pool) for pool in circuit.t4_outputs]}")
    print(f"T5 [a,b,c,d]: {[len(pool) for pool in circuit.t5_outputs]}")
    print("cell types:")
    for cell_type, count in sorted(counts.items()):
        print(f"  {cell_type}: {count}")
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
