"""
Reconstruct Elmer mesh.boundary from tetrahedral mesh geometry.
Classifies external faces by z-coordinate:
  BC 1 = bottom  (z ~ 0)
  BC 2 = top     (z ~ Lz = 0.02)
  BC 3 = sides
Updates mesh.header with the correct boundary-element count.
"""

from collections import defaultdict

MESH_DIR = "elmer_mesh"
Lz = 0.02
tol = 1e-6   # absolute tolerance for z-plane membership

# ── 1. Read nodes ──────────────────────────────────────────────────────────────
# Format: node_id  dummy  x  y  z
nodes = {}
with open(f"{MESH_DIR}/mesh.nodes") as f:
    for line in f:
        parts = line.split()
        if len(parts) == 5:
            nid = int(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            nodes[nid] = (x, y, z)

print(f"Nodes read: {len(nodes)}")
z_vals = [v[2] for v in nodes.values()]
print(f"  z range: {min(z_vals):.6f} – {max(z_vals):.6f}   (expect 0 – {Lz})")

# ── 2. Read volume elements (all tet4 = type 504) ─────────────────────────────
# Format: elem_id  body_id  504  n1 n2 n3 n4
elements = {}
with open(f"{MESH_DIR}/mesh.elements") as f:
    for line in f:
        parts = line.split()
        if len(parts) == 7 and parts[2] == "504":
            eid = int(parts[0])
            nids = (int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]))
            elements[eid] = nids

print(f"Volume elements (tet4): {len(elements)}")

# ── 3. Find external faces ─────────────────────────────────────────────────────
# Each tet4 has 4 triangular faces.
# A face is external if it appears in exactly one tetrahedron.
# key = frozenset of 3 node IDs → list of (elem_id, ordered_triple)
face_map = defaultdict(list)

TET_FACES = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]

for eid, (n0, n1, n2, n3) in elements.items():
    ns = [n0, n1, n2, n3]
    for i, j, k in TET_FACES:
        triple = (ns[i], ns[j], ns[k])
        key = frozenset(triple)
        face_map[key].append((eid, triple))

external_faces = {k: v[0] for k, v in face_map.items() if len(v) == 1}
print(f"External (boundary) faces: {len(external_faces)}")

# ── 4. Classify each external face ────────────────────────────────────────────
def classify(triple):
    zs = [nodes[n][2] for n in triple]
    if all(z < tol for z in zs):
        return 1   # bottom
    if all(abs(z - Lz) < tol for z in zs):
        return 2   # top
    return 3       # side

bc_counts = defaultdict(int)
boundary_rows = []   # (be_id, bc_id, parent_eid, 0, 303, n1, n2, n3)

for be_id, (face_key, (parent_eid, triple)) in enumerate(external_faces.items(), start=1):
    bc = classify(triple)
    bc_counts[bc] += 1
    boundary_rows.append((be_id, bc, parent_eid, 0, 303, triple[0], triple[1], triple[2]))

print(f"\nBC classification:")
print(f"  BC 1 (bottom, z=0):    {bc_counts[1]:4d} triangles")
print(f"  BC 2 (top,    z={Lz}): {bc_counts[2]:4d} triangles")
print(f"  BC 3 (sides):          {bc_counts[3]:4d} triangles")
print(f"  Total boundary elems:  {len(boundary_rows)}")

# ── 5. Write mesh.boundary ─────────────────────────────────────────────────────
# Format: be_id  bc_id  parent1  parent2  elem_type  n1 n2 n3
with open(f"{MESH_DIR}/mesh.boundary", "w") as f:
    for row in boundary_rows:
        be_id, bc_id, p1, p2, etype, n1, n2, n3 = row
        f.write(f"{be_id} {bc_id} {p1} {p2} {etype} {n1} {n2} {n3}\n")

print(f"\nmesh.boundary written ({len(boundary_rows)} lines)")

# ── 6. Update mesh.header ──────────────────────────────────────────────────────
# Format:
#   nNodes  nVolElems  nBoundaryElems
#   nElementTypes
#   elemType  count
#   [elemType  count ...]
n_nodes = len(nodes)
n_vol = len(elements)
n_bnd = len(boundary_rows)

with open(f"{MESH_DIR}/mesh.header", "w") as f:
    f.write(f"{n_nodes} {n_vol} {n_bnd}\n")
    f.write("2\n")              # two element types
    f.write(f"504 {n_vol}\n")   # tet4 volume
    f.write(f"303 {n_bnd}\n")   # tri3 boundary

print(f"mesh.header updated: {n_nodes} nodes, {n_vol} vol elems, {n_bnd} bnd elems")
print(f"\n>>> TOP_BCID = 2   BOTTOM_BCID = 1 <<<")
