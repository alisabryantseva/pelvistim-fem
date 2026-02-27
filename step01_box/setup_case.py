import math
from pathlib import Path

meshdir = Path("elmer_mesh")
nodes_path = meshdir / "mesh.nodes"
bnd_path   = meshdir / "mesh.boundary"

if not nodes_path.exists() or not bnd_path.exists():
    raise SystemExit("ERROR: run this in the folder that contains elmer_mesh/mesh.nodes and elmer_mesh/mesh.boundary")

# --- Read nodes: {node_id: (x,y,z)} ---
nodes = {}
with nodes_path.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        parts = line.split()
        try:
            nid = int(parts[0])
        except ValueError:
            continue
        # Elmer nodes lines can be 4+ cols; coords are always the last 3
        x, y, z = map(float, parts[-3:])
        nodes[nid] = (x, y, z)

if not nodes:
    raise SystemExit("ERROR: parsed 0 nodes from elmer_mesh/mesh.nodes")

z_all = [p[2] for p in nodes.values()]
zmin_global = min(z_all)
zmax_global = max(z_all)
Lz = zmax_global - zmin_global

# --- Map (Elmer element type) -> number of nodes (common ones) ---
etype_nnodes = {
    202: 2,    # line2
    203: 3,    # line3
    303: 3,    # tri3
    306: 6,    # tri6
    404: 4,    # tet4
    408: 8,    # tet8-ish / quad8 (rare)
    504: 6,    # prism6
    510: 10,   # prism10
    808: 8,    # hex8 (rare for boundaries)
}

# --- Aggregate z-ranges per boundary-id (2nd column in mesh.boundary) ---
stats = {}  # bcid -> {count, zmin, zmax}
unknown_lines = 0

with bnd_path.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        try:
            bcid = int(parts[1])
            etype = int(parts[4])   # Elmer boundary line: id bc_id tag1 tag2 etype nodes...
        except (ValueError, IndexError):
            continue

        nn = etype_nnodes.get(etype)

        node_ids = []
        if nn is not None and len(parts) >= nn:
            # Most robust: node ids are last nn entries
            cand = parts[-nn:]
            try:
                node_ids = [int(x) for x in cand]
            except ValueError:
                node_ids = []
        if not node_ids:
            # Fallback: take from the right, grabbing ids that exist as nodes, stop once we have 3 or 4
            for tok in reversed(parts):
                try:
                    v = int(tok)
                except ValueError:
                    continue
                if v in nodes:
                    node_ids.append(v)
                    if len(node_ids) >= 4:
                        break
            node_ids = list(reversed(node_ids))

        coords = [nodes.get(nid) for nid in node_ids if nid in nodes]
        if len(coords) < 3:
            unknown_lines += 1
            continue

        zvals = [c[2] for c in coords]
        zmin = min(zvals)
        zmax = max(zvals)

        s = stats.setdefault(bcid, {"count": 0, "zmin": float("inf"), "zmax": float("-inf")})
        s["count"] += 1
        s["zmin"] = min(s["zmin"], zmin)
        s["zmax"] = max(s["zmax"], zmax)

if not stats:
    raise SystemExit("ERROR: could not parse any boundary elements from elmer_mesh/mesh.boundary")

# --- classify top/bottom: boundary must be FLAT at the extreme (zmin == zmax == extreme) ---
tol = max(1e-10, 1e-6 * (1.0 if Lz == 0 else Lz))  # scale-aware tolerance
top_ids = []
bot_ids = []

for bcid, s in sorted(stats.items()):
    flat_at_top = (abs(s["zmin"] - zmax_global) <= tol and abs(s["zmax"] - zmax_global) <= tol)
    flat_at_bot = (abs(s["zmin"] - zmin_global) <= tol and abs(s["zmax"] - zmin_global) <= tol)
    if flat_at_top:
        top_ids.append(bcid)
    if flat_at_bot:
        bot_ids.append(bcid)

print(f"Global zmin={zmin_global:.6g}, zmax={zmax_global:.6g}, Lz={Lz:.6g}, tol={tol:.3g}")
print("Boundary ID summary (bcid: count, zmin, zmax):")
for bcid, s in sorted(stats.items()):
    print(f"  {bcid:4d}: {s['count']:6d}  zmin={s['zmin']:.6g}  zmax={s['zmax']:.6g}")

print("\nDetected TOP boundary IDs:", top_ids)
print("Detected BOTTOM boundary IDs:", bot_ids)
if unknown_lines:
    print(f"(Note: {unknown_lines} boundary lines couldn't be parsed cleanly; usually OK.)")

if not top_ids or not bot_ids:
    raise SystemExit("ERROR: Could not confidently detect top/bottom boundary IDs. See summary above.")

def fmt_target(ids):
    ids = sorted(set(ids))
    if len(ids) == 1:
        return f"Target Boundaries = {ids[0]}"
    return f"Target Boundaries({len(ids)}) = " + " ".join(map(str, ids))

case = f'''Header
  CHECK KEYWORDS Warn
  Mesh DB "." "elmer_mesh"
  Include Path ""
  Results Directory "results"
End

Simulation
  Max Output Level = 5
  Coordinate System = Cartesian 3D
  Coordinate Mapping(3) = 1 2 3
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 1
End

Constants
  Permittivity of Vacuum = 8.8542e-12
End

Body 1
  Target Bodies(1) = 1
  Name = "body"
  Equation = 1
  Material = 1
End

Equation 1
  Name = "Conduction"
  Active Solvers(2) = 1 2
End

Solver 1
  Equation = "Static Current Conduction"
  Procedure = "StatCurrentSolve" "StatCurrentSolver"
  Variable = "Potential"
  Variable DOFs = 1

  Calculate Volume Current = True

  Linear System Solver = Direct
  Linear System Direct Method = UMFPACK

  Steady State Convergence Tolerance = 1.0e-8
End

Solver 2
  Equation = "ResultOutput"
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = "case"
  Output Format = VTU
  VTU Format = Logical True
  Save Geometry IDs = Logical True
End

Material 1
  Name = "Tissue"
  Electric Conductivity = 0.2
End

Boundary Condition 1
  Name = "top"
  {fmt_target(top_ids)}
  Potential = 1.0
End

Boundary Condition 2
  Name = "bottom"
  {fmt_target(bot_ids)}
  Potential = 0.0
End
'''
Path("case.sif").write_text(case)
print("\nWrote case.sif with detected boundary IDs.")
