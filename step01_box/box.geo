SetFactory("OpenCASCADE");

// Box dimensions (meters)
Lx = 0.04;
Ly = 0.04;
Lz = 0.02;

// Mesh size (meters)
lc = 0.004;

Box(1) = {0, 0, 0, Lx, Ly, Lz};

eps = 1e-9;
top[]    = Surface In BoundingBox{-eps, -eps, Lz-eps, Lx+eps, Ly+eps, Lz+eps};
bottom[] = Surface In BoundingBox{-eps, -eps, -eps,   Lx+eps, Ly+eps, eps};
sides[]  = Surface In BoundingBox{-eps, -eps, -eps,   Lx+eps, Ly+eps, Lz+eps};

// Force physical tags (numeric IDs)
Physical Volume(1) = {1};
Physical Surface(101) = {top[]};
Physical Surface(102) = {bottom[]};
Physical Surface(103) = {sides[]};

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

