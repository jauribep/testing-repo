from dolfin import *
print("Running FEniCS...")
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "P", 1)
u = interpolate(Constant(1.0), V)
with XDMFFile("f.xdmf") as f:
    f.write(u)
