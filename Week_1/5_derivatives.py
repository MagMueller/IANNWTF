import sympy as sym

x, z, a, b = sym.symbols('x z a b')
f = 4*a*x**2 + b**2*(1 - b)**2 + z*(1 - z) + 6
# calculate gradient of f
result_x = f.diff(x)
result_z = f.diff(z)
result_a = f.diff(a)
result_b = f.diff(b)

print("")
print("Function: {}\n".format(f))
print("gradient w.r.t x:    {}".format(result_x))
print("gradient w.r.t z:    {}".format(result_z))
print("gradient w.r.t a:    {}".format(result_a))
print("gradient w.r.t b:    {}".format(result_b))
print("Done")
