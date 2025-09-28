import sympy as sp

# Declearing symbols
h = sp.Symbol("h", positive=True, real=True)
m = sp.Symbol("m", positive=True, real=True)
N = sp.Symbol("N", positive=True, real=True)
V = sp.Symbol("V", positive=True, real=True)
T = sp.Symbol("T", positive=True, real=True)
k = sp.Symbol("k", real=True)
k_B = sp.Symbol("k_B", positive=True, real=True)
r_0 = sp.Symbol("r_0", positive=True, real=True)
beta = sp.Symbol("beta", positive=True, real=True)

# Single canonical partition function (not expressed with beta)
Z_1 = (16*V*(m**3)*((T*k_B*sp.pi)**(sp.Rational(9/2)))/((k**sp.Rational(3,2))*(h**(6)))
      *((k*(r_0**2))/(k_B*T) + 2)
      *(1 + sp.sqrt(1 - sp.exp(-(k*(r_0**2))/(4*k_B*T)))
      *(1 + sp.Rational(277, 2000)*sp.exp(-(k*(r_0**2))/(4*k_B*T))))
      )

# Single canonical partition function expressed with beta
Z_1_beta = ((1/h**6)*16*V*(m**3)*((sp.pi/beta)**(sp.Rational(9, 2)))
           *((beta*k*(r_0**2) + 2)/(k**sp.Rational(3, 2)))
           *(1 + sp.sqrt(1 - sp.exp(-sp.Rational(1,4)*beta*k*(r_0**2)))
           *(1 + sp.Rational(277, 2000)*sp.exp(-sp.Rational(1,4)*beta*k*(r_0**2))))
           )

# Logarithm of many particle partition function
ln_Z = (N*sp.ln(Z_1) - N*sp.ln(N) + N)
ln_Z_beta = (N*sp.ln(Z_1_beta) - N*sp.ln(N) + N)

# Helmotz free energy
A = - k_B*T*ln_Z

S = - sp.diff(A, T)
P = - sp.diff(A, V)
mu = sp.diff(A, N)
C_V = -T*sp.diff(A, T, T)
U = -sp.diff(ln_Z_beta, beta)

print(f"\nA:\n{sp.latex(sp.simplify(A))}\n")
print(f"\nS:\n{sp.latex(sp.simplify(S))}\n")
print(f"\nP:\n{sp.latex(sp.simplify(P))}\n")
print(f"\nmu:\n{sp.latex(sp.simplify(mu))}\n")
print(f"\nC_V:\n{sp.latex(sp.simplify(C_V))}\n")
print(f"\nU:\n{sp.latex(sp.simplify(U))}\n")