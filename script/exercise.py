L = 64
D = 4096
F = 4 * D
V = 32000

# N=K, NH=D
# L * (3DF + 2D(N+K)H) + DV + 2D
# L * (3DF + (2D)^2) + DV + 2D
def calculate_params(L, D, F, V):
    return ((L * (3 * D * F + 4 * D**2 + 2 * D)) + D * V) 

# 2D(N+K)H = 2D(2D)H = 4D^2
# L * (4D^2) / params
def attention_params(L, D, params):
    attention_params = L * (4 * D**2)
    return attention_params / params

# KV cache[2, S, L, K, H]
# token per cahce = 2LKH = 2LD 
def kv_cache(L, D):
    return 2 * L * D


params = calculate_params(L, D, F, V)
print("params: ", int(params / 1e9), "B")
attention = attention_params(L, D, params)
print(f"attention params: {attention:.2g}", "%")
kvs = kv_cache(L, D)
print("KV cache params: ", kvs / 1024, "KiB")

# import math
# print(math.log2(1024))
# print(2**20 / 1024)