import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

# cálculo local de tasa de un vértice
def tasa_vertice(estados, i, j, lamb, mu):
    if estados[i, j] == 1:
        return mu
    N = estados.shape[0]
    n = 0 # contamos vecinos
    if i > 0     and estados[i-1, j] == 1: n += 1
    if i < N-1   and estados[i+1, j] == 1: n += 1
    if j > 0     and estados[i, j-1] == 1: n += 1
    if j < N-1   and estados[i, j+1] == 1: n += 1
    return lamb * n


#  simulación vectorizada de un batch con actualización local
def simulacion_batch_local(args):
    batch_grids, N, t_max, lamb, mu = args
    K_batch = batch_grids.shape[0]

    t = np.zeros(K_batch) # registramos tiempos por simulación
    alive = np.ones(K_batch, dtype=bool) # indicatrices de que un proceso esté activo
    infect_now = batch_grids.reshape(K_batch, -1).sum(axis=1) # cantidad de infectados inicial

    # construcción inicial de tasas
    tasas = np.zeros((K_batch, N, N), dtype=float)
    for k in range(K_batch):
        for i in range(N):
            for j in range(N):
                tasas[k,i,j] = tasa_vertice(batch_grids[k], i, j, lamb, mu)

    while alive.any():
        total_q = tasas.reshape(K_batch, -1).sum(axis=1)
        active = alive & (total_q > 0)
        if not active.any(): break

        # actualizamos tiempos por cadena
        dt = np.zeros(K_batch)
        dt[active] = np.random.exponential(1 / total_q[active])
        t += dt
        alive &= (t < t_max) # si una simulación se pasa del t_máximo, se deja de actualizar
        if not alive.any(): break

        # seleccionamos vértice a cambiar
        active_idx = np.where(active)[0]
        u_active = np.random.random(len(active_idx)) * total_q[active_idx]
        cdf_active = np.cumsum(tasas[active_idx].reshape(len(active_idx), -1), axis=1)
        e_active = (cdf_active >= u_active[:, None]).argmax(axis=1)

        i = e_active // N
        j = e_active % N

        # actualizamos el vértice seleccionado
        for idx, ii, jj in zip(active_idx, i, j):
            old = batch_grids[idx, ii, jj]
            batch_grids[idx, ii, jj] ^= 1
            infect_now[idx] += (1 - old) - old

            # actualización local de tasas
            vecinos = [(ii,jj),(ii-1,jj),(ii+1,jj),(ii,jj-1),(ii,jj+1)]
            for a,b in vecinos:
                if 0 <= a < N and 0 <= b < N:
                    tasas[idx,a,b] = tasa_vertice(batch_grids[idx], a, b, lamb, mu)

        alive &= (infect_now < 0.786 * N**2) # si una simulación tiene una cantidad mayor de infectados, se asume que sigue creciente y se deja de actualizar

    estados_fin = [batch_grids[k] for k in range(K_batch)] # guardamos todas los estados finales
    return t, estados_fin


# multiprocessing por batches
def procesoContacto_paralelo(K, N, t_max, lamb, mu, initial_process=None, batch_size=50):
    # si no se da condición inicial, genera una aleatoria en la grilla
    if initial_process is None:
        initial_process = np.random.choice(2, (K, N, N), p=[0.95,0.05])

    # dividimos la simulación en múltiples simulaciones más pequeñas
    args_list = [(initial_process[k:k+batch_size], N, t_max, lamb, mu) for k in range(0, K, batch_size)] 

    with Pool(cpu_count()) as p:
        resultados = p.map(simulacion_batch_local, args_list)

    # concatenamos resultados
    tiempos = np.concatenate([r[0] for r in resultados])
    estados_fin = sum([r[1] for r in resultados], [])
    return tiempos, estados_fin


# estimación de probabilidad de sobrevivencia
def estimar_prob_sobrevivencia(N, t_max, lamb, mu, n_corridas, initial_process=None):
    tiempos, estados_fin = procesoContacto_paralelo(
        n_corridas, N, t_max, lamb, mu, initial_process
    )
    return sum(estado.any() for estado in estados_fin) / n_corridas # promediamos con unos en caso de que un estado final tenga al menos un uno



# estimación del lambda crítico 
def estimar_lambda_critico_positivo(N, t_max, mu, lamb_low, lamb_high, n_corridas, tol, initial_process=None):
    while lamb_high - lamb_low > tol: # usamos búsqueda binaria hasta cierta tolerancia
        lamb_mid = 0.5 * (lamb_low + lamb_high)
        p_mid = estimar_prob_sobrevivencia(N, t_max, lamb_mid, mu, n_corridas, initial_process)
        
        if p_mid == 0:
            lamb_low  = lamb_mid
        else:
            lamb_high = lamb_mid

        print(f"λ_mid={lamb_mid:.4f}, p̂={p_mid:.3f}")

    return lamb_high


# --------------------------------------------------------- CÓDIGO PRINCIPAL ----------------------------------------------------------------------

if __name__ == "__main__":

    N = 100
    n_corridas = 2000
    t_max = 240
    mu = 1.0

    # condición inicial con un único infectado en el centro
    initial_process = np.zeros((n_corridas, N, N), dtype=int)
    medio = N // 2
    initial_process[:, medio:medio+1, medio:medio+1] = 1

    
    lambdas = np.linspace(1.02, 1.8, 26)
    curvas = {}

    estimar_lambda_critico_positivo(N, t_max, mu, 0.38, 0.43, n_corridas, tol = 1e-3, initial_process=initial_process)

    probs = []
    print(f"\n=== n_corridas = {n_corridas}, N = {N} ===")
    # estimamos probabilidad para cada lambda
    for lamb in lambdas:
        p = estimar_prob_sobrevivencia(N, t_max, lamb, mu, n_corridas=n_corridas, initial_process=initial_process)
        print(f"λ={lamb}, p≈{p}")
        probs.append(p)
    curvas[(n_corridas, N)] = probs

    # plotear lambdas
    plt.figure(figsize=(10, 6))
    for (n_corridas, N), probs in curvas.items():
        plt.plot(lambdas, probs, marker='o', linewidth=2, label=f"N={N}, {n_corridas} simulaciones")

    plt.xlabel(r"$\lambda$")
    plt.ylabel("Probabilidad de sobrevivir hasta t_max")
    plt.grid(True)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.show()
