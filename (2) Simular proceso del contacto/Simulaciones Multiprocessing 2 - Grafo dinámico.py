import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from datetime import datetime
# cálculo local de tasa de un vértice
def calculo_tasas(estados, H, V, lamb, mu):
    N = estados.shape[0]
    # Número de vecinos infectados conectados
    n = np.zeros_like(estados, dtype=int) 

    n[:, 1:] += estados[:, :-1] * H # izquierda (j>=1)
    n[:, :-1] += estados[:, 1:] * H # derecha (j<=N-2)
    n[1:, :] += estados[:-1, :] * V # arriba (i>=1)
    n[:-1, :] += estados[1:, :] * V # abajo (i<=N-2)

    # Tasa de infección
    tasas = lamb * n

    # Sitios infectados curan con tasa mu
    infectados = estados.astype(bool)
    tasas[infectados] = mu

    return tasas

# tasas de aristas del grafo para percolación dinámica
def tasas_grafo(H, V, v, p):
    H_rates = np.where(H == 0, v*p, v*(1-p))
    V_rates = np.where(V == 0, v*p, v*(1-p))
    return H_rates, V_rates

# simulación vectorizada de un batch con grafo dinámico (percolación)
import numpy as np
from multiprocessing import Pool, cpu_count

# Función de simulación para un batch
def simulacion_batch_percolacion(args):

    batch_grids, N, t_max, lamb, mu, v, p, batch_H, batch_V = args
    K = batch_grids.shape[0]

    # tiempo y procesos activos
    t = np.zeros(K)
    activos = np.ones(K, dtype=bool)

    H = batch_H.copy()
    V = batch_V.copy()

    M_nodes = N*N
    M_H = N*(N-1)
    M_V = (N-1)*N
    M_total = M_nodes + M_H + M_V

    # reshape común
    rng = np.random.default_rng()

    while activos.any():

        idx = np.where(activos)[0]
        B = len(idx)

        est = batch_grids[idx]
        H_act = H[idx]
        V_act = V[idx]

        # ---------------------
        # 1. VECINOS (vectorizado)
        # ---------------------
        n = np.zeros_like(est, dtype=int)
        n[:, :, 1:]  += est[:, :, :-1] * H_act
        n[:, :, :-1] += est[:, :, 1:]  * H_act
        n[:, 1:, :]  += est[:, :-1, :] * V_act
        n[:, :-1, :] += est[:, 1:, :]  * V_act

        tasas = lamb * n
        tasas[est == 1] = mu

        # ---------------------
        # 2. TASAS ARISTAS (vectorizado)
        # ---------------------
        H_rates = np.where(H_act == 0, v*p, v*(1-p))
        V_rates = np.where(V_act == 0, v*p, v*(1-p))

        # ---------------------
        # 3. TASAS TOTALES
        # ---------------------
        q_nodes = tasas.reshape(B, -1).sum(axis=1)
        q_H = H_rates.reshape(B, -1).sum(axis=1)
        q_V = V_rates.reshape(B, -1).sum(axis=1)

        q = q_nodes + q_H + q_V

        # si no hay eventos → muere
        zero = (q == 0)
        if zero.any():
            activos[idx[zero]] = False
            idx = np.where(activos)[0]
            if len(idx) == 0:
                break
            est = batch_grids[idx]
            H_act = H[idx]
            V_act = V[idx]
            tasas = tasas[~zero]
            H_rates = H_rates[~zero]
            V_rates = V_rates[~zero]
            q = q[~zero]
            B = len(q)

        # ---------------------
        # 4. MUESTRA TIEMPOS
        # ---------------------
        dt = rng.exponential(1 / q)
        t[idx] += dt

        # cortar por t_max
        activos[t >= t_max] = False
        if not activos.any():
            break

        # ---------------------
        # 5. GENERAR EVENTOS (vectorizado)
        # ---------------------
        probs_nodes = tasas.reshape(B, -1) / q[:, None]
        probs_H = H_rates.reshape(B, -1) / q[:, None]
        probs_V = V_rates.reshape(B, -1) / q[:, None]

        probs = np.concatenate([probs_nodes, probs_H, probs_V], axis=1)

        u = rng.random(B)
        idx_event = np.argmax(u[:, None] < np.cumsum(probs, axis=1), axis=1)

        # ---------------------
        # 6. APLICAR EVENTOS (vectorizado)
        # ---------------------
        # nodos
        mask_nodes = idx_event < M_nodes
        if mask_nodes.any():
            e = idx_event[mask_nodes]
            ii = e // N
            jj = e % N
            batch_grids[idx[mask_nodes], ii, jj] ^= 1

        # aristas horizontales
        mask_H = (idx_event >= M_nodes) & (idx_event < M_nodes + M_H)
        if mask_H.any():
            e2 = idx_event[mask_H] - M_nodes
            ii = e2 // (N-1)
            jj = e2 % (N-1)
            H[idx[mask_H], ii, jj] ^= 1

        # aristas verticales
        mask_V = idx_event >= (M_nodes + M_H)
        if mask_V.any():
            e3 = idx_event[mask_V] - (M_nodes + M_H)
            ii = e3 // N
            jj = e3 % N
            V[idx[mask_V], ii, jj] ^= 1

        # ---------------------
        # 7. CRITERIO DE EXTINCIÓN
        # ---------------------
        infectados = batch_grids.reshape(K, -1).sum(axis=1)
        activos &= (infectados < 0.786 * N**2)

    # ------------------------
    # SALIDA
    # ------------------------
    estados_fin = [batch_grids[k] for k in range(K)]
    grafos_fin = [{"H": H[k], "V": V[k]} for k in range(K)]
    return t, estados_fin, grafos_fin


# Función principal que usa multiprocessing
def procesoContacto_paralelo(K, N, t_max, lamb, mu, v, p, initial_process=None, initial_graph=None, batch_size=50):
    if initial_process is None:
        initial_process = np.random.choice(2, (K, N, N), p=[0.95,0.05])
    if initial_graph is None:
        batch_H = np.random.choice(2, size=(K, N, N-1), p=[1-p, p])
        batch_V = np.random.choice(2, size=(K, N-1, N), p=[1-p, p])
    else:
        batch_H = initial_graph["H"].copy()
        batch_V = initial_graph["V"].copy()

    args_list = [(initial_process[k:k+batch_size],
                  N, t_max, lamb, mu, v, p,
                  batch_H[k:k+batch_size], batch_V[k:k+batch_size])
                 for k in range(0, K, batch_size)]

    with Pool(cpu_count()) as pool:
        resultados = pool.map(simulacion_batch_percolacion, args_list)

    tiempos = np.concatenate([r[0] for r in resultados])
    estados_fin = sum([r[1] for r in resultados], [])
    grafos_fin = sum([r[2] for r in resultados], [])
    return tiempos, estados_fin, grafos_fin


# Función principal que usa multiprocessing
def procesoContacto_paralelo(K, N, t_max, lamb, mu, v, p, initial_process=None, initial_graph=None, batch_size=50):
    if initial_process is None:
        initial_process = np.random.choice(2, (K, N, N), p=[0.95,0.05])
    if initial_graph is None:
        batch_H = np.random.choice(2, size=(K, N, N-1), p=[1-p, p])
        batch_V = np.random.choice(2, size=(K, N-1, N), p=[1-p, p])
    else:
        batch_H = initial_graph["H"].copy()
        batch_V = initial_graph["V"].copy()

    args_list = [(initial_process[k:k+batch_size],
                  N, t_max, lamb, mu, v, p,
                  batch_H[k:k+batch_size], batch_V[k:k+batch_size])
                 for k in range(0, K, batch_size)]
    
    n_batches = len(args_list)
    resultados = []
    with Pool(cpu_count()) as pool:
        for i, r in enumerate(pool.imap(simulacion_batch_percolacion, args_list), 1):
            resultados.append(r)

            # imprimir progreso con hora
            ahora = datetime.now().strftime("%H:%M:%S")
            print(f"[{ahora}] Batch {i}/{n_batches} completado")


    tiempos = np.concatenate([r[0] for r in resultados])
    estados_fin = sum([r[1] for r in resultados], [])
    grafos_fin = sum([r[2] for r in resultados], [])
    return tiempos, estados_fin, grafos_fin


# estimación de probabilidad de sobrevivencia
def estimar_prob_sobrevivencia(N, t_max, lamb, mu, n_corridas, v, p, initial_process=None, initial_graph=None):
    tiempos, estados_fin, grafos_fin = procesoContacto_paralelo(
        n_corridas, N, t_max, lamb, mu, v, p, initial_process, initial_graph
    )
    return sum(estado.any() for estado in estados_fin) / n_corridas # promediamos con unos en caso de que un estado final tenga al menos un uno



# estimación del lambda crítico 
def estimar_lambda_critico_positivo(N, t_max, mu, v,p, lamb_low, lamb_high, n_corridas, tol, initial_process=None, initial_graph=None):
    while lamb_high - lamb_low > tol: # usamos búsqueda binaria hasta cierta tolerancia
        lamb_mid = 0.5 * (lamb_low + lamb_high)
        p_mid = estimar_prob_sobrevivencia(N, t_max, lamb_mid, mu, n_corridas, v, p, initial_process, initial_graph)
        
        if p_mid == 0:
            lamb_low  = lamb_mid
        else:
            lamb_high = lamb_mid

        print(f"λ_mid={lamb_mid}, p̂={p_mid}")

    return lamb_high


# ------------------------------------------------------------------------------------------ EJEMPLO ------------------------------------------------------------------------------------------------


if __name__ == "__main__":


    K = 900
    N = 80
    v = 0.1
    lambdas = [0.75, 1.4]
    t_max = 20
    mu = 1

    # Infección encerrada en un cuadrado central
    initial_process = np.zeros((K, N, N), dtype=int)
    m = N//2
    initial_process[:,m, m] = 1

    # Todas las aristas abiertas
    H = np.ones((K, N, N-1), dtype=int)
    V = np.ones((K, N-1, N), dtype=int)

    initial_graph = {"H": H, "V": V}

    n_corridas = K  # usamos K simulaciones en paralelo

    curvas = {}
    p_aristas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    probs = []
    print(f"\n=== n_corridas = {n_corridas}, N = {N} ===")
    for lamb in lambdas:
        for p in p_aristas:
            prob = estimar_prob_sobrevivencia(
                N, t_max, lamb, mu, 
                n_corridas, v, p,
                initial_process, initial_graph
            )
            print(f"λ={lamb}, p≈{p}, prob≈{prob}, tmax≈{t_max}")
            probs.append(prob)

    curvas[(n_corridas, N)] = probs

    print(curvas)