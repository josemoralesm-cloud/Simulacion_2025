import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from datetime import datetime

# simulación vectorizada de un batch con grafo dinámico (percolación)
import numpy as np
from multiprocessing import Pool, cpu_count

def simulacion_batch_percolacion(args):
    batch_grids, N, t_max, lamb, mu, v, p, batch_H, batch_V = args
    K_batch = batch_grids.shape[0]  # tamaño real del batch

    # tiempo y procesos activos
    t = np.zeros(K_batch)
    activos = np.ones(K_batch, dtype=bool)

    H = batch_H.copy()
    V = batch_V.copy()

    M_nodes = N * N
    M_H = N * (N - 1)
    M_V = (N - 1) * N
    max_infectados = batch_grids.reshape(K_batch, -1).sum(axis=1).copy()

    rng = np.random.default_rng()

    while activos.any():
        idx = np.where(activos)[0]
        B = len(idx)
        est = batch_grids[idx]
        H_act = H[idx]
        V_act = V[idx]

        # vecinos
        n = np.zeros_like(est, dtype=int)
        n[:, :, 1:]  += est[:, :, :-1] * H_act
        n[:, :, :-1] += est[:, :, 1:]  * H_act
        n[:, 1:, :]  += est[:, :-1, :] * V_act
        n[:, :-1, :] += est[:, 1:, :]  * V_act

        tasas = lamb * n
        tasas[est == 1] = mu

        # aristas
        H_rates = np.where(H_act == 0, v*p, v*(1-p))
        V_rates = np.where(V_act == 0, v*p, v*(1-p))

        # tasas totales
        q_nodes = tasas.reshape(B, -1).sum(axis=1)
        q_H = H_rates.reshape(B, -1).sum(axis=1)
        q_V = V_rates.reshape(B, -1).sum(axis=1)
        q = q_nodes + q_H + q_V

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

        # sampleo de tiempos considerando todas las tasas
        dt = rng.exponential(1 / q)

        activos[t >= t_max] = False
        if not activos.any():
            break

        too_late = (t[idx] + dt >= t_max)
        if too_late.any():
            activos[idx[too_late]] = False
            continue

        # generar eventos
        probs_nodes = tasas.reshape(B, -1) / q[:, None]
        probs_H = H_rates.reshape(B, -1) / q[:, None]
        probs_V = V_rates.reshape(B, -1) / q[:, None]
        probs = np.concatenate([probs_nodes, probs_H, probs_V], axis=1)

        u = rng.random(B)
        idx_event = np.argmax(u[:, None] < np.cumsum(probs, axis=1), axis=1)

        # aplicar eventos
        mask_nodes = idx_event < M_nodes
        if mask_nodes.any():
            e = idx_event[mask_nodes]
            ii = e // N
            jj = e % N
            batch_grids[idx[mask_nodes], ii, jj] ^= 1

        mask_H = (idx_event >= M_nodes) & (idx_event < M_nodes + M_H)
        if mask_H.any():
            e2 = idx_event[mask_H] - M_nodes
            ii = e2 // (N-1)
            jj = e2 % (N-1)
            H[idx[mask_H], ii, jj] ^= 1

        mask_V = idx_event >= (M_nodes + M_H)
        if mask_V.any():
            e3 = idx_event[mask_V] - (M_nodes + M_H)
            ii = e3 // N
            jj = e3 % N
            V[idx[mask_V], ii, jj] ^= 1

        # criterio de extincion
        infectados = batch_grids.reshape(K_batch, -1).sum(axis=1)
        max_infectados = np.maximum(max_infectados, infectados)
        muertos = (infectados == 0)
        sobreviven = activos & (~muertos)

        idx_rel = np.where(activos)[0]
        rel_survive = np.where(~muertos[activos])[0]
        t[idx_rel[rel_survive]] += dt[rel_survive]

        activos &= ~muertos
        activos &= (infectados < 0.786 * N**2)

    estados_fin = [batch_grids[k] for k in range(K_batch)]
    grafos_fin = [{"H": H[k], "V": V[k]} for k in range(K_batch)]

    return t, estados_fin, grafos_fin, max_infectados




# función principal que usa multiprocessing
def procesoContacto_paralelo(K, N, t_max, lamb, mu, v, p, initial_process=None, initial_graph=None, batch_size=50, imprimir = False):
    if initial_process is None:
        initial_process = np.random.choice(2, (K, N, N), p=[0.95,0.05])
    if initial_graph is None:
        batch_H = np.random.choice(2, size=(K, N, N-1), p=[1-p, p])
        batch_V = np.random.choice(2, size=(K, N-1, N), p=[1-p, p])
    else:
        batch_H = initial_graph["H"].copy()
        batch_V = initial_graph["V"].copy()

    args_list = [
        (initial_process[k:k+batch_size],
        N, t_max, lamb, mu, v, p,
        batch_H[k:k+batch_size], batch_V[k:k+batch_size])
        for k in range(0, K, batch_size)
        if initial_process[k:k+batch_size].size > 0
    ]

    
    n_batches = len(args_list)
    resultados = []
    with Pool(cpu_count()) as pool:
        for i, r in enumerate(pool.imap(simulacion_batch_percolacion, args_list), 1):
            resultados.append(r)

            # imprimir progreso con hora
            if imprimir:
                ahora = datetime.now().strftime("%H:%M:%S")
                print(f"[{ahora}] Batch {i}/{n_batches} completado")


    tiempos = np.concatenate([r[0] for r in resultados if len(r[0]) > 0])
    estados_fin = sum([r[1] for r in resultados if len(r[1]) > 0], [])
    grafos_fin = sum([r[2] for r in resultados if len(r[2]) > 0], [])
    maximos = np.concatenate([r[3] for r in resultados if len(r[3]) > 0])
    return tiempos, estados_fin, grafos_fin, maximos



# aceptación rechazo para simular condicional a estar en un estrato dado
def simular_en_estrato(N, t_max, lamb, mu, v, p, estrato, initial_process=None, initial_graph=None, m_por_estrato=90, batch_size=15):
    a, b = estrato
    while True:
        tiempos, estados, _, maximos = procesoContacto_paralelo(
            K= m_por_estrato,
            N= N,
            t_max= t_max,
            lamb= lamb,
            mu= mu,
            v= v,
            p= p,
            initial_process= initial_process,
            initial_graph= initial_graph,
            batch_size= batch_size
        )
        mask = (maximos >= a) & (maximos < b)
        if mask.any():
            idx = np.where(mask)[0]
            return tiempos[idx], [estados[i] for i in idx]

def estimar_prob_sobrevivencia_estratificada(
    N, t_max, lamb, mu, v, p,
    estratos=[(0,5), (5,10), (10,15), (15,20), (20, float('inf'))],
    m_por_estrato=90,
    initial_process=None,
    initial_graph=None
):
    # corrida para determinar oesis
    pilot_K = 90
    tiempos_pilot, estados_pilot, _, maximos_pilot = procesoContacto_paralelo(
        pilot_K, N, t_max, lamb, mu, v, p, initial_process, initial_graph, batch_size=15
    )

    pesos = np.array([
        np.mean((maximos_pilot >= a) & (maximos_pilot < b))
        for (a,b) in estratos
    ])


    print("pesos pilot", pesos)

    # filtrar estratos activos
    indices_activos = np.where(pesos > 0)[0]
    estratos_activos = [estratos[i] for i in indices_activos]
    pesos_activos = pesos[indices_activos]

    pesos_activos = pesos_activos / np.sum(pesos_activos)

    print("estratos activos:", estratos_activos)
    print("pesos activos:", pesos_activos)

    # solo consideramos estratos activos
    p_k_activos = []

    for (a,b) in estratos_activos:
        sobrevivio = 0
        total = 0

        while total < m_por_estrato:
            tiempos, estados = simular_en_estrato(
                N, t_max, lamb, mu, v, p,
                estrato=(a,b),
                initial_process=initial_process,
                initial_graph=initial_graph,
                m_por_estrato=m_por_estrato,
                batch_size=15
            )
            total += len(tiempos)
            sobrevivio += sum(np.any(est) for est in estados)
            #print((a,b),sobrevivio, total)

        ahora = datetime.now().strftime("%H:%M:%S")
        print(f"[{ahora}] Estrato {(a,b)} completado, p = {p}, lambda = {lamb}")
        p_k_activos.append(sobrevivio / total)

    p_k_activos = np.array(p_k_activos)


    # combinar con pesos
    p_final = np.sum(pesos_activos * p_k_activos)
    estimador = p_final  

    p_k_completo = np.zeros(len(estratos))
    p_k_completo[indices_activos] = p_k_activos

    pesos_completos = np.zeros(len(estratos))
    pesos_completos[indices_activos] = pesos_activos

    return p_final, pesos_completos, p_k_completo, estimador

def estimar_lambda_critico_por_p(N, t_max, mu, v, p_values, lamb_low, lamb_high, n_corridas, tol, initial_process=None, initial_graph=None):
    lambda_criticos = {}

    for p in p_values:
        l_low = lamb_low
        l_high = lamb_high

        if p <= 0.3:
            estratos = [(0,4), (4,60), (60, np.inf)]
        elif p <= 0.6:
            estratos = [(0,4), (4,80), (80, np.inf)]
        else:
            estratos = [(0,2), (3,8), (8,15), (15,25), (25, np.inf)]

        while l_high - l_low > tol:
            l_mid = 0.5 * (l_low + l_high)
            
            # estimación con método estratificado
            p_mid, _, _, _ = estimar_prob_sobrevivencia_estratificada(
                N, t_max, l_mid, mu, v, p,
                estratos=estratos,
                m_por_estrato=90,
                initial_process=initial_process,
                initial_graph=initial_graph
            )
            
            if p_mid < 1e-8:
                l_low = l_mid
            else:
                l_high = l_mid

            print(f"p={p}, λ_mid={l_mid}, p̂={p_mid}")

        lambda_criticos[p] = l_high

    return lambda_criticos



if __name__ == "__main__":


    K = 200
    N = 70
    v = 0.1
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
    p_aristas = [0.5]

    if False:
        probs = []
        print(f"\n=== n_corridas = {n_corridas}, N = {N} ===")
        for lamb in lambdas:
            for p in p_aristas:
                prob, pesos, p_k , estimador = estimar_prob_sobrevivencia_estratificada(
                    N, t_max, lamb, mu, v, p, 
                    estratos=[(0,5), (5,10), (10,20), (20,50), (50, float('inf'))],
                    m_por_estrato=120,
                    initial_process=initial_process,
                    initial_graph=initial_graph
                )
                print("===========================================================")
                print(f"λ={lamb}, p≈{p}, prob≈{prob}, tmax≈{t_max}, pk = {p_k}, estimador = {estimador}")
                print("===========================================================")
                probs.append(prob)

        curvas[(n_corridas, N)] = probs

    print(curvas)

    p = estimar_lambda_critico_por_p(N, t_max, mu, v, p_aristas, 0.7, 0.9, n_corridas, 1e-2, initial_process=initial_process, initial_graph=initial_graph)
    p