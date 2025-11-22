import numpy as np


# CÁLCULO DE TASAS DE NODOS (η depende de ζ)
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


# TASAS DE ARISTAS DEL GRAFO (ζ)
def tasas_grafo(H, V, v, p):
    H_rates = np.where(H == 0, v*p, v*(1-p))
    V_rates = np.where(V == 0, v*p, v*(1-p))

    return H_rates, V_rates



# PROCESO COMPLETO CPDE EN Z^2
def procesoContacto(N, t_max, lamb, mu, v, p, initial_process=None, initial_graph=None):
    # Inicialización ---------------------------------------------------------------------

    # Condición inicial del grafo dinámico
    if initial_graph is not None:
        H = initial_graph["H"].copy()
        V = initial_graph["V"].copy()
    else:
        H = np.random.choice(2, size=(N, N-1), p=[1-p, p])
        V = np.random.choice(2, size=(N-1, N), p=[1-p, p])

    # Condición inicial del proceso de contacto
    if initial_process is not None:
        estados = initial_process.copy()
    else:
        estados = np.random.choice(2, (N, N), p=[0.95, 0.05])

    # Registros
    t = 0.0
    tiempos = [t]
    estados_registro = [estados.copy()]
    grafos_registro = [ {"H": H.copy(), "V": V.copy()} ]

    M_nodes = N*N
    M_H = N*(N-1)
  

    # Simulación cadena de Markov --------------------------------------------------------
    while t <= t_max:

        tasas_eta = calculo_tasas(estados, H, V, lamb, mu) # tasas η(x)
        H_rates, V_rates = tasas_grafo(H, V, v, p) # tasas de aristas
        q_total = tasas_eta.sum() + H_rates.sum() + V_rates.sum() # tasa total

        if q_total == 0:
            break

        # tiempo del siguiente salto
        dt = np.random.exponential(1/q_total)
        t += dt
        tiempos.append(t)

        # probabilidades concatenadas
        probs_nodes = tasas_eta.flatten() / q_total
        probs_H     = H_rates.flatten() / q_total
        probs_V     = V_rates.flatten() / q_total

        probs = np.concatenate([probs_nodes, probs_H, probs_V])
        idx = np.random.choice(len(probs), p=probs)

        # evento en nodos
        if idx < M_nodes:
            i, j = np.unravel_index(idx, (N, N))
            estados[i, j] = 1 - estados[i, j]

        # evento en aristas H
        elif idx < M_nodes + M_H:
            idx_H = idx - M_nodes
            i, j = np.unravel_index(idx_H, (N, N-1))
            H[i, j] = 1 - H[i, j]

        # evento en aristas V
        else:
            idx_V = idx - M_nodes - M_H
            i, j = np.unravel_index(idx_V, (N-1, N))
            V[i, j] = 1 - V[i, j]

        
        estados_registro.append(estados.copy())
        grafos_registro.append({
            "H": H.copy(),
            "V": V.copy()
        })

    return tiempos, estados_registro, grafos_registro

# ------------------------------------------------------------------------------------------ VIDEO ------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection



# reespacioar tiempos
def t_equispaceado(tiempos, estados, grafos, intervalo, t_final):

    equispaceado = np.arange(0, t_final, intervalo)

    t_asociado = []
    estados_asociado = []
    grafos_asociado = []

    i = 0  
    for t in equispaceado:

        while i < len(tiempos) and tiempos[i] <= t:
            i += 1
        
        t_asociado.append(t)
        estados_asociado.append(estados[i-1])
        grafos_asociado.append(grafos[i-1])

    return t_asociado, estados_asociado, grafos_asociado


# animacion
def animar(tiempos_salto, estados_salto, grafos_salto,
           lamb, v, p,
           intervalo=0.02, velocidad_ms=50):

    tiempos, estados, grafos = t_equispaceado(
        tiempos_salto,
        estados_salto,
        grafos_salto,
        intervalo,
        tiempos_salto[-1]
    )

    N = estados[0].shape[0]
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(estados[0], cmap="Oranges", vmin=0, vmax=1.5)
    title = ax.set_title(f"t = {tiempos[0]:.3f}, λ={lamb}, v={v}, p={p}")

    def grafo_to_lines(H, V):
        segments = []
        N = H.shape[0]

        for i in range(N):
            for j in range(N - 1):
                if H[i, j] == 1:
                    segments.append([(j, i), (j + 1, i)])

        for i in range(N - 1):
            for j in range(N):
                if V[i, j] == 1:
                    segments.append([(j, i), (j, i + 1)])

        return segments

    segments = grafo_to_lines(grafos[0]["H"], grafos[0]["V"])
    lc = LineCollection(segments, colors="black", linewidths=0.8, alpha=0.8)
    ax.add_collection(lc)

    def update(frame):
        im.set_data(estados[frame])
        title.set_text(f"t = {tiempos[frame]:.3f}, λ={lamb}, v={v}, p={p}")
        segments = grafo_to_lines(grafos[frame]["H"], grafos[frame]["V"])
        lc.set_segments(segments)
        return im, lc, title

    ani = FuncAnimation(fig, update, frames=len(tiempos),
                        interval=velocidad_ms, blit=False, repeat=False)

    ani.save("video.gif", writer="pillow", fps=60)
    plt.show()


# ------------------------------------------------------------------------------------------ EJEMPLO ------------------------------------------------------------------------------------------------

N = 40
# Infección encerrada en un cuadrado central
initial_process = np.zeros((N, N), dtype=int)
initial_process[18:22, 18:22] = 1

# Todas las aristas cerradas
H = np.ones((N, N-1), dtype=int)
V = np.ones((N-1, N), dtype=int)

if False:
    centro = N // 2
    radio_nucleo = 3   # núcleo central abierto
    grosor_anillo = 6  # grosor del anillo cerrado

    inicio_anillo = centro - radio_nucleo - grosor_anillo
    fin_anillo = centro + radio_nucleo + grosor_anillo

    # Cerrar todas las aristas dentro del anillo (bloque sólido)
    H[inicio_anillo:fin_anillo, inicio_anillo:fin_anillo-1] = 0
    V[inicio_anillo:fin_anillo-1, inicio_anillo:fin_anillo] = 0

initial_graph = {"H": H, "V": V}

lamb = 0.5
v=1
p=0.009

tiempos, estados_registro, grafos_registro = procesoContacto(
    N=40, t_max=5, lamb=lamb, mu=1, v=v, p=p, initial_graph=initial_graph, initial_process=initial_process
)

#print(tiempos[-1], estados_registro[-1], grafos_registro[-1])

animar(tiempos, estados_registro, grafos_registro, lamb, v, p,
       intervalo=0.02, velocidad_ms=20)
