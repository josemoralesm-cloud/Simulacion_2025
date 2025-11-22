import numpy as np


# ============================================================
#           Fenwick Tree (Binary Indexed Tree)
# ============================================================

class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.data = np.zeros(n + 1, dtype=float)

    def build(self, arr):
        for i in range(self.n):
            self.data[i+1] = arr[i]
        for i in range(1, self.n+1):
            j = i + (i & -i)
            if j <= self.n:
                self.data[j] += self.data[i]

    def prefix_sum(self, i):
        s = 0.0
        while i > 0:
            s += self.data[i]
            i -= (i & -i)
        return s

    def total(self):
        return self.prefix_sum(self.n)

    def update(self, i, delta):
        i += 1
        while i <= self.n:
            self.data[i] += delta
            i += (i & -i)

    def find_by_prefix(self, x):
        i = 0
        bitmask = 1 << (self.n.bit_length() - 1)
        while bitmask != 0:
            t = i + bitmask
            if t <= self.n and self.data[t] < x:
                x -= self.data[t]
                i = t
            bitmask >>= 1
        return i



# ============================================================
#         Cálculo local de tasas y actualizaciones
# ============================================================

def tasa_nodo(estados, i, j, lamb, mu, N):
    if estados[i, j] == 1:
        return mu

    n = 0
    if i > 0     and estados[i-1, j] == 1: n += 1
    if i < N-1   and estados[i+1, j] == 1: n += 1
    if j > 0     and estados[i, j-1] == 1: n += 1
    if j < N-1   and estados[i, j+1] == 1: n += 1
    return lamb * n



# ============================================================
#               Proceso de contacto (optimizado)
# ============================================================



def procesoContactoTrayectoria(N, t_max, lamb, mu, initial_grid=None):

    M = N * N

    # grid inicial
    if initial_grid is not None:
        grid = initial_grid.copy()
    else:
        grid = np.random.choice(2, (N, N), p=[0.95, 0.05])

    # almacenamiento de trayectoria
    times = []
    states = []

    # cálculo inicial de tasas
    tasas = np.zeros(M, dtype=float)
    for i in range(N):
        for j in range(N):
            tasas[i*N + j] = tasa_nodo(grid, i, j, lamb, mu, N)

    ft = FenwickTree(M)
    ft.build(tasas)

    tiempo = 0.0
    infect_now = int(grid.sum())

    # guardar el estado inicial
    times.append(tiempo)
    states.append(grid.copy())

    while True:

        total_q = ft.total()
        if total_q <= 1e-6:
            times.append(tiempo)
            states.append(np.zeros((N,N)))
            break
   

        dt = np.random.exponential(1 / total_q)
        tiempo += dt
        if tiempo >= t_max:
            times.append(tiempo)
            states.append(grid.copy())
            break

        # elegir nodo
        u = np.random.random() * total_q
        e = ft.find_by_prefix(u)
        i = e // N
        j = e % N

        old = grid[i, j]
        grid[i, j] ^= 1

        # actualizar infectados
        infect_now += (1 if old == 0 else -1)

        # criterio de saturación
        if infect_now > 0.786 * N**2 / 2:
            times.append(tiempo)
            states.append(grid.copy())
            break

        # actualizar tasas locales
        vecinos = [(i, j),
                   (i-1, j), (i+1, j),
                   (i, j-1), (i, j+1)]

        for (a, b) in vecinos:
            if 0 <= a < N and 0 <= b < N:
                idx = a*N + b
                old_t = tasas[idx]
                new_t = tasa_nodo(grid, a, b, lamb, mu, N)
                tasas[idx] = new_t
                ft.update(idx, new_t - old_t)

        # almacenar estado post-evento
        times.append(tiempo)
        states.append(grid.copy())

    return times, states







# ------------------------------------------------------------------------------------------ VIDEO ------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

def t_equispaceado(tiempos, estados, intervalo, t_final):

    equispaceado = np.arange(0, t_final+intervalo, intervalo)

    t_asociado = []
    estados_asociado = []

    i = 0  
    for t in equispaceado:

        while i < len(tiempos) and tiempos[i] <= t:   # <-- solo simulación 0
            i += 1
        
        t_asociado.append(t)
        estados_asociado.append(estados[i-1])         # <-- solo simulación 0
        

    return t_asociado, estados_asociado



def animar(tiempos_salto, estados_salto,
           intervalo=0.02, velocidad_ms=50):
    t_final = tiempos_salto[-1]
    if not np.isfinite(t_final) or t_final <= 0 or t_final > 10*t_max:
        t_final = t_max

    tiempos, estados = t_equispaceado(
        tiempos_salto,
        estados_salto,
        intervalo,
        t_final             # <-- solo simulación 0
    )

    N = estados[0].shape[0]

    fig, ax = plt.subplots(figsize=(6,6))

    # Mostrar nodos
    im = ax.imshow(estados[0], cmap="Oranges", vmin=0, vmax=1.5)

    title = ax.set_title(f"t = {tiempos[0]:.3f}")

    # -----------------------------------------------
    # TEXTO: número de infectados
    # Se muestra abajo a la izquierda del gráfico
    # -----------------------------------------------
    infect_now = int(np.sum(estados[0] == 1))
    title = ax.set_title(f"t = {tiempos[0]:.3f}   |   Infectados: {infect_now}")

    # -----------------------------------------------------------
    # ACTUALIZACIÓN FRAME A FRAME
    # -----------------------------------------------------------
    def update(frame):

        im.set_data(estados[frame])
   

        # actualizar número de infectados
        infect_now = int(np.sum(estados[frame] == 1))
        title.set_text(f"t = {tiempos[frame]:.3f}   |   Infectados: {infect_now}")

        return im, title


    ani = FuncAnimation(
        fig, update,
        frames=len(tiempos),
        interval=velocidad_ms,
        blit=False,
        repeat=False
    )

    plt.show()


N = 250


initial_process = np.zeros((N, N), dtype=int)
medio = N // 2
initial_process[medio:medio+1, medio:medio+1] = 1 

t_max =  40

tiempos, estados_registro= procesoContactoTrayectoria(
    N= N, t_max=t_max, lamb=0.8, mu=1 , initial_grid=initial_process
)

#print(tiempos[-1], estados_registro[-1], grafos_registro[-1])
#print(initial_process)
animar(tiempos, estados_registro,
       intervalo=0.02, velocidad_ms=30)
