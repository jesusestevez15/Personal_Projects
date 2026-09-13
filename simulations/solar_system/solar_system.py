import os
import yaml
import sys
import functions
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def on_close(event):
    sys.exit(0)

# Constantes
G = 6.67*10**-11
msol = 1.9891*10**30
ua = 1.496*10**11

# Variables por configuracion
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

h = config["simulation"]["h"]
Tmax = config["simulation"]["Tmax"]
N = config["simulation"]["N"]
omega = config["simulation"]["omega"]
plot_posiciones = config["plot"]["plot_posiciones"]
plot_velocidades = config["plot"]["plot_velocidades"]

# Masa de planetas
m = np.random.rand(N) * 10
m[np.random.choice(N, size=int(0.3 * N), replace=False)] *= 0.01 # planetas menos masivos

# Posiciones iniciales en 3D
x = np.random.rand(N, 3) - 0.5

# Velocidad inicial rotacional usando producto vectorial (disco tipo protoplanetario en plano XY)
eje_rotacion = np.array([0, 0, 1])
v = omega * np.cross(eje_rotacion, x)

# Estrella estatica
m[0] = 10000000
x[0] = 0
v[0] = 0

# Distancia minima para impacto entre cuerpos
th = 0.02

# Traza de los planetas en los ultimos 10 segundos
trail_seconds = 10
trail_max_len = max(1, int(trail_seconds / h))
history_x = deque(maxlen=trail_max_len)
history_v = deque(maxlen=trail_max_len)

plt.ion()

if plot_posiciones and plot_velocidades:
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
elif plot_posiciones:
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111, projection='3d')
elif plot_velocidades:
    fig = plt.figure(figsize=(6, 6))
    ax2 = fig.add_subplot(111, projection='3d')

fig.canvas.mpl_connect('close_event', on_close)

# Comienzo de algoritmo. Calculo de aceleraciones de cada cuerpo en t=0
a = np.zeros((N,3))
a = functions.Aceleracion(m, x, N, G)

t = 0
while t < Tmax:
    # Calculo de posiciones, velocidades, aceleraciones e impactos en cada instante temporal
    x, w = functions.Posicion(x, v, np.zeros((N,3)), h, a, N)
    a = functions.Aceleracion(m, x, N, G)
    v = functions.Velocidad(v, w, a, h, N)
    m, x, v, w, a, N = functions.Impacto(m, x, v, w, a, N, th)

    t = t + h

    # Ploteo de los resultados
    if plot_posiciones:
        history_x.append(x.copy())  # historial de posiciones
        functions.PlotPosiciones(ax1, history_x, m, x, t)

    if plot_velocidades:
        history_v.append(v.copy())  # historial de velocidades
        functions.PlotVelocidades(ax2, history_v, v, t)

    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()