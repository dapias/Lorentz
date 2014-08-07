from lorentz import crear_simulacion
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation

#Tamano caja
L = 1000
#Radio particula
r = 30
#Radio Disco
R = 500
#Velocidades minima y maxima
v_min = -25
v_max = 25
sim = crear_simulacion(r,R,L,v_min,v_max)
sim.run(100)


lista = sim.posiciones


#print lista, lista2, lista3


def simData():

    lista = sim.posiciones
    lista2 = sim.velocidades
    lista3 = sim.t_eventos

    t = 0
    deltat = 1.

    for i in xrange(len(lista)):
        x = lista[i]
        v = lista2[i]

        while t < lista3[i]:
            x +=  v*deltat
            t += deltat

            yield x,t
#patch = plt.Circle(lista[0], r, fc = 'r')

def simPoints(simData):
    x, t = simData[0], simData[1]
    line.set_data(x)
    return line

fig = plt.figure()
ax = plt.axes(xlim=(-L, L), ylim=(-L, L))
line, = ax.plot([], [], 'yo', ms=10)


patch1 = plt.Circle((0, 0), R, fc='y')
ax.add_patch(patch1)
#patch2 = plt.Circle(lista[0], r, fc='r')

#def init():
 #   patch1.center = (0.,0.)
  #  patch2.center = (lista[0])
   # ax.add_patch(patch1)
    #ax.add_patch(patch2)
    #return patch1, patch2

#def animate(i):
  #  x, y = patch2.center
#    x = lista[0][0]* np.sin(np.radians(i))
 #   y =lista[0][1] * np.cos(np.radians(i))
  #  patch2.center = (x, y)
   # return patch2,

anim = animation.FuncAnimation(fig, simPoints, simData, interval = 1, blit=False)

#anim = animation.FuncAnimation(fig, animate, init_func=init, frames=540, interval=10, blit=False)

plt.show()
