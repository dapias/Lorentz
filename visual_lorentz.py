# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 18:14:56 2014

@author: maquinadt
"""

from lorentz import crear_simulacion
from visual import *

#Tamaño caja
L = 1000
#Radio partícula
r = 30
#Radio Disco
R = 500
#Velocidades mínima y máxima
v_min = -25
v_max = 25
sim = crear_simulacion(r,R,L,v_min,v_max)
sim.run(100)



lista = sim.posiciones
lista2 = sim.velocidades
lista3 = sim.t_eventos
#print lista, lista2, lista3

scene = display(title="Disco", width=L, height=L, center=(0,0,0))

curve(pos=[(-L,L,0),(L,L,0),(L,-L,0),(-L,-L,0), \
(-L,L,0)], color=color.blue, radius = r/2)

disco = sphere(color = color.green, radius= R) 

bola = sphere(color = color.red, radius= r)
bola.trail = curve(color=bola.color)


t = 0
deltat = 0.005



for i in xrange(len(lista)):
    bola.pos = vector(lista[i])
    bola.velocity = vector(lista2[i])


    while t < lista3[i]:
        rate(10000)
        bola.pos = bola.pos + bola.velocity*deltat
        t += deltat
        bola.trail.append(pos=bola.pos)

