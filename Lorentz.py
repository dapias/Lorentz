# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 17:56:59 2014

@author: maquinadt
"""

import numpy as np
import random
from matplotlib import pyplot as plt

class Particula(object):
    """clase Disco que recibe una lista de posiciones y velocidades"""

    def __init__(self,x,v,radio = 1.):
        self.x = np.array(x)
        self.v = np.array(v)
        self.radio = radio

    def __repr__(self):
        return "Partícula(%s,%s)"%(self.x,self.v)

    def movimiento(self,delta_t):
        self.x += delta_t * self.v

class Disco(object):


    def __init__(self, radio_interno, posicion_disco = np.array([0.,0.])):
        self.radio_interno = radio_interno
        self.posicion_disco = posicion_disco


class Billar(object):
    def __init__(self,tamano, disco):
        self.tamano = tamano
        self.disco = disco

class ReglasColision(object):
   def __init__(self, billar):
        self.billar = billar

   def colision_pared(self, particula):

        if particula.x[1] < 0:
            if particula.x[1] > particula.x[0] or particula.x[1] > -particula.x[0]:
                particula.v[0] = -particula.v[0]
            else:
                particula.v[1] = -particula.v[1]

        else:
            if particula.x[1] < -particula.x[0] or particula.x[1] < particula.x[0]:
                particula.v[0] = -particula.v[0]
            else:
                particula.v[1] = -particula.v[1]

   def colision_disco(self,particula):
        #Consideraré que la partícula tiene masa unitaria

       x_ij = particula.x - self.billar.disco.posicion_disco
       vector_unitario = x_ij/np.linalg.norm(x_ij)
       v_ij = particula.v
       v_ij_punto_v_unitario = np.dot(v_ij,vector_unitario)


       # Esto es lo importante de esta función:
       # actualiza velocidades!

       particula.v -= 2*(v_ij_punto_v_unitario)*vector_unitario


   def tiempo_colision_disco(self, particula):
       x_ij = particula.x - self.billar.disco.posicion_disco
       vector_unitario = x_ij/np.linalg.norm(x_ij)
       v_ij = particula.v
       #x_ij_punto_v_unitario = np.dot(x_ij,vector_unitario)
       v_ij_punto_v_unitario = np.dot(v_ij,vector_unitario)
       x_ij_punto_v_ij = np.dot(x_ij,v_ij)

       # Si no hay condiciones para la colisión
       if x_ij_punto_v_ij > 0:
           return float('inf')

       d_ij = np.linalg.norm(x_ij)

       #Ahora sí, viene la fórmula para el tiempo

       delta_t = (d_ij - particula.radio - self.billar.disco.radio_interno ) / abs(v_ij_punto_v_unitario)

       return delta_t

   def _dt(self, particula,indice):

        dt = float('inf')

        if particula.v[indice] > 0:
            dt = (self.billar.tamano - (particula.x[indice] + particula.radio)) /\
                particula.v[indice]

        elif particula.v[indice] < 0:
            dt = (self.billar.tamano + (particula.x[indice] - particula.radio)) /\
                -particula.v[indice]

        return dt


   def tiempo_colision_pared(self, particula):

        resultado = min(self._dt(particula, 0), self._dt(particula, 1))
        return resultado

class Simulacion(object):


    def __init__(self, particula, reglas_colision, billar):

        self.particula = particula
        self.reglas_colision = reglas_colision
        self.billar = billar
        self.eventos = dict()
        self.tiempo = 0
        self.t_eventos = []
        self.posiciones = []
        self.velocidades = []
        self.actualizar(particula)

    def actualizar(self, particula):
#        print self.eventos

        self.eventos = dict()

        dt = self.reglas_colision.tiempo_colision_disco(particula)

        if dt < float('inf'):
                tiempo_col = self.tiempo + dt
                self.eventos[tiempo_col] = (particula, 2.)

        dt = self.reglas_colision.tiempo_colision_pared(particula)

        if dt < float('inf'):
            tiempo_col = self.tiempo + dt
            self.eventos[tiempo_col] = (particula, None)


    def run(self, steps=10):

        for i in xrange(steps):

            t_siguiente_evento = min(self.eventos.keys())
            siguiente_evento = self.eventos[t_siguiente_evento]
            print i,  self.tiempo, self.particula

            #Tiempo de la última colisión:

            self.t_eventos.append(self.tiempo)



            if siguiente_evento[1] is None:
                delta_t = self.reglas_colision.tiempo_colision_pared(siguiente_evento[0])
                self.particula.movimiento(delta_t)
                self.reglas_colision.colision_pared(siguiente_evento[0])

            else:
                delta_t = self.reglas_colision.tiempo_colision_disco(siguiente_evento[0])
                self.particula.movimiento(delta_t)
                self.reglas_colision.colision_disco(siguiente_evento[0])

            self.tiempo = t_siguiente_evento
            self.actualizar(siguiente_evento[0])

            self.posiciones.append(siguiente_evento[0].x)
            self.velocidades.append(siguiente_evento[0].v)

def crear_particula_aleatoria(radio, billar,  v_min, v_max):
    #    np.random.seed(seed)

    coords_max = billar.tamano - radio
    coords_min = -billar.tamano + radio
    traslape = True

    while(traslape):
        x = np.random.uniform(coords_min, coords_max, 2)
        if   ((x[0] - billar.disco.posicion_disco[0])**2. + (x[1] -  billar.disco.posicion_disco[1])**2.) > billar.disco.radio_interno + radio:
            traslape = False

    v = np.random.uniform(v_min, v_max, 2)
    particula = Particula(x, v,radio)

    return particula


def crear_simulacion(radio, radio_interno, tamano_caja, v_min, v_max, posicion_disco = np.array([0.,0.])):
    disco = Disco(radio_interno,posicion_disco)
    billar = Billar(tamano_caja,disco)
    particula = crear_particula_aleatoria(radio, billar, v_min, v_max)
    reglas = ReglasColision(billar)

    return Simulacion(particula, reglas, billar)

if __name__ == '__main__':
    sim = crear_simulacion(1.,5.,10.,-1.,1.)
    sim.run(10)
