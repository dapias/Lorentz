# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:38:56 2014

@author: diego
"""

import numpy as np
import random
from matplotlib import pyplot as plt

class particula(object):
    """clase Disco que recibe una lista de posiciones y velocidades"""

    def __init__(self,x,v,radio = 1.):
        self.x = np.array(x)
        self.v = np.array(v)
        self.radio = radio

    def __repr__(self):
        return "Partícula(%s,%s)"%(self.x,self.v)

    def movimiento(self,delta_t):
        self.x += delta_t * self.v

class disco(object):
    def__init__(self, radio_interno, posicion_disco = np.array([0.,0.])):
        self.radio = radio_interno
        self.posicion_disco = posicion_disco


class billar(object):
    def __init__(self,tamano,radio_interno, disco):
        self.tamano = tamano
        self.disco = disco

class ReglasColision(object):
    def __init__(self):
        pass

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

    def colision_discos(self,particula,disco):
       x_ij = particula.x - disco.x
       v_ij = particula.v
       vector_unitario = x_ij/np.linalg.norm(x_ij)
       h = np.dot(v_ij, vector_unitario)

       # Esto es lo importante de esta función:
       # actualiza velocidades!

       particula.v -= h*vector_unitario


   def tiempo_colision_discos(self, particula, disco):
       x_ij = particula.x - disco.x
       v_ij = particula.v
       x_ij_punto_v_ij = np.dot(x_ij,v_ij)

       # Si no hay condiciones para la colisión
       if x_ij_punto_v_ij > 0:
           return float('inf')

       d_cuadrada = np.sum(x_ij**2.)
       q = d_cuadrada-(particula.radio + disco.radio)**2.0
       velocidad_cuadrada = np.sum(v_ij**2.)
       w = x_ij_punto_v_ij **2. -  q*velocidad_cuadrada

       #De nuevo, si no hay condiciones para la colisión
       if w<0:
           return float('inf')

        #Ahora sí, viene la fórmula para el tiempo

       delta_t= q/(-x_ij_punto_v_ij  + np.sqrt(w))

       x_i = disco_i.x + delta_t * disco.v

       if not self.caja.contiene(x_i,disco_i.radio):
           return float('inf')

       x_j = disco_j.x + delta_t * disco_j.v

       if not self.caja.contiene(x_j, disco_j.radio) :
           return float('inf')


       return delta_t

    def _dt(self, particula,indice):

        dt = float('inf')

        if particula.v[indice] > 0:
            dt = (self.caja.tamano - (particula.x[indice] + particula.radio)) /\
                particula.v[indice]
        elif particula.v[indice] < 0:
            dt = (self.caja.tamano + (particula.x[indice] - particula.radio)) /\
                -particula.v[indice]
        return dt


   def tiempo_colision_pared(self, particula):

        resultado = min(self._dt(particula, 0), self._dt(particula, 1))
        return resultado

    class Simulacion(object):

    def __init__(self, particulas, reglas_colision=None, visualizacion = False):

        self.particulas = particulas
        self.long = len(self.particulas)
        if reglas_colision is None:
            reglas_colision = ReglasColision()
        self.reglas_colision = reglas_colision
        self.eventos = dict()
        self.tiempo = 0
        self.actualizar_particulas()
        self.t_eventos = []

        self.visualizacion = visualizacion
        if self.visualizacion:
            self.registro_posiciones = dict()
            self.registro_velocidades = dict()




    def actualizar_particulas(self):
        for particula in self.particulas:
            self.actualizar(particula)

    def actualizar(self, particula):

        for tiempo in particula.tiempos_eventos:
            if tiempo in self.eventos:
                del self.eventos[tiempo]

        particula.tiempos_eventos = []

        for otra_particula in self.particulas:
            if otra_particula != particula:
                dt = self.reglas_colision.tiempo_colision_discos(particula,
                 otra_particula)
                if dt < float('inf'):
                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula,otra_particula)
                    particula.tiempos_eventos.append(tiempo_col)
                    otra_particula.tiempos_eventos.append(tiempo_col)

        dt = self.reglas_colision.tiempo_colision_pared(particula)


        if dt < float('inf'):
            tiempo_col = self.tiempo + dt
            self.eventos[tiempo_col] = (particula, None)
            particula.tiempos_eventos.append(tiempo_col)

    def mover_particulas(self, delta_t):
        for particula in self.particulas:
            particula.movimiento(delta_t)

    def run(self, steps=10):

        if self.visualizacion:
            self.registro_posiciones = {"Disco" + str(i + 1) : np.ones((steps, 2)) for i in range(int(self.long))}
            self.registro_velocidades = {"Disco" + str(i + 1) : np.ones((steps, 2)) for i in range(int(self.long))}

#        for j in xrange(self.long):
#            self.registro_posiciones["Disco"+str(j+1)] = self.particulas[j].x
#            self.registro_velocidades["Disco"+str(j+1)] = self.particulas[j].v




        for i in xrange(steps):
            t_siguiente_evento = min(self.eventos.keys())
            siguiente_evento = self.eventos[t_siguiente_evento]



            if self.visualizacion:
                for j in xrange(self.long):
                    self.registro_posiciones["Disco"+str(j+1)][i] = self.particulas[j].x
                    self.registro_velocidades["Disco"+str(j+1)][i] = self.particulas[j].v


            print i,  self.tiempo, self.particulas



            #Tiempo de la última colisión:

            self.t_eventos.append(self.tiempo)


            if siguiente_evento[1] is None:
                delta_t = self.reglas_colision.tiempo_colision_pared(siguiente_evento[0])
                self.mover_particulas(delta_t)
                self.tiempo = t_siguiente_evento
                self.reglas_colision.colision_pared(siguiente_evento[0])
                self.actualizar(siguiente_evento[0])

            else:
                delta_t = self.reglas_colision.tiempo_colision_discos(siguiente_evento[0], siguiente_evento[1])
                self.mover_particulas(delta_t)
                self.tiempo = t_siguiente_evento
                self.reglas_colision.colision_discos(siguiente_evento[0], siguiente_evento[1])
                self.actualizar(siguiente_evento[0])
                self.actualizar(siguiente_evento[1])





    def energia_cin(self):
        return np.sum(np.sum(particula.v**2) for particula in self.particulas)


def es_traslape(nueva_particula, particulas):
    for particula in particulas:
        if nueva_particula.traslape_con(particula):
            return True
    return False

def crear_particulas_aleatorias(radio, tamano_caja,  v_min, v_max, num_particulas):
    #    np.random.seed(seed)
    particulas = []
    coords_max = tamano_caja - radio
    coords_min = -tamano_caja + radio
    for i in xrange(num_particulas):
        traslape = True
        while(traslape):
            x = np.random.uniform(coords_min, coords_max, 2)
            v = np.random.uniform(v_min, v_max, 2)
            nueva_particula = Disco(x, v,radio)
            traslape = es_traslape(nueva_particula, particulas)
        particulas.append(nueva_particula)
    return particulas

def crear_simulacion(radio, tamano_caja, v_min, v_max, num_particulas, visualizacion = False):
    caja = Caja(tamano_caja)
    particulas = crear_particulas_aleatorias(radio, tamano_caja, v_min, v_max,num_particulas)
    #particulas = [Disco([1,1],[3,1]),Disco([4,4],[4,0])]
    reglas = ReglasColision(caja)
    return Simulacion(particulas, reglas, visualizacion)

if __name__ == '__main__':

