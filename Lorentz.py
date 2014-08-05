# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:38:56 2014

@author: diego
"""

import numpy as np
import random
from matplotlib import pyplot as plt



def sign(number):return cmp(number,0)



def brent(f,a,b,tol=1.0e-10):
    x1 = a; x2 = b;
    f1 = f(x1)
    if f1 == 0.0: return x1
    f2 = f(x2)
    if f2 == 0.0: return x2
    if f1*f2 > 0.0: raise ValueError("Root is not bracketed")
    x3 = 0.5*(a + b)
    for i in range(30):
        f3 = f(x3)
        if abs(f3) < tol: return x3
    # Tighten the brackets on the root
        if f1*f3 < 0.0: b = x3
        else: a = x3
        if (b - a) < tol*max(abs(b),1.0): return 0.5*(a + b)
    # Try quadratic interpolation
        denom = (f2 - f1)*(f3 - f1)*(f2 - f3)
        numer = x3*(f1 - f2)*(f2 - f3 + f1) \
            + f2*x1*(f2 - f3) + f1*x2*(f3 - f1)
    # If division by zero, push x out of bounds
        try: dx = f3*numer/denom
        except ZeroDivisionError: dx = b - a
        x = x3 + dx
    # If iterpolation goes out of bounds, use bisection
        if (b - x)*(x - a) < 0.0:
            dx = 0.5*(b - a)
            x = a + dx
    # Let x3 <-- x & choose new x1 and x2 so that x1 < x3 < x2
        if x < x3:
            x2 = x3; f2 = f3
        else:
            x1 = x3; f1 = f3
        x3 = x
    print "Too many iterations in brent"

#def bisect(f,x1,x2,switch=0,epsilon=1.0e-9):
#   f1 = f(x1)
#   if f1 == 0.0: return x1
#   f2 = f(x2)
#   if f2 == 0.0: return x2
#   if f1*f2 > 0.0: return float('inf')
#   n = int(np.ceil(np.log(abs(x2 - x1)/epsilon)/np.log(2.0)))
#   for i in range(n):
#        x3 = 0.5*(x1 + x2); f3 = f(x3)
#        if (switch == 1) and (abs(f3) >abs(f1)) \
#                         and (abs(f3) > abs(f2)):
#            return None
#        if f3 == 0.0: return x3
#        if f2*f3 < 0.0:
#            x1 = x3; f1 = f3
#        else:
#            x2 = x3; f2 = f3
#   return (x1 + x2)/2.0

class Oscilador(object):
    """clase Oscilador que recibe amplitud, fase, frecuencia y posición inicial"""

    def __init__(self, equilibrio, amplitud, fase, omega):
        self.a = amplitud
        self.fase = fase
        self.omega = omega
        self.equilibrio = equilibrio
        self.x = equilibrio
        self.tiempos_colisiones = [0.]
#        self.posiciones = [self.a*np.sin(self.fase) + self.equilibrio]
        self.tiempos_eventos = [0.]


    def __repr__(self):
        return "Oscilador(%s,%s,%s,%s)"%(self.x,self.a,self.fase,self.omega)


    def movimiento(self,delta_t):
        self.x = self.a* np.sin(self.omega*delta_t + self.fase) + self.equilibrio



class Particula_libre(object):

    """ Recibe posición y velocidad inicial"""
    def __init__(self,x,v,etiqueta = 0):
       self.x = x
       self.v = v
       self.tiempos_eventos = []
       self.tiempos_colisiones = [0.]
       self.etiqueta = etiqueta
       self.velocidades_colisiones = [v]
    def __repr__(self):
        return "Partícula(%s,%s)"%(self.x,self.v)

    def movimiento(self,delta_t):
        self.x += delta_t * self.v

class Reservorio(object):

    def __init__(self, temperatura = 2.5, deltaT = 1., sentido = 0):
        self.sentido = sentido
        self.temperatura = temperatura
        self.deltaT = deltaT

    def velocidad(self, j = 0):
        i = np.random.random()
        if self.sentido == 0:
            vel = (np.sqrt(-np.log(1.-i)*2.*self.temperatura))
        if self.sentido == 1 or j==1:
            vel = -(np.sqrt(-np.log(1.-i)*2.*(self.temperatura-self.deltaT)))
        return vel

class Caja(object):
    """Es la longitud de la mitad de la caja, estoy suponiendo
    que la extensión de la caja es de [-tamano,tamano]"""
    def __init__(self,tamano = 100):
        self.tamano = tamano

class ReglasColision(object):

   def __init__(self, caja, reservorio):
        self.caja = caja
        self.reservorio = reservorio




   def tiempo_colision_particula_oscilador(self,particula, oscilador, tiempo_inicial = 1.e-7, tol = 1.e-10):

        tiempo = float('inf')

        x_p0 = particula.x
        v_p = particula.v
        a_o = abs(oscilador.a)
        f_o = oscilador.fase
        w = oscilador.omega
        eq_o = oscilador.equilibrio

        periodo = (2*np.pi)/(w)
        k = 0

        f_o = f_o % (2*np.pi)


        if x_p0 > (eq_o + a_o):
            if v_p > 0:
                t = float('inf')
                return t
            else:
                t = abs((eq_o + a_o- x_p0)/v_p)
                t_minimo = ((np.arccos(v_p/(a_o*w)) - f_o)/w)
                if np.isnan(t_minimo):
                    t_minimo = abs((eq_o - a_o- x_p0)/v_p)

                t_minimo2 = (2*np.pi - w*t_minimo - 2*f_o)/w

                k = 1

        elif x_p0 < (eq_o - a_o) :
            if v_p < 0:
                t = float('inf')
                return t
            else:
                t = abs((eq_o - a_o - x_p0)/v_p)
                t_minimo = ((np.arccos(v_p/(a_o*w)) - f_o)/w)
                if np.isnan(t_minimo):
                    t_minimo =  abs((eq_o + a_o- x_p0)/v_p)

                t_minimo2 = (2*np.pi - w*t_minimo - 2*f_o)/w

                k = 1

        else:
            t = tiempo_inicial
            t_minimo = (np.arccos(v_p/(a_o*w)) - f_o)/w
            if np.isnan(t_minimo):
                if v_p > 0.:
                    t_minimo = abs((eq_o + a_o - x_p0)/v_p)
                elif v_p < 0.:
                    t_minimo = abs((eq_o - a_o - x_p0)/v_p)

            t_minimo2 = (2*np.pi - w*t_minimo - 2*f_o)/w
            k = 0



        while t_minimo < 0.:
            t_minimo +=  periodo
            t_minimo2 += periodo

        while 1:
            t_minimo -= periodo
            if t_minimo < 0:
                break

        t_minimo = t_minimo + periodo


        while 1:
            t_minimo2-= periodo

            if t_minimo2< 0:
                break

        t_minimo2 = t_minimo2 + periodo

        if k == 1:
            x_p = x_p0 + t*v_p
            x_o = a_o*np.sin(w*t + f_o )  + eq_o

            if abs(x_p - x_o) < tol:
                return t

            m = np.ceil((t-t_minimo)/periodo)
            n = np.ceil((t-t_minimo2)/periodo)


            t_desplazado1 = t_minimo + m*periodo
            t_desplazado2 = t_minimo2 + n*periodo

            t_menor = min(t_desplazado1, t_desplazado2)
            t_mayor = max(t_desplazado1, t_desplazado2)

        elif k == 0:
            t_menor = min(t_minimo,t_minimo2)
            t_mayor = max(t_minimo, t_minimo2)

        def r(t):
            return a_o*np.sin(w*t + f_o) + eq_o - v_p*t- x_p0

        for i in xrange(12):
            if i == 0:
                t2 = t
                t3 = t_menor
                #print t2, t3
            elif i == 1:
                t2 = t_menor
                t3 = t_mayor


            elif i % 2 == 0 and i != 0:
                t2 = t3
                t3 = t_menor + (i-1)*periodo


            else:
                t2 = t3
                t3 = t_mayor + (i-2)*periodo




            g = sign(r(t2))
            h = sign(r(t3))

            if g !=h:
                tiempo = brent(r,t2,t3)
                break

        if tiempo < 0.:
            tiempo = float('inf')


#        print tiempo
        return tiempo



   def colision_particula_oscilador(self, particula_i,oscilador_j,delta_t):
   # Actualiza velocidades, amplitudes y fases.
      v_vieja = particula_i.v
      a_vieja = oscilador_j.a
      f_vieja = oscilador_j.fase
      w = oscilador_j.omega

      h = v_vieja**2 + (a_vieja**2*oscilador_j.omega**2)/2.*(1. - np.cos(2.*(oscilador_j.omega*delta_t + f_vieja)))

#      particula_i.v = (a_vieja*np.cos(oscilador_j.omega*delta_t + f_vieja))

      #La amplitud siempre va a ser positiva
      oscilador_j.a = np.sqrt(h)/oscilador_j.omega

      if oscilador_j.x < oscilador_j.equilibrio:
          oscilador_j.fase = -abs(np.arccos(v_vieja/np.sqrt(h)))
      else:
          oscilador_j.fase = abs(np.arccos(v_vieja/np.sqrt(h)))

      if abs(oscilador_j.fase % np.pi) < 1e-10 and oscilador_j.x < oscilador_j.equilibrio :
          oscilador_j.fase = -abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))

      elif abs(oscilador_j.fase % np.pi) < 1e-10 and oscilador_j.x > oscilador_j.equilibrio :
          oscilador_j.fase = abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))

      elif abs(oscilador_j.fase) < 1e-10 and oscilador_j.x > oscilador_j.equilibrio :
          oscilador_j.fase = abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))


      elif abs(oscilador_j.fase) < 1e-10 and oscilador_j.x < oscilador_j.equilibrio :
          oscilador_j.fase = -abs(np.arcsin(a_vieja/oscilador_j.a*np.sin(oscilador_j.omega*delta_t + f_vieja)))


      particula_i.v = v_vieja + a_vieja*w*np.cos(w*delta_t + f_vieja)  - oscilador_j.a*w*np.cos(oscilador_j.fase)




   def colision_pared(self, particula):

       if particula.etiqueta == -1:
           particula.v = self.reservorio.velocidad(0)

       if particula.etiqueta == 1:
           particula.v = self.reservorio.velocidad(1)

   def tiempo_colision_pared(self, particula):
       dt = float('inf')
       if particula.etiqueta == -1:
           if particula.v < 0:
               dt = (self.caja.tamano + particula.x)/-particula.v
       elif particula.etiqueta == 1:
           if particula.v > 0:
               dt = (self.caja.tamano - particula.x)/particula.v

       return dt


class Simulacion(object):

    def __init__(self, particulas_y_osciladores, reglas_colision=None):

        self.particulas_y_osciladores = particulas_y_osciladores
        self.particulas = [particula for particula in self.particulas_y_osciladores if isinstance(particula,Particula_libre)]
        self.osciladores = [oscilador for oscilador in self.particulas_y_osciladores if isinstance(oscilador, Oscilador)]
        self.longpart = len(self.particulas)
        self.longosc = len(self.osciladores)
        self.longtodo = len(self.particulas_y_osciladores)

        if reglas_colision is None:
            reglas_colision = ReglasColision()
        self.reglas_colision = reglas_colision
        self.eventos = dict()
        self.tiempo = 0
        self.t_eventos = [0.]



#Los siguientes diccionarios permitirán hacer el "ploteo" después
        self.registro_posiciones = dict()
        self.registro_velocidades = dict()
        self.registro_amplitudes = dict()
        self.registro_fases = dict()

#Lo siguiente me permite calcular el flujo.

        self.flujos_promedio = [[] for _ in xrange(self.longpart)]
        self.flujos_promedio2 = [[] for _ in xrange(self.longpart)]
        self.velocidades_extremos = [[] for _ in xrange(2)]
        self.flujos_reservorio =  [[] for _ in xrange(2)]
        self.tiempos_extremos1 = []
        self.tiempos_extremos2 = []
        self.Temperaturas = [[] for _ in xrange(self.longtodo)]



    def actualizar_particulas(self):
        for particula in self.particulas_y_osciladores:
            self.actualizar(particula)

    def actualizar(self, particula):

        for tiempo in particula.tiempos_eventos:
            if tiempo in self.eventos:
                del self.eventos[tiempo]

        particula.tiempos_eventos = []

        indice = self.particulas_y_osciladores.index(particula)



        try:

        #Para las partículas de los extremos
            if particula.etiqueta == -1:

                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[1])
                if dt < float('inf'):

                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[1])
                        particula.tiempos_eventos.append(tiempo_col)

                dt = self.reglas_colision.tiempo_colision_pared(particula)
                if dt < float('inf'):
                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, None)
                    particula.tiempos_eventos.append(tiempo_col)


            elif particula.etiqueta == 1:

                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[-2])
                if dt < float('inf'):
                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[-2])
                        particula.tiempos_eventos.append(tiempo_col)

                dt = self.reglas_colision.tiempo_colision_pared(particula)
                if dt < float('inf'):
                    tiempo_col = self.tiempo + dt
                    self.eventos[tiempo_col] = (particula, None)
                    particula.tiempos_eventos.append(tiempo_col)

            else:

                #Para el resto de partículas


                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[indice - 1])
                if dt < float('inf'):
                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[indice - 1])
                        particula.tiempos_eventos.append(tiempo_col)



                dt = self.reglas_colision.tiempo_colision_particula_oscilador(particula, self.particulas_y_osciladores[indice + 1])
                if dt < float('inf'):
                        tiempo_col = self.tiempo + dt
                        self.eventos[tiempo_col] = (particula, self.particulas_y_osciladores[indice + 1])
                        particula.tiempos_eventos.append(tiempo_col)




        except(AttributeError):
#Esta excepción permite calcular los tiempos de colisión partícula-oscilador,"partiendo del oscilador", habrá que pensar si vale la pena

#Si el oscilador choca con su partícula del lado izquierdo.
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(self.particulas_y_osciladores[indice - 1], particula)
            if dt < float('inf'):
                tiempo_col = self.tiempo + dt
                self.eventos[tiempo_col] = (self.particulas_y_osciladores[indice - 1], particula)
                particula.tiempos_eventos.append(tiempo_col)

#Si el oscilador choca con su partícula del lado derecho.
            dt = self.reglas_colision.tiempo_colision_particula_oscilador(self.particulas_y_osciladores[indice + 1], particula)
            if dt < float('inf'):
                tiempo_col = self.tiempo + dt
                self.eventos[tiempo_col] = (self.particulas_y_osciladores[indice + 1], particula)
                particula.tiempos_eventos.append(tiempo_col)


    def mover_particulas_y_osciladores(self, delta_t):
        for particula in self.particulas_y_osciladores:
                particula.movimiento(np.float(delta_t))


    def actualizar_fases(self, delta_t, oscilador_prohibido = Oscilador(0,0,0,0)):
        for oscilador in self.particulas_y_osciladores:
            if oscilador != oscilador_prohibido:
                try:
                    oscilador.fase += delta_t*oscilador.omega
                except(AttributeError):
                    pass


    def run(self, imprimir = 0, flujos = 0, terminar = 0, cota = 2000,numero_puntos = 100):


        self.registro_posiciones = {"Particula" + str(i + 1) : [] for i in range(int(self.longpart))}
        self.registro_velocidades = {"Particula" + str(i + 1) : [] for i in range(int(self.longpart))}
        self.registro_amplitudes = {"Oscilador" + str(i + 1) : [] for i in range(int(self.longosc))}
        self.registro_fases = {"Oscilador" + str(i + 1) : [] for i in range(int(self.longosc))}
        self.cota2 = 0.

        deltaEs_extremos = [[] for _ in xrange(2)]



        num_eventos_pared = 0
        i = 0

        try:
            while True:
                self.actualizar_particulas()
                t_siguiente_evento = min(self.eventos.keys())
                siguiente_evento = self.eventos[t_siguiente_evento]

        #Estos datos los usaré después para plotear y calcular los flujos.
                for j in xrange(self.longpart):
                    self.registro_posiciones["Particula" + str(j+1)].append(self.particulas[j].x)
                    self.registro_velocidades["Particula" + str(j+1)].append(self.particulas[j].v)

                for k in xrange(self.longosc):
                    self.registro_amplitudes["Oscilador" + str(k+1)].append(self.osciladores[k].a)
                    self.registro_fases["Oscilador" + str(k+1)].append(self.osciladores[k].fase)





        #Si se estrella contra la pared
                if siguiente_evento[1] is None:

                    if siguiente_evento[0].etiqueta == -1:
                        a = siguiente_evento[0].v
                    if siguiente_evento[0].etiqueta == 1:
                        b = siguiente_evento[0].v
                    delta_t = self.reglas_colision.tiempo_colision_pared(siguiente_evento[0]) #Lo mismo que ya había hecho

                    for particula in self.particulas_y_osciladores:
                       self.Temperaturas[self.particulas_y_osciladores.index(particula)].append(self.temperatura(particula,delta_t))

                    self.tiempo = t_siguiente_evento #El tiempo que había más éste delta t"
                    self.mover_particulas_y_osciladores(delta_t) #Cambio posiciones del oscilador y las particulas
                    self.actualizar_fases(np.float(delta_t))      #Actualizo las fases de todos los ociladores.
                    self.reglas_colision.colision_pared(siguiente_evento[0])

                    if siguiente_evento[0].etiqueta == -1:
                        c = siguiente_evento[0].v
                        deltaEs_extremos[0].append((c**2 - a**2)*0.5)
                        self.tiempos_extremos1.append(t_siguiente_evento)
                    elif siguiente_evento[0].etiqueta == 1:
                        d = siguiente_evento[0].v
                        deltaEs_extremos[1].append((d**2 - b**2)*0.5)
                        self.tiempos_extremos2.append(t_siguiente_evento)

                    num_eventos_pared += 1

                    if num_eventos_pared == cota:
                        self.cota2 = i



        # Si chocan la particula y el oscilador

                else:
                    delta_t = self.reglas_colision.tiempo_colision_particula_oscilador(siguiente_evento[0], siguiente_evento[1])

                    for particula in self.particulas_y_osciladores:
                        self.Temperaturas[self.particulas_y_osciladores.index(particula)].append(self.temperatura(particula,delta_t))

                    self.tiempo = t_siguiente_evento
                    self.mover_particulas_y_osciladores(delta_t)
                    self.actualizar_fases(delta_t, siguiente_evento[1]) #Acualizo las fases de todos los osciladores, exceptuando el que chocó
                    self.reglas_colision.colision_particula_oscilador(siguiente_evento[0], siguiente_evento[1],np.float(delta_t))



                self.t_eventos.append(self.tiempo)


                if imprimir == 1:
                    print i + 1,  self.tiempo #self.particulas_y_osciladores

        #Esta condición es por si algo sale mal.
                for particula in self.particulas:
                    if abs(particula.x) > self.reglas_colision.caja.tamano + 1e-5:
                        raise ValueError ('Alguna particula salió de la caja' + repr(particula))



                if flujos == 1:

                    if num_eventos_pared > cota*5:
                        if siguiente_evento[1] is None:
                            if siguiente_evento[0].etiqueta == -1:
                                self.flujos_reservorio[0].append(np.sum(deltaEs_extremos[0][cota:])/(self.tiempos_extremos1[-1] - self.tiempos_extremos1[cota]))
                            else:
                                self.flujos_reservorio[1].append(np.sum(deltaEs_extremos[1][cota:])/(self.tiempos_extremos2[-1] - self.tiempos_extremos2[cota]))


                        if num_eventos_pared > 6*cota:
#                        if num_eventos_pared > 12*cota:
                            print self.flujos_reservorio[0][-1], self.flujos_reservorio[1][-1]






                i += 1

                if terminar == 1:
                    if i == numero_puntos:
                        break

        except(KeyboardInterrupt):
            print 'interrupted!'
            print 'promedio y desviación estándar lado izquierdo', np.average(self.flujos_reservorio[0]), np.std(self.flujos_reservorio[0])
            print 'promedio y desviación estándar lado derecho', np.average(self.flujos_reservorio[1]), np.std(self.flujos_reservorio[1])




    def temperatura(self,particula,delta_t):
        try:
            return particula.v**2.*delta_t
        except(AttributeError):
            return particula.a**2.*delta_t*particula.omega**2./2.




def crear_particulas_aleatorias(tamano_caja, num_particulas_y_osciladores, omega, reservorio):
    #    np.random.seed(seed)
    particulas_y_osciladores = []
    energia = []

    for i in xrange(num_particulas_y_osciladores):
        if i % 2 == 0:
            if i == 0:
                v1 = reservorio.velocidad()
                energia.append(v1**2./2.)
            elif i == num_particulas_y_osciladores-1:
                v2 = -reservorio.velocidad(1)
                energia.append(v2**2./2.)
            else:
                i = np.random.random()
                v3= random.choice([-1.,1.])*random.choice([reservorio.velocidad(),reservorio.velocidad(1)])
                energia.append(v3**2./2.)
        else:
            A = np.random.randn()
            energia.append(A**2. * omega**2.)

    norma = np.sum(energia)/(num_particulas_y_osciladores)

    energia2 = energia/(norma)
    for i in xrange(num_particulas_y_osciladores):
        #Si son partículas
        if i % 2 == 0:
            x = -(tamano_caja-1) + i + np.random.uniform(-0.5,0.5)
            if i == 0:
                v1 = np.sqrt(energia2[i]*2)
                nueva_particula = Particula_libre(x, v1, -1)
                energia.append(v1**2/2)
            elif i == num_particulas_y_osciladores-1:
                v2 = -np.sqrt(energia2[i]*2)

                nueva_particula = Particula_libre(x, v2, 1)
            else:
                i = np.random.random()
                v3= random.choice([-1.,1.])*np.sqrt(energia2[i]*2)
                nueva_particula = Particula_libre(x, v3)

        else:
            x_eq = -(tamano_caja -1.)+ i
            A = np.sqrt(energia2[i]/omega**2.) % 0.5
            Fase = np.random.uniform(0.,np.pi/2.)
            nueva_particula = Oscilador(x_eq,A,Fase,omega)


        particulas_y_osciladores.append(nueva_particula)

    print particulas_y_osciladores
    return particulas_y_osciladores



def plot_datos(sim, total_particulas, omega, puntos = 1):
    tiempo = []
    num_particulas = (total_particulas - 1) * 0.5 + 1
    num_osciladores = (total_particulas - 1) * 0.5
    px = [[] for _ in xrange(int(num_particulas))]
    osx = [[] for _ in xrange(int(num_osciladores))]

    if puntos == 1:
        tiempo_exacto = []
        px_exacto = [[] for _ in xrange(int(num_particulas))]
        osx_exacto = [[] for _ in xrange(int(num_osciladores))]

        for i in xrange(len(sim.t_eventos)-1):
            for j in xrange(int(num_particulas)):
                px_exacto[j].append(sim.registro_posiciones["Particula" + str(j + 1)][i])
            for j in xrange(int(num_osciladores)):
                osx_exacto[j].append(sim.registro_amplitudes["Oscilador" +  str(j + 1) ][i]*np.sin(np.float(sim.registro_fases["Oscilador" +  str(j + 1)][i])) + sim.osciladores[j].equilibrio)

            tiempo_exacto.append(sim.t_eventos[i])

        for j in xrange(int(num_particulas)):
            plt.plot(tiempo_exacto, px_exacto[j],'o')
        for j in xrange(int(num_osciladores)):
            plt.plot(tiempo_exacto, osx_exacto[j], 'o')

    t = 0
    dt = 0.005



    for i in xrange(len(sim.t_eventos) - 1):

        while  t < float(sim.t_eventos[i + 1]):

            for j in xrange(int(num_particulas)):
                px[j].append(sim.registro_posiciones["Particula" + str(j + 1)][i] + (t - sim.t_eventos[i]) * sim.registro_velocidades["Particula" + str(j + 1)][i])
#
            for j in xrange(int(num_osciladores)):
                osx[j].append(sim.registro_amplitudes["Oscilador" +  str(j + 1) ][i]*np.sin(np.float(omega*(t - sim.t_eventos[i]) + sim.registro_fases["Oscilador" +  str(j + 1)][i])) + sim.osciladores[j].equilibrio)


            tiempo.append(t)
            t += dt

    for j in xrange(int(num_particulas)):
        plt.plot(tiempo, px[j])
    for j in xrange(int(num_osciladores)):
        plt.plot(tiempo, osx[j])

    #plt.axis([0,25,-2,2])
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.show()

def plot_flujos(sim):
    numero_extremos1 = np.arange(len(sim.flujos_reservorio[0]))
    numero_extremos2 = np.arange(len(sim.flujos_reservorio[1]))

    plt.plot(numero_extremos1, sim.flujos_reservorio[0], '-o')
    plt.plot(numero_extremos2, sim.flujos_reservorio[1], '-o')

    plt.show()


def plot_temperaturas(sim):
     numero_total= np.arange(sim.longtodo)
     cota2 = sim.cota2
     print "cota2", cota2

     T_promedio = [np.sum(sim.Temperaturas[j][cota2:]) for j in xrange(sim.longtodo)]/(sim.t_eventos[-1] - sim.t_eventos[cota2])
     print 'temperaturas promedio', T_promedio
     plt.figure()
     plt.plot(numero_total, T_promedio , 'o', label='Original data', markersize=5)

     A = np.vstack([numero_total, np.ones(len(numero_total))]).T
     m, c = np.linalg.lstsq(A, T_promedio)[0]
     a = np.linalg.lstsq(A, T_promedio)[0]
     print "gradiente", m

     resids = T_promedio - np.dot(A,a)       # e = y - Xa;
     RSS = sum(resids**2)           # residual sum of squares
     TSS = sum((T_promedio - np.mean(T_promedio))**2) # total sum of squares
     R2 = 1 - RSS/TSS
     print "R^2", R2

     plt.plot(numero_total, m*numero_total + c, 'r', label='Fitted line')
     plt.ylabel(r'T')
     plt.xlabel(u'Número de partícula')
     plt.title(u'T vs. N para el estado estacionario')
     plt.legend()
     plt.show()



if __name__ == '__main__':
    lista = []
    particula1 = Particula_libre(-1.03353521727,-0.498883053656,-1)
    oscilador1 = Oscilador(-1.,0.0772580376392,-0.44900312108,10.0)
    particula2 = Particula_libre(0.547901461924,-0.512428895414)
    oscilador2 = Oscilador(1.,0.0983684058091,6.46542810578,10.0)
    particula3 = Particula_libre(1.24550828356,-1.01517149803,1)
    lista.append(particula1)
    lista.append(oscilador1)
    lista.append(particula2)
    lista.append(oscilador2)
    lista.append(particula3)
#    np.random.seed(211)
    frecuencia = 0.2
    num_total = 7
    reservorio = Reservorio()
    caja = Caja(np.float(num_total - 1.)/2. + 1.)
    lista = crear_particulas_aleatorias(caja.tamano,num_total,frecuencia,reservorio)
    reglas = ReglasColision(caja, reservorio)
    sim = Simulacion(lista, reglas)
    try:
        sim.run(1,0,1)
        plot_datos(sim, num_total, frecuencia, 0)
#        plot_temperaturas(sim)
#        plot_flujos(sim)
    except(ValueError):
        print "Hubo un error en alguna particula"
        plot_datos(sim, num_total, frecuencia, 0)
        plot_temperaturas(sim)
        plot_flujos(sim)

#print sim.eventos
