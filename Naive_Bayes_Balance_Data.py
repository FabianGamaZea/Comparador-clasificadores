import math
import numpy as np
import pandas as pd

from Balance_Data import Hold_Out, Rendimiento_Hold_Out,K_Fold_Cross,Leave_One_Out,Rendimiento_Leave_One_Out

def Media(rasgo):
    Media = np.sum(rasgo)/len(rasgo)
    return Media

def Desviacion_Estandar(rasgo):
    media = Media(rasgo)
    suma_acumulada=0;
    n = 1/(len(rasgo)-1)
    
    for i in range(len(rasgo)):
        suma_acumulada=((rasgo[i]-media)**2)+suma_acumulada
    
    desviacion_estandar= math.sqrt(n* suma_acumulada)
    return float(desviacion_estandar)

def Contar_Aciertos(resultados,C_P):
    cont=0;
    for i in range(len(resultados)):
        A= resultados[i]
        A="".join(A)
        B= C_P[i][4:5]
        B="".join(B)
        if(A==B):
            cont= cont+1
    return cont

def Distribucion_Normal(patron,media,desviacion_estandar):
    if(desviacion_estandar==0):
        return 10
    e= float(math.e)
    part1=1/((desviacion_estandar)*((2*math.pi)**0.5))
    elev1= (patron-media)**2
    elev2 = 2*(desviacion_estandar**2)
    elev = -(elev1/elev2)
    distribucion_normal = part1*(e**elev)
    return float(distribucion_normal)

def Comparacion(rasgo_1,rasgo_2,rasgo_3,patron):
    
    Distribucion_Normal_1=round(Distribucion_Normal(patron,Media(rasgo_1),Desviacion_Estandar(rasgo_1)),5)
    Distribucion_Normal_2=round(Distribucion_Normal(patron,Media(rasgo_2),Desviacion_Estandar(rasgo_2)),5)
    Distribucion_Normal_3=round(Distribucion_Normal(patron,Media(rasgo_3),Desviacion_Estandar(rasgo_3)),5)
    
    distribucion_normal=[Distribucion_Normal_1,Distribucion_Normal_2,Distribucion_Normal_3]
    mayor=0.0
    cant= 0
    
    for i in range(len(distribucion_normal)):
        if (i==0):
            mayor=distribucion_normal[i]
            cant=i+1
            
        if (distribucion_normal[i]>mayor):
            mayor=distribucion_normal[i]
            cant = i+1
    
    #print("gana",cant)
    return cant

def Mayority_Comparaciones(Comparacion_rasgo_1,Comparacion_rasgo_2,Comparacion_rasgo_3,Comparacion_rasgo_4):
    tipo_1 =0
    tipo_2 =0
    tipo_3 =0
    Comparaciones= [Comparacion_rasgo_1,Comparacion_rasgo_2,Comparacion_rasgo_3,Comparacion_rasgo_4]
    
    for i in range(len(Comparaciones)):
        if (Comparaciones[i]==1):
            tipo_1=tipo_1+1
        if (Comparaciones[i]==2):
            tipo_2=tipo_2+1
        if (Comparaciones[i]==3):
            tipo_3=tipo_3+1
    
    may=0
    posicion=0
    
    tipos=[tipo_1,tipo_2,tipo_3]
    
    
    for i in range(len(tipos)):
        if (tipos[i]>may):
            may=tipos[i]
            posicion = i
    
    nom_tipo1='B'
    nom_tipo2='L'
    nom_tipo3='R'
    
    if(posicion==0):
        return nom_tipo1
    if(posicion==1):
        return nom_tipo2
    if(posicion==2):
        return nom_tipo3
    
    return'N'
    
def Hold_Out_Clasificador_Naive_Bayes(C_P,C_F):
    
    tamaño = len(C_F)//3 
    
    Setosa =np.asarray(C_F [:tamaño])
    Setosa_rasgo_1 = Setosa[:,:1]
    Setosa_rasgo_2 = Setosa[:,1:2]
    Setosa_rasgo_3 = Setosa[:,2:3]
    Setosa_rasgo_4 = Setosa[:,3:4]
    
    Virginica =np.asarray(C_F [tamaño:tamaño*2])
    Virginica_rasgo_1 = Virginica[:,:1]
    Virginica_rasgo_2 = Virginica[:,1:2]
    Virginica_rasgo_3 = Virginica[:,2:3]
    Virginica_rasgo_4 = Virginica[:,3:4]
    
    Versicolor =np.asarray(C_F [tamaño*2:])
    Versicolor_rasgo_1 = Versicolor[:,:1]
    Versicolor_rasgo_2 = Versicolor[:,1:2]
    Versicolor_rasgo_3 = Versicolor[:,2:3]
    Versicolor_rasgo_4 = Versicolor[:,3:4]
    
    resultados=[]
    np.asarray(resultados)
    for i in range(len(C_P)):
        res= Mayority_Comparaciones(Comparacion(Setosa_rasgo_1,Virginica_rasgo_1,Versicolor_rasgo_1,C_P[i][:1]),
                                     Comparacion(Setosa_rasgo_2,Virginica_rasgo_2,Versicolor_rasgo_2,C_P[i][1:2]),
                                     Comparacion(Setosa_rasgo_3,Virginica_rasgo_3,Versicolor_rasgo_3,C_P[i][2:3]),
                                     Comparacion(Setosa_rasgo_4,Virginica_rasgo_4,Versicolor_rasgo_4,C_P[i][3:4]))
        
        if i == 0:
            resultados = [res]
        else:
            resultados = np.concatenate((resultados,[res]))
        
    return resultados

def Recorrer_K_Fold_Cross_Clasificador_Naive_Bayes(B_D_lista):
    C_P=[]
    C_F=[]
    C_F_Partes=[]
    aux=0
    votaciones=[]
    for i in range(len(B_D_lista)):
        aux = aux+1
        #print(aux,"parte")
        if i ==0:
            C_P = B_D_lista[i]
            C_F_Partes = B_D_lista[i+1:]
            
        elif i == len(B_D_lista)-1:
            
            C_P = B_D_lista[i]
            C_F_Partes = B_D_lista[:i]
            
        else:
             C_P = B_D_lista[i]
             C_F_Partes = np.concatenate((B_D_lista[:i],B_D_lista[i+1:]))
             
        for i in range(len(C_F_Partes)):
            #print("parte",i)
            if i == 0:
                C_F = C_F_Partes[i]
            else:
                C_F = np.concatenate((C_F,C_F_Partes[i]))
        if i == 0:
            votaciones = Hold_Out_Clasificador_Naive_Bayes(C_P,C_F)
        else:
            votaciones = np.concatenate((votaciones,Hold_Out_Clasificador_Naive_Bayes(C_P,C_F)))
    return votaciones

def Recorrer_Leave_One_Out_Knn_Clasificador_Naive_Bayes(matriz):
    Votaciones= []
    for i in range(len(matriz)):
        Leave_One_Out_aux = Leave_One_Out(i, matriz)
        C_F = Leave_One_Out_aux[0]
        C_P = Leave_One_Out_aux[1]
        
        if i == 0:
            Votaciones=[Hold_Out_Clasificador_Naive_Bayes(C_P,C_F)]
        else:
            Votaciones = Votaciones + [Hold_Out_Clasificador_Naive_Bayes(C_P,C_F)]
            
    return Votaciones 

archivo = pd.read_csv(".//balance-scale.csv",header=0)

matriz = np.asarray(archivo)

C_F = Hold_Out(70, matriz)[0]
C_P = Hold_Out(70, matriz)[1]

print()
print("Clasificador Naive Bayes")
print("Rendimiento Hold Out  :",round(Rendimiento_Hold_Out(Contar_Aciertos(Hold_Out_Clasificador_Naive_Bayes(C_P,C_F),C_P),C_P),2),"%")
print()
print("Rendimiento K Fold Cross : ",round(Rendimiento_Hold_Out(Contar_Aciertos(Recorrer_K_Fold_Cross_Clasificador_Naive_Bayes(K_Fold_Cross(5,matriz)),matriz),matriz),2),"%")
print()
print("Rendimiento leave One Out :" ,round(Rendimiento_Leave_One_Out(Contar_Aciertos(Recorrer_Leave_One_Out_Knn_Clasificador_Naive_Bayes(matriz),matriz),matriz),2),"%")


