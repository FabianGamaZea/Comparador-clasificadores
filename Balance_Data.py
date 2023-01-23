import math
import numpy as np
import pandas as pd


def Fase_Aprendizaje(C_F):
    tamaño = len(C_F)//3 
        
    Conjunto_m1 =np.asarray(C_F [:tamaño])
    Conjunto_m2 =np.asarray(C_F [tamaño:tamaño*2])
    Conjunto_m3 =np.asarray(C_F [tamaño*2:])
    
    
    Suma_m1 = np.add.reduce(Conjunto_m1[:,:4],axis= 0)
    Suma_m2 = np.add.reduce(Conjunto_m2[:,:4],axis= 0)
    Suma_m3 = np.add.reduce(Conjunto_m3[:,:4],axis= 0)
    
    
    m1 = np.asarray([Suma_m1[:1]/len(Conjunto_m1),Suma_m1[1:2]/len(Conjunto_m1),Suma_m1[2:3]/len(Conjunto_m1),Suma_m1[3:4]/len(Conjunto_m1)])
    m2 = np.asarray([Suma_m2[:1]/len(Conjunto_m2),Suma_m2[1:2]/len(Conjunto_m2),Suma_m2[2:3]/len(Conjunto_m2),Suma_m2[3:4]/len(Conjunto_m2)])
    m3 = np.asarray([Suma_m3[:1]/len(Conjunto_m3),Suma_m3[1:2]/len(Conjunto_m3),Suma_m3[2:3]/len(Conjunto_m3),Suma_m3[3:4]/len(Conjunto_m3)])
    
    
    return m1,m2,m3

def Verificar_Aciertos(respuesta, patron):
    
    respuesta_1 = 'B'
    respuesta_2 = 'L'
    respuesta_3 = 'R'
    
    if respuesta == 1 and patron [4]==respuesta_1:
        return 1
    if respuesta == 2 and patron [4]==respuesta_2:
        return 1
    if respuesta == 3 and patron [4]==respuesta_3:
        return 1
    
    return 0


def Fase_Clasificacion(C_P,ms):
    cont_Aciertos = 0
    for cp in C_P:
        
        cont_Aciertos = Verificar_Aciertos(Busca_Patron(ms, cp), cp) + cont_Aciertos
    return cont_Aciertos

def Cont_Aciertos(votaciones,C_P):
    aciertos =0;
    for i in range(len(C_P)):
        if C_P[i][4]==votaciones[i]:
            aciertos= aciertos+1
    return aciertos


def Cont_insidencias(Distancias_k):
    
    respuesta_1 = 'B'
    respuesta_2 = 'L'
    respuesta_3 = 'R'
    
    cont_1 = 0
    cont_2 = 0
    cont_3 = 0
    
    for i in Distancias_k:
        if i[1]==respuesta_1:
            cont_1=cont_1+1
        if i[1]==respuesta_2:
            cont_2=cont_2+1
        if i[1]==respuesta_3:
            cont_3=cont_3+1
    
    if cont_1>cont_2 and cont_1>cont_3:
        return respuesta_1
    if cont_2>cont_1 and cont_2>cont_3:
        return respuesta_2
    if cont_3>cont_1 and cont_3>cont_2:
        return respuesta_3
    
    return 
    


def Rendimiento_Hold_Out(aciertos , C_P):
    rendimiento =(aciertos)/len(C_P)
    return rendimiento*100

def Rendimiento_Leave_One_Out(aciertos , C_P):  
    rendimiento =(aciertos)/len(C_P)
    return rendimiento*100

def Rendimiento (aciertos):
    #print(aciertos)
    rendimiento =aciertos/len(matriz)
    return rendimiento*100

def Hold_Out(r, matriz):
    tamaño = len(matriz)//3 
    tamaño_C_F= int (tamaño*(r/100))
    
    clase1 =np.asarray(matriz [:tamaño])
    clase2 =np.asarray(matriz [tamaño:tamaño*2])
    clase3 =np.asarray(matriz [tamaño*2:])

    C_F = np.concatenate((clase1 [0:tamaño_C_F],clase2 [0:tamaño_C_F],clase3 [0:tamaño_C_F]))
    C_P = np.concatenate((clase1 [tamaño_C_F:],clase2 [tamaño_C_F:],clase3 [tamaño_C_F:]))
    return C_F,C_P


def Leave_One_Out(iteracion, matriz):
    iteracion = iteracion + 1
    if iteracion == 1:
        C_P = matriz [:iteracion]
        C_F = matriz [iteracion:len(matriz)]
        
        return C_F,C_P
    else:
        C_P = matriz [iteracion-1:iteracion]
        C_F = np.delete(matriz, iteracion-1 , axis=0) 
        return C_F,C_P


def Recorrer_Leave_One_Out_Clasificador(matriz):
    aciertos=0
    for i in range(len(matriz)):
        Leave_One_Out_aux = Leave_One_Out(i, matriz)
        C_F = Leave_One_Out_aux[0]
        C_P = Leave_One_Out_aux[1]
        #print("C_F",C_F)
        #print("c_p",C_P)
        ms = Fase_Aprendizaje(C_F)
        #print("aciertos: ",Fase_Clasificacion(C_P,ms))
        aciertos=Fase_Clasificacion(C_P,ms)+aciertos
    return aciertos 


def K_Fold_Cross(K, matriz):
    
    tamaño = len(matriz)//3 
    
    if tamaño % K != 0:
        print("\033[1;31m"+" la K tiene que ser exactamente divisible entre cada clase  "+'\033[0;m') 
        return 
        
    clase1 =np.asarray(matriz [:tamaño])
    clase2 =np.asarray(matriz [tamaño:tamaño*2])
    clase3 =np.asarray(matriz [tamaño*2:])
    
    divicion = tamaño // K
    
    clase_1_lista = [np.asarray(clase1 [:divicion])]
    clase_2_lista = [np.asarray(clase2 [:divicion])]
    clase_3_lista = [np.asarray(clase3 [:divicion])]
    
    for i in range(K):
        
        if i != 0:
            clase_1_lista.insert( i , np.asarray(clase1 [(divicion * i):(divicion*(i+1) )]) )
            clase_2_lista.insert( i , np.asarray(clase2 [(divicion * i):(divicion*(i+1) )]) )
            clase_3_lista.insert( i , np.asarray(clase3 [(divicion * i):(divicion*(i+1) )]) )
    
    B_D = np.concatenate(( clase_1_lista[0],clase_2_lista[0] ,clase_3_lista[0]),  axis=0)
    B_D_lista = []
    
    for i in range(K):
        B_D = np.concatenate(( clase_1_lista[i],clase_2_lista[i] ,clase_3_lista[i]),  axis=0)
        B_D_lista.insert(i,B_D)

    return B_D_lista


def Recorrer_K_Fold_Cross_Clasificador(B_D_lista):
    C_P=[]
    C_F=[]
    C_F_Partes=[]
    aux=0
    aciertos = 0
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
            
            if i == 0:
                C_F = C_F_Partes[i]
            else:
                C_F = np.concatenate((C_F,C_F_Partes[i]))
        
        ms = Fase_Aprendizaje(C_F)
        
        aciertos=Fase_Clasificacion(C_P,ms)+aciertos
        
    return aciertos


def Knn_Clasificador(patron_desconocido,C_F,K):
    
    Distancias = []
    cont = 0
    for i in C_F:
        cont= 1 + cont
        if cont == 1:
            Distancias=Extraer_Distancias(patron_desconocido,i)
        else:
            Distancias=np.concatenate((Distancias,Extraer_Distancias(patron_desconocido,i)))
            
    np.array(Distancias)
    Distancias_ord =sorted(Distancias, key= lambda Distancias : Distancias[0])
    Distancias_k=Distancias_ord[:k]
    votacion_may = Cont_insidencias(Distancias_k)
    return votacion_may


def Hold_Out_Clasificador_Knn(C_P,C_F,k):
    Votacion=[]
    np.asarray(Votacion)
    cont= 0
    for i in C_P:
        cont= 1+ cont
        if cont == 1:
            Votacion=[Knn_Clasificador(i,C_F,k)]
        else:
            Votacion = Votacion + [Knn_Clasificador(i,C_F,k)]
            
    return Votacion

def Extraer_Distancias(patron_desconocido,patron_conocido):
    
    Distancia =  math.sqrt(  (patron_desconocido[0]-patron_conocido[0])**2 
                           + (patron_desconocido[1]-patron_conocido[1])**2 
                           + (patron_desconocido[2]-patron_conocido[2])**2 
                           + (patron_desconocido[3]-patron_conocido[3])**2)
    
    patron= [[Distancia,patron_conocido[4]]]
    return patron

def Recorrer_Leave_One_Out_Knn_Clasificador(matriz,K):
    Votaciones= []
    for i in range(len(matriz)):
        Leave_One_Out_aux = Leave_One_Out(i, matriz)
        C_F = Leave_One_Out_aux[0]
        C_P = Leave_One_Out_aux[1]
        
        if i == 0:
            Votaciones=[Knn_Clasificador(C_P[0],C_F,K)]
        else:
            Votaciones = Votaciones + [Knn_Clasificador(C_P[0],C_F,K)]
            
    return Votaciones 

def Busca_Patron(ms,patron_desconocido):
    #print('________________________________________')
    #print(patron_desconocido)
    Distancia_1 =  math.sqrt(  (ms[0][0]-patron_desconocido[0])**2 
                             + (ms[0][1]-patron_desconocido[1])**2 
                             + (ms[0][2]-patron_desconocido[2])**2 
                             + (ms[0][3]-patron_desconocido[3])**2)
    
    
    Distancia_2 =  math.sqrt(  (ms[1][0]-patron_desconocido[0])**2 
                             + (ms[1][1]-patron_desconocido[1])**2 
                             + (ms[1][2]-patron_desconocido[2])**2 
                             + (ms[1][3]-patron_desconocido[3])**2)
    
    
    Distancia_3 =  math.sqrt(  (ms[2][0]-patron_desconocido[0])**2 
                             + (ms[2][1]-patron_desconocido[1])**2 
                             + (ms[2][2]-patron_desconocido[2])**2 
                             + (ms[2][3]-patron_desconocido[3])**2)
    
    
    
    if Distancia_1<Distancia_2 and Distancia_1<Distancia_3:
        #print("resultado 1")
        return 1
    if Distancia_2<Distancia_1 and Distancia_2<Distancia_3:
        #print("resultado 2")
        return 2
    if Distancia_3<Distancia_1 and Distancia_3<Distancia_2:
        #print("resultado 3")
        return 3
        
    return


def Recorrer_K_Fold_Cross_Clasificador_Knn(B_D_lista,K):
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
            
            if i == 0:
                C_F = C_F_Partes[i]
            else:
                C_F = np.concatenate((C_F,C_F_Partes[i]))
        if i == 0:
            votaciones = Hold_Out_Clasificador_Knn(C_P,C_F,K)
        else:
            votaciones = np.concatenate((votaciones,Hold_Out_Clasificador_Knn(C_P,C_F,K)))
    return votaciones

def Unir_Listas__K_Fold_Cross(B_D_lista):
    
    Union=[]
    for i in range(len(B_D_lista)):
        if i == 0:
            Union = B_D_lista[i]
        else:
            Union = np.concatenate((Union,B_D_lista[i]))
    return Union


if __name__ == '__main__':
    archivo = pd.read_csv(".//balance-scale.csv",header=0)

    matriz = np.asarray(archivo)

    C_F = Hold_Out(70, matriz)[0]
    C_P = Hold_Out(70, matriz)[1]

    ms = Fase_Aprendizaje(C_F)

    print("\nClasificador Euclidiano")
    print ( "Rendimiento Hold Out  :" ,round(Rendimiento_Hold_Out(Fase_Clasificacion(C_P,ms),C_P),2),"%")
    print ("   ")

    print("Rendimiento K Fold Cross :" ,Rendimiento(Recorrer_K_Fold_Cross_Clasificador( K_Fold_Cross(2, matriz))), "%")
    print ("   ")

    print("Rendimiento leave One Out :" ,round( Rendimiento(Recorrer_Leave_One_Out_Clasificador(matriz)),2),"%")

    k=3
    print("\nClasificador Knn", "con K: ",k)
    print ( "Rendimiento Hold Out  :" ,round(Rendimiento_Hold_Out(Cont_Aciertos(Hold_Out_Clasificador_Knn(C_P,C_F,k),C_P),C_P),2),"%")
    print ("   ")
    print("Rendimiento K Fold Cross :" ,round(Rendimiento(Cont_Aciertos(Recorrer_K_Fold_Cross_Clasificador_Knn( K_Fold_Cross(5, matriz),k),Unir_Listas__K_Fold_Cross(K_Fold_Cross(5, matriz)))),2),"%")
    print ("   ")
    print("Rendimiento leave One Out :" ,round( Rendimiento(Cont_Aciertos(Recorrer_Leave_One_Out_Knn_Clasificador(matriz,k),matriz)),2),"%")

    k=5
    print("\nClasificador Knn", "con K: ",k)
    print ( "Rendimiento Hold Out  :" ,round(Rendimiento_Hold_Out(Cont_Aciertos(Hold_Out_Clasificador_Knn(C_P,C_F,k),C_P),C_P),2),"%")
    print ("   ")
    print("Rendimiento K Fold Cross :" ,round(Rendimiento(Cont_Aciertos(Recorrer_K_Fold_Cross_Clasificador_Knn( K_Fold_Cross(5, matriz),k),Unir_Listas__K_Fold_Cross(K_Fold_Cross(5, matriz)))),2),"%")
    print ("   ")
    print("Rendimiento leave One Out :" ,round( Rendimiento(Cont_Aciertos(Recorrer_Leave_One_Out_Knn_Clasificador(matriz,k),matriz)),2),"%")

    k=7
    print("\nClasificador Knn", "con K: ",k)
    print ( "Rendimiento Hold Out  :" ,round(Rendimiento_Hold_Out(Cont_Aciertos(Hold_Out_Clasificador_Knn(C_P,C_F,k),C_P),C_P),2),"%")
    print ("   ")
    print("Rendimiento K Fold Cross :" ,round(Rendimiento(Cont_Aciertos(Recorrer_K_Fold_Cross_Clasificador_Knn( K_Fold_Cross(5, matriz),k),Unir_Listas__K_Fold_Cross(K_Fold_Cross(5, matriz)))),2),"%")
    print ("   ")
    print("Rendimiento leave One Out :" ,round( Rendimiento(Cont_Aciertos(Recorrer_Leave_One_Out_Knn_Clasificador(matriz,k),matriz)),2),"%")

    k=9
    print("\nClasificador Knn", "con K: ",k)
    print ( "Rendimiento Hold Out  :" ,round(Rendimiento_Hold_Out(Cont_Aciertos(Hold_Out_Clasificador_Knn(C_P,C_F,k),C_P),C_P),2),"%")
    print ("   ")
    print("Rendimiento K Fold Cross :" ,round(Rendimiento(Cont_Aciertos(Recorrer_K_Fold_Cross_Clasificador_Knn( K_Fold_Cross(5, matriz),k),Unir_Listas__K_Fold_Cross(K_Fold_Cross(5, matriz)))),2),"%")
    print ("   ")
    print("Rendimiento leave One Out :" ,round( Rendimiento(Cont_Aciertos(Recorrer_Leave_One_Out_Knn_Clasificador(matriz,k),matriz)),2),"%")

    k=11
    print("\nClasificador Knn", "con K: ",k)
    print ( "Rendimiento Hold Out  :" ,round(Rendimiento_Hold_Out(Cont_Aciertos(Hold_Out_Clasificador_Knn(C_P,C_F,k),C_P),C_P),2),"%")
    print ("   ")
    print("Rendimiento K Fold Cross :" ,round(Rendimiento(Cont_Aciertos(Recorrer_K_Fold_Cross_Clasificador_Knn( K_Fold_Cross(5, matriz),k),Unir_Listas__K_Fold_Cross(K_Fold_Cross(5, matriz)))),2),"%")
    print ("   ")
    print("Rendimiento leave One Out :" ,round( Rendimiento(Cont_Aciertos(Recorrer_Leave_One_Out_Knn_Clasificador(matriz,k),matriz)),2),"%")
