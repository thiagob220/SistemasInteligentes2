#Aluno: Thiago Berto Minson
#RA: 2270412
#Professor: Rafael Gomes Mantovani
#Disciplina: Sistemas Inteligentes 2 
#---------------------------------------------------------#
#Atividade 2 - Perceptron - Teste com Situações Diferentes
#---------------------------------------------------------#
# Etapa Principal: Desenvolvimento do Código Base para 
# aplicação genérica


#---------------------------------------------------------#
#Variáveis de inicialização - Caso simples 1
#---------------------------------------------------------#
w1 = [-0.5441, 0.5562, 0.4074] #pesos
x1 = [[2, 4], [2, 4]] #entradas
d1 = [1, 0] #classe
n1 = 0.1 #taxa de aprendizado
bias1 = -1 #viés
e1 = True #erro inicial verdadeiro
e_list1 = [e1] #lista de erros inicial
#---------------------------------------------------------#

#---------------------------------------------------------#
#Variáveis de inicialização - Caso simples 2 (do OR)
#---------------------------------------------------------#
w2 = [0.5, 0.5, 0.5] #pesos
x2 = [[0, 0, 1, 1], [0, 1, 0, 1]] #entradas
d2 = [0, 1, 1, 1] #classe
n2 = 0.1 #taxa de aprendizado
bias2 = 1 #viés
e2 = True #erro inicial verdadeiro
e_list2 = [e2] #lista de erros inicial
#---------------------------------------------------------#




#---------------------------------------------------------#
#Função genérica do sinal de saída do neurônio
#---------------------------------------------------------#
#recebe v (spike) e verifica se positivo
def phi(v):
    if v >= 0:
        print("phi(v) = 1") #pra mostrar no prompt
        return 1
    else:
        print("phi(v) = 0") #pra mostrar no prompt
        return 0
#---------------------------------------------------------#

    

#---------------------------------------------------------#   
#Função genérica de alteração dos pesos (ajuste)
#---------------------------------------------------------#
#recebe w (pesos), n (taxa de aprendizado),
# vp (numero do peso respectivo), d (desejado), y (obtido) e x (os outros valores) 
def wn(vp, w, n, d, y, x):
    print("Ajuste de peso", vp, ": de", w, "para", w+n*(d-y)*x) #pra mostrar no prompt
    return w+n*(d-y)*x
#---------------------------------------------------------#



#---------------------------------------------------------#
#Função genérica do PERCEPTRON
#---------------------------------------------------------#
#e(erro), b(bias ou viés), w(vetor inicial de pesos), 
# n(taxa de aprendizado), d(resultado esperado),
#x (entradas)
def perceptron(e, e_list, b, w, n, d, x):
    #prints iniciais de contexto
    print("--- Iniciando o Perceptron ---")
    print("Pesos Iniciais: ", w) #pesos
    print("Bias: ", b) #bias
    print("Vetores de entrada: ", x) #entradas x
    print("Saída esperada: ", d) #saida
    print("-----------------------------------------------------------------\n-----------------------------------------------------------------")

    #variáveis internas da função
    ep_c = 0 #contador das épocas
    a_len = len(x[0]) #tamanho das execuções para cada época equivalente ao tamanho de cada x
    while(True in e_list): #representa uma época inteira
        e_list = [] #lista que diz se há erro em cada parte da época (E)
        count = 0 #contador pras iterações internas pra uma época
        print("-----------------------------------------------------------------\nÉpoca", ep_c+1,"\n-----------------------------------------------------------------")
        while(count < a_len): #representa todas as iterações até o fim de uma época
            print("-----------------------------------------------------------------\nE", count+1,"\n-----------------------------------------------------------------")
            v = 0 #inicialização da variável v (sinal spike)
            l_t_p = [] #lista temporária dos pesos pra iteração

            #faz o cálculo dos produtos entre os pesos e a entrada (carga (v))
            for i in range (len(w)):
                if i == 0: #iteração do bias
                    l_t_p.append(w[i]*b)
                else: #demais iterações
                    l_t_p.append(w[i]*x[i-1][count]) #-1 pois pega o primeiro elemento
    
            #faz o resultado de v da iteração
            for i in l_t_p:
                v += i

            y = phi(v) #resultado da função de ativação (y)

            #verificando se era o esperado
            if y != d[count]:
                e = True
            else:
                e = False
    
            #realizando a alteração dos pesos
            #lista auxiliar para os novos pesos
            w_aux = []
            if e == True: #há erro
                for i in range(len(w)):
                    if i == 0: #iteração do bias
                        w_p = wn(i, w[i], n, d[count], y, b)
                    else: #iterações restantes
                        w_p = wn(i, w[i], n, d[count], y, x[i-1][count])
                    w_aux.append(w_p)
                w = w_aux

            print("Pesos atuais: ", w) #pesos
            print("-----------------------------------------------------------------")

            e_list.append(e) #colocando a ocorrência de erro
            print(e_list) #mostrando como está 
            print("-----------------------------------------------------------------\n\n")
            count+=1 #sobe pro próximo conjunto da época

        ep_c+=1 #aumenta uma época
    
#---------------------------------------------------------#
#Execuções das Atividades
#---------------------------------------------------------#
#Estão comentados, só descomentar para verificar :)
#Exercício 1 -
#perceptron(e1, e_list1, bias1, w1, n1, d1, x1)   
#Exercício 2 - 
perceptron(e2, e_list2, bias2, w2, n2, d2, x2) 
#---------------------------------------------------------#

    



    
    

