#Aluno: Thiago Berto Minson
#RA: 2270412
#Professor: Rafael Gomes Mantovani
#Disciplina: Sistemas Inteligentes 2 
#---------------------------------------------------------#
#Atividade 2 - Perceptron - Teste com Situações Diferentes
#---------------------------------------------------------#
#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
#---------------------------------------------------------#
# Função de ativação degrau
# Entrada: valor de ativação v
# Saída: 1 se v >= 0, senão 0
#---------------------------------------------------------#
def phi(v):
    return 1 if v >= 0 else 0  # Retorna 1 se v >= 0, senão 0

#---------------------------------------------------------#
# Função de atualização dos pesos
# Entrada: índice do peso (não usado), peso atual, taxa de aprendizado,
#          saída desejada, saída obtida, entrada correspondente
# Saída: novo valor do peso
#---------------------------------------------------------#
def wn(vp, w, n, d, y, x):
    return w + n * (d - y) * x  # Regra de aprendizado do Perceptron

#---------------------------------------------------------#
# Função principal do Perceptron
# Entrada: erro inicial, lista de erro, bias, lista de pesos,
#          taxa de aprendizado, saídas desejadas, entradas, flag para gráfico, título
# Saída: nenhuma (imprime gráfico com fronteira de decisão)
#---------------------------------------------------------#
def perceptron(e, e_list, b, w, n, d, x, exibir_grafico=False, titulo=""):
    ep_c = 0  # Contador de épocas
    a_len = len(x[0])  # Número de amostras

    while(True in e_list):
        if ep_c > 100000:
            raise RuntimeError("Perceptron não convergiu após 100000 épocas.")  # Limite de épocas
        e_list = []  # Zera a lista de erros
        count = 0
        while(count < a_len):  # Para cada amostra
            v = 0  # Valor de ativação
            l_t_p = []  # Lista de produtos peso*entrada
            for i in range(len(w)):
                if i == 0:
                    l_t_p.append(w[i] * b)  # Produto do bias
                else:
                    l_t_p.append(w[i] * x[i-1][count])  # Produto peso*entrada
            for i in l_t_p:
                v += i  # Soma os produtos
            y = phi(v)  # Saída do neurônio

            if y != d[count]:
                e = True  # Houve erro
            else:
                e = False  # Sem erro

            w_aux = []
            if e:  # Se houve erro, ajusta os pesos
                for i in range(len(w)):
                    if i == 0:
                        w_p = wn(i, w[i], n, d[count], y, b)  # Atualiza peso do bias
                    else:
                        w_p = wn(i, w[i], n, d[count], y, x[i-1][count])  # Atualiza pesos das entradas
                    w_aux.append(w_p)
                w = w_aux  # Aplica pesos ajustados
            e_list.append(e)  # Registra se houve erro nessa amostra
            count += 1  # Próxima amostra
        ep_c += 1  # Incrementa a época

    # Geração do gráfico com a fronteira de decisão
    if exibir_grafico:
        plt.figure(figsize=(8, 6))
        for classe in set(d):
            indices = [i for i, val in enumerate(d) if val == classe]
            plt.scatter([x[0][i] for i in indices], [x[1][i] for i in indices], label=f"Classe {classe}")
        x_vals = np.array(plt.gca().get_xlim())  # Pega limites do eixo X
        if w[2] != 0:  # Evita divisão por zero
            y_vals = -(w[0] + w[1] * x_vals) / w[2]  # Equação da reta
            plt.plot(x_vals, y_vals, 'k--', label='Fronteira de decisão')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'{titulo}\nÉpocas: {ep_c}')  # Título do gráfico
        plt.legend()
        plt.grid(True)
        plt.show()

#---------------------------------------------------------#
# Função para rodar o perceptron com conjuntos de dados
# Entrada: vetores de entrada x1 e x2, vetor de saída, título
#---------------------------------------------------------#
def run_dataset(x1, x2, saida, titulo):
    x = [x1, x2]  # Entradas organizadas em listas
    d = saida  # Saídas desejadas
    b = 1  # Bias
    w = [0.1, 0.1, 0.1]  # Pesos iniciais
    n = 0.1  # Taxa de aprendizado
    e = True
    e_list = [e]
    perceptron(e, e_list, b, w, n, d, x, exibir_grafico=True, titulo=titulo)

#---------------------------------------------------------#
# Funções separadas para cada exemplo
#---------------------------------------------------------#
def exemplo_or():
    x = [[0, 0, 1, 1], [0, 1, 0, 1]]  # Entradas OR
    d = [0, 1, 1, 1]  # Saídas OR
    run_dataset(x[0], x[1], d, "Porta OR")

def exemplo_xor():
    x = [[0, 0, 1, 1], [0, 1, 0, 1]]  # Entradas XOR
    d = [0, 1, 1, 0]  # Saídas XOR
    run_dataset(x[0], x[1], d, "Porta XOR (não separável)")

def exemplo_and():
    x = [[0, 0, 1, 1], [0, 1, 0, 1]]  # Entradas AND
    d = [0, 0, 0, 1]  # Saídas AND
    run_dataset(x[0], x[1], d, "Porta AND")

def exemplo_iris():
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)  # Dados do Iris
    df_iris['label'] = iris.target  # Rótulos
    df_iris = df_iris[df_iris['label'] != 2]  # Apenas Setosa e Versicolor
    df_iris = df_iris[['sepal length (cm)', 'sepal width (cm)', 'label']]  # Seleciona colunas
    df_iris.columns = ['x1', 'x2', 'saida']  # Renomeia
    run_dataset(df_iris['x1'].tolist(), df_iris['x2'].tolist(), df_iris['saida'].tolist(), "Iris (Setosa x Versicolor)")

#---------------------------------------------------------#
# Execução individual — descomente o exemplo desejado
#---------------------------------------------------------#
if __name__ == "__main__":
    #exemplo_or()
    #exemplo_xor()
    #exemplo_and()
    #exemplo_iris()
#---------------------------------------------------------#