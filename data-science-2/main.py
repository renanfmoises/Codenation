#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as sct
import seaborn as sns


# In[23]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[24]:


athletes = pd.read_csv("athletes.csv")


# In[25]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[26]:


# Sua análise começa aqui.
print(athletes.shape)

athletes.head(10)


# In[27]:


print(athletes['height'].mean())
print(athletes['height'].std())


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# #### A. Formulação da hipotese
# $H_0$ = A amostra provém de uma população normal
# 
# $H_1$ = Não podemos afirmar que a amostra provém de uma população normal
# 
# #### B. Definir o nível de significância do teste ($\alpha$)
# 
# $\alpha$ = 0.05
# 
# #### 3. Calcular o coeficiente de Shapiro-Wilk e o p-valor
#  
# `sct.shapiro()`
# 
# #### 4. Fazer a análise do resultado
# 
# Se $p-valor > \alpha$ aceita-se $H_0$ (hipótese nula)
# 
# Se $p-valor \le \alpha$ não é possível rejeitar $H_0$ (hipótese nula)

# In[50]:


def q1():
    #hist_atlhetes = sns.distplot(athletes['height'])
    
    # H0 = As alturas são normalmente distribuidas
    # H1 = Não podemos afirmar que as alturas são normalmente distribuidas
    
    alpha = 0.05
    
    sample_ht = get_sample(athletes, 'height', n = 3000)
    
    stats, p = sct.shapiro(sample_ht)
    
    print(p)
    
    if p > alpha:
        return True
    else:
        return False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[61]:


sample_ht = get_sample(athletes, 'height', n = 3000)

hist_ath_height = sns.distplot(sample_ht, bins = 25)


# In[36]:


sm.qqplot(sample_ht, fit = True, line = '45');


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[63]:


def q2():
    # Retorne aqui o resultado da questão 2.

    alpha = 0.05
    
    sample_ht = get_sample(athletes, 'height', n = 3000)
    
    print(sct.jarque_bera(sample_ht))
    
    return sct.jarque_bera(sample_ht)[1] > alpha


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[62]:


def q3():
    # Retorne aqui o resultado da questão 3.

    alpha = 0.05
    
    sample_wt = get_sample(athletes, 'weight', n = 3000)
    
    stats, p = sct.normaltest(sample_wt)

    print(p)
    
    if p > alpha:
        return True
    else:
        return False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[59]:


sample_wt = get_sample(athletes, 'weight', n = 3000)

hist_ath_weight = sns.distplot(sample_wt, bins = 25)


# In[60]:


sns.boxplot(sample_wt);


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[52]:


def q4():
    # Retorne aqui o resultado da questão 4.
    
    alpha = 0.05
    
    sample_ht = get_sample(athletes, 'height', n = 3000)
    
    sample_wt = get_sample(athletes, 'weight', n = 3000)
    
    log_wt = sct.expon.rvs(sample_wt)
    
    stats, p = sct.normaltest(log_wt)
    
    print(p)
    
    if p > alpha:
        return True
    else:
        return False


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[53]:


sample_wt = get_sample(athletes, 'weight', n = 3000)

log_wt = sct.expon.rvs(sample_wt)

sns.distplot(sample_wt, bins = 25, label = 'D\'agostino-Pearson')
sns.distplot(log_wt, bins = 25, label = 'D\'agostino-Pearson - log')
plt.legend();


# 

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[54]:


ath_bra = athletes[athletes['nationality'] == 'BRA']
ath_usa = athletes[athletes['nationality'] == 'USA']
ath_can = athletes[athletes['nationality'] == 'CAN']

print(len(ath_bra))
print(len(ath_usa))
print(len(ath_can))

print('\n')

print(ath_bra['height'].var())
print(ath_usa['height'].var())
print(ath_can['height'].var())


# In[55]:


def q5():
    # Retorne aqui o resultado da questão 5.
    
    alpha = 0.05
    
    stat_brausa, p_brausa = sct.ttest_ind(ath_bra['height'], ath_usa['height'], equal_var = False, nan_policy = 'omit')

    if p_brausa > alpha:
        return True # H0: as médias são iguais
    else:
        return False # H1: não podemos dizer que as médias são iguais


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[56]:


def q6():
    # Retorne aqui o resultado da questão 6.
    alpha = 0.05
    
    stat_bracan, p_bracan = sct.ttest_ind(ath_bra['height'], ath_can['height'], equal_var = False, nan_policy = 'omit')

    if p_bracan > alpha:
        return True # H0: as médias são iguais
    else:
        return False # H1: não podemos dizer que as médias são iguais


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[57]:


def q7():
    alpha = 0.05
    
    stat_usacan, p_usacan = sct.ttest_ind(ath_usa['height'], ath_can['height'], equal_var = False, nan_policy = 'omit')

    return round(p_usacan, 8)


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
