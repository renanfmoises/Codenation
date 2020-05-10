#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


black_friday.head()


# In[4]:


black_friday.info()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[27]:


def q1():
    
    return tuple(black_friday.shape)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[28]:


def q2():
    usuarios_unicos = black_friday[['User_ID', 'Gender', 'Age']].drop_duplicates(subset = 'User_ID')

    mulheres_2635 = usuarios_unicos[(usuarios_unicos['Gender'] == 'F')
                                & (usuarios_unicos['Age'] == '26-35')]
    
    return int(len(mulheres_2635))


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[29]:


def q3():
    
    return int(black_friday['User_ID'].nunique())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[30]:


def q4():
    
    return int(black_friday.dtypes.nunique())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[31]:


def q5():
    
    return float(round(1 - (len(black_friday.dropna(axis=0)) / len(black_friday)), 4))


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[32]:


def q6():
    
    return int(black_friday.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[33]:


def q7():
    
    return black_friday['Product_Category_3'].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[36]:


def q8():
    purchase = black_friday['Purchase']
    
    norm = (purchase - purchase.min()) / (purchase.max() - purchase.min())
    
    return float(norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[42]:


def q9():
    purchase = black_friday['Purchase']
    
    padr = (purchase - purchase.mean()) / purchase.std()
    
    return int(((padr >= -1) & (padr <= 1)).sum())


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[41]:


def q10():
    pcat2_na = black_friday[black_friday['Product_Category_2'].isnull()]

    return bool(pcat2_na['Product_Category_3'].isna().all())

