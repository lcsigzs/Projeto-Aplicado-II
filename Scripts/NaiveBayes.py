# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:07:37 2025

@author: Oscar Augusto de Oliveira Luz

Disciplina: Projeto Aplicado II

Grupo 13
Thaís Cristine de Andrade Gomes - 10721642
Paulo Ricardo de Oliveira Ramos - 10721464
Lucas Iglezias dos Anjos - 10433522
Oscar Augusto de Oliveira Luz - 10435099
"""

import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Carregar os conjuntos de dados e o vetorizador
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

# Chamar e treinar o modelo Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Avaliar o modelo
acuracia = accuracy_score(y_test, y_pred)
relatorio = classification_report(y_test, y_pred)

# Salvar e apresentar os resultados
joblib.dump(model, "multinomial_nb_model.pkl")

print("\n--- Resultados da Avaliação ---")
print(f"Acurácia: {acuracia:.4f}")
print("\nRelatório de Classificação:\n", relatorio)
print("\nModelo treinado salvo como multinomial_nb_model.pkl")