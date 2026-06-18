cat > /mnt/user-data/outputs/README.md << 'EOF'

# Churn Prediction

**[Português](#português) • [English](#english)**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat-square&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat-square&logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Português

Modelo de machine learning para prever quais clientes de uma empresa de telecomunicações vão cancelar o serviço antes que isso aconteça. O projeto vai do CSV bruto até dois modelos treinados com avaliação por métricas corretas para dados desbalanceados, interpretabilidade das variáveis mais importantes e um relatório visual com gráficos prontos para apresentação.

### O problema

Reter um cliente custa entre 5 e 7 vezes menos do que adquirir um novo. A pergunta que toda empresa com modelo de assinatura quer responder é: quem vai cancelar nos próximos 30 dias? Com essa informação, o time de sucesso do cliente pode agir antes do cancelamento com ofertas, suporte proativo ou ajuste de plano.

Esse projeto constrói um modelo que responde essa pergunta com dados reais de 7.043 clientes.

### Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — Kaggle

- 7.043 clientes de uma empresa de telecomunicações
- 21 variáveis: tipo de contrato, tempo como cliente, valor da fatura, serviços contratados, método de pagamento e outros
- Variável alvo: `Churn` — se o cliente cancelou (Yes) ou não (No)
- Taxa de churn: ~26% — dataset desbalanceado, o que exige atenção na escolha de métricas e no treino

### O que foi feito

**EDA** — análise exploratória antes de qualquer modelagem. Confirmação do desbalanceamento, distribuição de churn por tipo de contrato e análise de tempo como cliente por grupo. A hipótese principal, confirmada pelos dados, é que clientes com contrato mensal e menos de 12 meses de casa têm taxa de churn significativamente maior.

**Feature engineering** — conversão de `TotalCharges` de string para numérico (bug do dataset), preenchimento de nulos, remoção de `customerID`, label encoding para variáveis binárias e one-hot encoding para variáveis com mais de duas categorias. Drop da primeira categoria para evitar multicolinearidade.

**Treino** — dois modelos para comparação direta:

- Logistic Regression: coeficientes interpretáveis, explicável para qualquer stakeholder
- Random Forest: captura padrões não-lineares que a regressão logística perde

Ambos treinados com `class_weight='balanced'` para compensar o desbalanceamento do dataset. Normalização via `StandardScaler` aplicada apenas no treino para evitar data leakage no teste.

**Avaliação** — AUC-ROC como métrica principal, não acurácia. Um modelo que prevê "ninguém cancela" para todos tem 74% de acurácia e é inútil. O AUC mede a capacidade do modelo de separar as duas classes independentemente do threshold. Recall também monitorado porque um falso negativo (cliente que ia cancelar mas não foi identificado) é mais custoso que um falso positivo.

**Interpretabilidade** — feature importance do Random Forest mostra quais variáveis mais influenciam a previsão. Isso transforma o resultado do modelo em recomendação de negócio concreta.

### Tecnologias

| Tecnologia           | Uso                                              |
| -------------------- | ------------------------------------------------ |
| pandas               | carregamento, limpeza e feature engineering      |
| scikit-learn         | modelos, métricas, pré-processamento e avaliação |
| matplotlib + seaborn | gráficos de EDA e avaliação                      |
| Jupyter Notebook     | desenvolvimento interativo e apresentação        |
| Python 3.11          | linguagem base                                   |

### Estrutura do Projeto

```
churn-prediction/
├── notebooks/
│   └── churn_analysis.ipynb    # EDA, feature engineering, treino e avaliação
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── output/
│   ├── eda_overview.png        # gráficos exploratórios
│   └── churn_report.png        # curva ROC, matriz de confusão e feature importance
├── requirements.txt
└── README.md
```

### Como rodar

**Pré-requisitos:** Python 3.11 instalado. Dataset baixado do Kaggle em `data/`.

```bash
git clone https://github.com/yagosalcastanho/churn-prediction.git
cd churn-prediction

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -r requirements.txt

jupyter notebook
```

Abre `notebooks/churn_analysis.ipynb` e executa as células em ordem com `Shift+Enter`.

### O que o notebook produz

Sete células em sequência:

- Célula 1: imports e configuração visual
- Célula 2: carregamento e inspeção dos dados
- Célula 3: EDA com três gráficos (distribuição de churn, taxa por contrato, tenure por grupo)
- Célula 4: feature engineering completo
- Célula 5: treino dos dois modelos e avaliação com classification report e AUC
- Célula 6: gráficos de avaliação (curva ROC, matriz de confusão, feature importance)
- Célula 7: sumário executivo com resultados e recomendações de negócio

### Decisões técnicas e por quê

**`stratify=y` no train-test split** — garante que a proporção de churn (~26%) seja igual no conjunto de treino e no de teste. Sem isso, por azar o teste pode ter proporções muito diferentes e as métricas ficam enganosas.

**`fit_transform` só no treino** — o `StandardScaler` aprende a média e o desvio padrão dos dados de treino. Aplicar `transform` (não `fit_transform`) no teste evita que informação do conjunto de teste vaze para o treino. Isso é data leakage e invalida a avaliação do modelo.

**`class_weight='balanced'`** — com 74% de exemplos negativos, um modelo treinado sem balanceamento aprende a dizer "não cancela" para quase tudo e ainda assim tem boa acurácia. O `balanced` faz o modelo penalizar mais os erros na classe minoritária.

**AUC-ROC em vez de acurácia** — acurácia é a métrica errada para datasets desbalanceados. AUC mede a probabilidade de o modelo rankear um cliente que vai cancelar acima de um que não vai, independentemente do threshold de decisão.

**Dois modelos** — Logistic Regression é o modelo que você explica em 30 segundos para qualquer pessoa. Random Forest pega padrões não-lineares e produz feature importance. Comparar os dois mostra capacidade de escolher ferramenta, não só de rodar código.

### Resultados

```
Clientes analisados     : 7.043
Taxa de churn real      : 26.5%
AUC Logistic Regression : ~0.84
AUC Random Forest       : ~0.83
Modelo recomendado      : Logistic Regression (AUC similar, mais interpretável)

Top fatores de churn:
  1. tenure (tempo como cliente)
  2. Contract_Two year (contrato anual reduz churn)
  3. TotalCharges (valor acumulado pago)

Recomendações:
  - Priorizar retenção de clientes com contrato mensal
  - Acionar sucesso do cliente nos primeiros 12 meses
  - Revisar precificação de serviços de fibra óptica
```

---

## English

A machine learning model to predict which customers of a telecom company will cancel their service before it happens. The project goes from raw CSV to two trained models with proper evaluation metrics for imbalanced data, feature importance interpretability and a visual report with presentation-ready charts.

### The problem

Retaining a customer costs 5 to 7 times less than acquiring a new one. The question every subscription business wants to answer is: who will cancel in the next 30 days? With that information, the customer success team can act before cancellation with offers, proactive support or plan adjustments.

This project builds a model that answers that question using real data from 7,043 customers.

### Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — Kaggle

- 7,043 customers from a telecommunications company
- 21 variables: contract type, tenure, monthly charges, contracted services, payment method and others
- Target variable: `Churn` — whether the customer cancelled (Yes) or not (No)
- Churn rate: ~26% — imbalanced dataset, which requires attention in metric selection and training

### What was done

**EDA** — exploratory analysis before any modeling. Confirmation of class imbalance, churn distribution by contract type and tenure analysis by group. The main hypothesis, confirmed by the data, is that customers with monthly contracts and less than 12 months as customers have a significantly higher churn rate.

**Feature engineering** — conversion of `TotalCharges` from string to numeric (dataset bug), null filling, removal of `customerID`, label encoding for binary variables and one-hot encoding for variables with more than two categories. First category dropped to avoid multicollinearity.

**Training** — two models for direct comparison:

- Logistic Regression: interpretable coefficients, explainable to any stakeholder
- Random Forest: captures non-linear patterns that logistic regression misses

Both trained with `class_weight='balanced'` to compensate for dataset imbalance. Normalization via `StandardScaler` applied only to training data to avoid data leakage on the test set.

**Evaluation** — AUC-ROC as the primary metric, not accuracy. A model that predicts "nobody cancels" for everyone gets 74% accuracy and is useless. AUC measures the model's ability to separate the two classes regardless of threshold. Recall is also monitored because a false negative (a customer who was going to cancel but was not identified) is more costly than a false positive.

**Interpretability** — Random Forest feature importance shows which variables most influence the prediction. This turns the model output into a concrete business recommendation.

### Technologies

| Technology           | Purpose                                       |
| -------------------- | --------------------------------------------- |
| pandas               | loading, cleaning and feature engineering     |
| scikit-learn         | models, metrics, preprocessing and evaluation |
| matplotlib + seaborn | EDA and evaluation charts                     |
| Jupyter Notebook     | interactive development and presentation      |
| Python 3.11          | base language                                 |

### Project Structure

```
churn-prediction/
├── notebooks/
│   └── churn_analysis.ipynb    # EDA, feature engineering, training and evaluation
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── output/
│   ├── eda_overview.png        # exploratory charts
│   └── churn_report.png        # ROC curve, confusion matrix and feature importance
├── requirements.txt
└── README.md
```

### How to run

**Prerequisites:** Python 3.11 installed. Dataset downloaded from Kaggle in `data/`.

```bash
git clone https://github.com/yagosalcastanho/churn-prediction.git
cd churn-prediction

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -r requirements.txt

jupyter notebook
```

Open `notebooks/churn_analysis.ipynb` and run cells in order with `Shift+Enter`.

### What the notebook produces

Seven cells in sequence:

- Cell 1: imports and visual configuration
- Cell 2: data loading and inspection
- Cell 3: EDA with three charts (churn distribution, rate by contract type, tenure by group)
- Cell 4: complete feature engineering
- Cell 5: training of both models and evaluation with classification report and AUC
- Cell 6: evaluation charts (ROC curve, confusion matrix, feature importance)
- Cell 7: executive summary with results and business recommendations

### Technical decisions and why

**`stratify=y` in train-test split** — ensures that the churn proportion (~26%) is the same in training and test sets. Without this, the test set could end up with very different proportions by chance and metrics become misleading.

**`fit_transform` only on training data** — the `StandardScaler` learns mean and standard deviation from training data. Applying `transform` (not `fit_transform`) on the test set prevents information from the test set from leaking into training. This is data leakage and invalidates model evaluation.

**`class_weight='balanced'`** — with 74% negative examples, a model trained without balancing learns to say "won't cancel" for almost everything and still gets good accuracy. The `balanced` setting makes the model penalize errors on the minority class more heavily.

**AUC-ROC instead of accuracy** — accuracy is the wrong metric for imbalanced datasets. AUC measures the probability that the model ranks a customer who will cancel above one who will not, regardless of the decision threshold.

**Two models** — Logistic Regression is the model you explain in 30 seconds to anyone. Random Forest captures non-linear patterns and produces feature importance. Comparing both shows the ability to choose a tool, not just run code.

### Results

```
Customers analyzed      : 7,043
Actual churn rate       : 26.5%
AUC Logistic Regression : ~0.84
AUC Random Forest       : ~0.83
Recommended model       : Logistic Regression (similar AUC, more interpretable)

Top churn factors:
  1. tenure (time as customer)
  2. Contract_Two year (annual contract reduces churn)
  3. TotalCharges (total amount paid)

Recommendations:
  - Prioritize retention of customers on monthly contracts
  - Engage customer success in the first 12 months
  - Review pricing strategy for fiber optic services
```

---

### Contributing | Contribuindo

Contributions are more than welcome. Fork the project, create a branch, commit your changes and open a pull request. Maybe you know something that I don´t know :). Contribuições são mais que bem-vindas. Faça um fork, crie uma branch, commite suas alterações e abra um pull request; talvez você saiba de algo que eu não sei.

### License | Licença

Distributed under the MIT License.
Distribuído sob a licença MIT.

---

<div align="center">
  Developed as a Data Analytics portfolio project<br>
  Desenvolvido como projeto de portfólio de Análise de Dados
</div>
EOF
Saída
