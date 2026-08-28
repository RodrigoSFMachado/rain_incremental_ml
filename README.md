# 🌧️ Rain Prediction — Machine Learning em produção com aprendizado incremental

Serviço de previsão de chuva que serve predições via API, recebe dados rotulados
novos e **atualiza os pesos do modelo sem recriá-lo**.

O projeto nasceu de um experimento acadêmico sobre aprendizado incremental em
dados sujeitos a mudanças temporais e foi transformado em um sistema pequeno,
funcional e reproduzível: PyTorch para o modelo, FastAPI para servir, Docker para
empacotar, MLflow para tracking e SQLite para monitoramento.

```
Dados → Modelo → API → Predições → Novos rótulos → Incremental → Nova versão
```

---

## Resultado principal

| | F1 | Precisão | Recall | AUC |
|---|---|---|---|---|
| Baseline de persistência | 0.510 | 0.511 | 0.510 | 0.743 |
| **MLP (PyTorch)** | **0.535** | 0.545 | 0.526 | **0.893** |

Teste em 2024–2025 (17.443 horas), com o limiar calibrado em 2023.

E o resultado que **contraria a premissa do projeto**: em um experimento
controlado com 117 janelas temporais, **atualizar o modelo não melhorou o
desempenho**. Os detalhes estão em [Experimento](#experimento-vale-a-pena-atualizar-o-modelo),
incluindo a investigação da causa.

---

## O problema

Prever se **vai chover na próxima hora** no aeroporto de Miami (estação ASOS
`MIA`), a partir de observações meteorológicas de superfície.

**Por que isso motiva aprendizado incremental?** Dados climáticos mudam de
comportamento ao longo do tempo — sazonalidade, variações interanuais, mudanças
de regime. Um modelo treinado uma vez e nunca mais tocado pode ir perdendo
qualidade conforme a relação entre as variáveis muda. Neste dataset a taxa de
chuva caiu de ~5,5% (2022–2024) para 3,8% em 2025.

Esse fenômeno é conhecido como *concept drift*. **O projeto não implementa
detecção de drift** — nem ADWIN, nem DDM, nem PSI. O drift aparece apenas como
motivação para a pergunta prática: *vale a pena manter um modelo aprendendo em
produção?*

---

## Arquitetura

```
OFFLINE (executado na máquina de desenvolvimento)
──────────────────────────────────────────────────────────────
  CSV ASOS ──► prepare_data.py ──► train_initial.py
                                        │
                                        ├──► models/model.pt
                                        │    (pesos + estado do Adam +
                                        │     scaler + limiar + versão)
                                        └──► mlflow.db  (params, métricas)

  experiment.py ──► reports/  (results.csv + gráficos)


ONLINE (o serviço)
──────────────────────────────────────────────────────────────
                     ┌──────────────────┐
        cliente ────►│  FastAPI         │
                     │                  │
                     │  POST /predict   │
                     │  POST /update    │
                     │  GET  /metrics   │
                     │  GET  /health    │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  RainModel       │
                     │  (PyTorch MLP)   │
                     │  predict()       │
                     │  incremental_fit()│
                     └────────┬─────────┘
                              │
                ┌─────────────┴────────────┐
                ▼                          ▼
        models/model.pt            data/monitoring.db
        models/registry.json            (SQLite)
                                            │
                                   ┌────────▼────────┐
                                   │    Streamlit    │
                                   │    dashboard    │
                                   └─────────────────┘
```

Dois contêineres: `api` e `dashboard`. O dashboard lê o mesmo SQLite que a API
escreve, montado como somente leitura — se ele cair, o serviço não sente.

---

## Dados e features

**Fonte:** [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml) —
observações ASOS/METAR da estação MIA.

| | |
|---|---|
| Período usado | 2021–2025 |
| Observações | 43.624 (horárias) |
| Features | 16 |
| Taxa de positivos | 5,07% (1 chuva a cada ~19 horas) |

### Decisões de preparação

**Somente METAR de rotina (`minuto == 53`).** O ASOS emite relatórios de rotina
horários e relatórios especiais (SPECI), disparados quando o tempo muda —
inclusive quando começa a chover. Manter os SPECI faria a *quantidade* de
observações carregar informação sobre o alvo. Depois do filtro, 99,9% dos
intervalos são exatamente 1 hora, e a pressão (`mslp`) sai de 18k valores
ausentes para 71.

**Alvo deslocado em 1 hora.** `rain_next_hour = p01i[t+1] >= 0.01`. Sem o
deslocamento, as features e o alvo seriam do mesmo instante e o modelo estaria
diagnosticando o presente, não prevendo o futuro. O alvo só é considerado válido
se a observação seguinte for de fato 1 hora depois.

**Vento decomposto em componentes.** Direção é circular: 359° e 1° são vizinhos,
mas numericamente distantes. `wind_u = -sknt·sin(θ)`, `wind_v = -sknt·cos(θ)`.
Os 4,6% de registros com direção variável mantêm a velocidade e recebem `u=v=0`,
em vez de serem descartados (o que furaria os lags).

**Tendências, não só níveis.** `mslp_delta_3h` (queda de pressão antecede chuva),
`relh_delta_1h`, `dew_spread` e ciclos diurno/sazonal via `sin`/`cos`.

<details>
<summary>Lista completa das 16 features</summary>

`tmpf`, `dwpf`, `relh`, `mslp`, `vsby`, `wind_speed`, `wind_u`, `wind_v`,
`dew_spread`, `mslp_delta_3h`, `relh_delta_1h`, `rain_now`, `hour_sin`,
`hour_cos`, `month_sin`, `month_cos`

Testei 4 features adicionais (`rain_last_3h`, `vsby_delta_1h`, `relh_delta_3h`,
`vsby_min_3h`) e o F1 caiu de 0,526 para 0,517. Ficaram de fora.
</details>

---

## Modelo

```
Entrada (16) → Linear(16) → ReLU → Linear(1) → logit
```

289 parâmetros. Treino completo em **4 segundos**, atualização incremental em
**10 milissegundos**, checkpoint de **9,5 KB**.

| Decisão | Escolha | Motivo |
|---|---|---|
| Saída | Logit (sigmoid só na inferência) | Permite `BCEWithLogitsLoss`, numericamente estável |
| Loss | `BCEWithLogitsLoss(pos_weight=18)` | Trata o desbalanceamento sem reamostrar |
| Otimizador | Adam | Estado salvo junto com os pesos |
| LR inicial / update | 1e-3 / 3e-5 | O valor de update saiu de uma [ablação](#a-investigação) |
| Épocas inicial / update | 30 / 2 | Poucas épocas no update evitam overfitting no lote recente |
| Limiar | 0,84, calibrado em validação | Nunca no conjunto de teste |

### Como o aprendizado incremental funciona aqui

Esta é a parte central do projeto. A diferença entre continuar treinando e
recomeçar do zero:

```python
# ❌ Isto parece incremental, mas é retreino
model = RainModel()            # __init__ reinicializa os pesos
optimizer = Adam(...)          # estado do otimizador zerado
model.fit(novos_dados)

# ✅ Isto é incremental
model = RainModel.load("model.pt")   # pesos + estado do Adam restaurados
model.incremental_fit(novos_dados)   # continua de onde parou
model.save("model.pt")               # versão incrementada
```

`fit` e `incremental_fit` chamam **o mesmo laço de treino**. Aprendizado
incremental não é um algoritmo diferente — é o mesmo passo de gradiente aplicado
sobre um estado preservado, com passo menor para não sobrescrever o que já
existia.

**Três coisas precisam sobreviver entre uma sessão e outra:**

1. **Os pesos** — óbvio, mas é o que muita implementação erra ao recriar a rede.
2. **O estado do otimizador** — o Adam guarda médias móveis dos gradientes
   (`exp_avg`, `exp_avg_sq`). Recarregar só os pesos e criar um Adam novo zera
   essas médias e torna os primeiros passos erráticos. É um bug silencioso: o
   modelo piora logo depois de ser carregado, sem motivo aparente.
3. **O scaler** — calculado uma vez e **congelado**. Os pesos foram aprendidos
   num espaço de features com determinada média e escala; se as estatísticas de
   normalização mudarem, esse espaço muda e os pesos deixam de valer. Esta é a
   armadilha nº 1 de incremental learning com rede neural.

Tudo isso vive em um único `.pt` autocontido, junto com `feature_names` (que
elimina o bug clássico de ordem de colunas divergente entre treino e serving),
o limiar, a versão e o histórico.

**O teste que prova a continuidade:**

```python
# tests/test_model.py::test_incremental_continues_from_saved_weights
```

Verifica que (1) os pesos recarregados são idênticos aos salvos via
`torch.equal`, (2) após um update eles mudam pouco, e (3) um modelo recriado do
zero fica muito mais distante do estado anterior do que a atualização
incremental.

---

## Experimento: vale a pena atualizar o modelo?

Três políticas de atualização, avaliadas sobre as mesmas 117 janelas
(180 dias de treino, 14 de teste, passo de 14 dias):

- **Static** — treina uma vez na primeira janela e nunca mais muda
- **Retrain** — a cada passo, cria um modelo novo com os 180 dias mais recentes
- **Incremental** — treina uma vez e continua o treinamento com cada lote novo

Protocolo prequencial: o modelo **prevê primeiro e aprende depois**. Nenhuma
janela usa dados do futuro.

### Resultado

| Cenário | F1 | Precisão | Recall | Acurácia |
|---|---|---|---|---|
| Persistência | 0.495 | 0.494 | 0.496 | 0.944 |
| **Static** | **0.499** | 0.486 | 0.513 | 0.943 |
| Incremental | 0.497 | 0.470 | 0.527 | 0.941 |
| Retrain | 0.495 | 0.532 | 0.463 | 0.947 |

*(Executado com `LR_UPDATE = 3e-5`, o valor escolhido a partir da ablação
abaixo. Com valores maiores o incremental piora — é exatamente o que a
ablação mostra.)*

![F1 por ano](reports/f1_por_ano.png)

**Atualizar o modelo não ajudou.** Os três cenários ficam dentro de 0,005 de F1
uns dos outros — e da baseline trivial. A diferença é menor que a oscilação
entre anos, ou seja, não é distinguível de ruído.

### A investigação

Antes de aceitar o resultado, variei a intensidade da atualização:

| Configuração | F1 | Precisão | Recall |
|---|---|---|---|
| **sem update (= static)** | **0.499** | 0.486 | 0.513 |
| 2 épocas, lr 1e-5 | 0.499 | 0.480 | 0.519 |
| 2 épocas, lr 3e-5 | 0.497 | 0.470 | 0.527 |
| 2 épocas, lr 1e-4 | 0.488 | 0.433 | 0.559 |
| 2 épocas, lr 3e-4 | 0.462 | 0.379 | 0.593 |
| 5 épocas, lr 1e-4 | 0.470 | 0.395 | 0.580 |
| 10 épocas, lr 1e-4 | 0.467 | 0.385 | 0.591 |

O padrão é monotônico: **quanto mais o modelo treina nos dados recentes, pior o
F1** — a precisão despenca e o recall sobe. Replay buffer (500 e 2.000 amostras)
piorou mais ainda.

**Causa:** com `pos_weight = 18`, cada passo de gradiente adicional empurra o
modelo a prever positivo com mais frequência, e o limiar fixo vai ficando
desalinhado. Não é o modelo aprendendo drift — é a calibração escorregando.

**Conclusão:** o mecanismo incremental funciona como projetado; é o *problema*
que não recompensa a atualização. O modelo em produção é o estático, e o
`/update` fica disponível e testado — o clima de Miami pode não ser estável para
sempre.

### Duas correções metodológicas em relação ao experimento original

**O Retrain treina com o mesmo esforço do treino inicial.** No experimento
acadêmico original ele fazia uma única passada de SGD enquanto o incremental
acumulava milhares de passos de gradiente. A vantagem aparente do incremental
podia ser apenas *número de atualizações*. Igualando o esforço, a diferença
desaparece.

**O Incremental não recalibra o limiar.** Ele usa exatamente o mesmo limiar do
Static, de modo que a única diferença entre os dois é a atualização dos pesos —
a variável que o experimento se propõe a medir. Com recalibração, o F1 caía para
0,477 e não seria possível saber qual dos dois efeitos explicava a diferença.

---

## API

Documentação interativa em `http://localhost:8000/docs`.

| Endpoint | Descrição |
|---|---|
| `GET /health` | Verificação de vida |
| `GET /model` | Versão, nº de atualizações, limiar, nº de parâmetros |
| `POST /predict` | Previsão para uma observação |
| `POST /update` | Atualização incremental com lote rotulado |
| `GET /metrics` | Métricas operacionais |
| `GET /versions` | Histórico de versões |

### Exemplo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tmpf": 82.0, "dwpf": 75.0, "relh": 79.5, "mslp": 1012.5,
    "vsby": 10.0, "wind_speed": 8.0, "wind_u": -5.6, "wind_v": -5.6,
    "dew_spread": 7.0, "mslp_delta_3h": -1.2, "relh_delta_1h": 3.5,
    "rain_now": 0.0, "hour_sin": 0.5, "hour_cos": -0.866,
    "month_sin": -0.5, "month_cos": -0.866
  }'
```

```json
{
  "prediction": 0,
  "probability": 0.7935,
  "threshold": 0.84,
  "model_version": 1,
  "prediction_id": 1
}
```

### `/update` — a ordem das operações importa

```
1. o modelo prevê sobre o lote  →  matriz de confusão registrada
2. só depois ele aprende com esses dados
3. checkpoint salvo com a versão incrementada
```

**Avaliar antes de aprender** é o que torna a métrica honesta: medir depois
mostraria o modelo acertando dados que acabou de estudar. É o mesmo protocolo
prequencial do experimento offline, agora em produção.

O endpoint aceita de 1 a 5.000 observações. Lote é preferível a uma observação
por vez: com 5% de positivos, um lote de tamanho 1 quase nunca contém chuva e o
gradiente resultante é ruído. Também é realista — o rótulo só existe uma hora
depois, então na prática se acumula.

---

## Monitoramento

Toda predição e toda atualização são gravadas em SQLite. O dashboard Streamlit
lê o mesmo arquivo.

**O que é monitorado:** volume de predições, distribuição das probabilidades,
taxa de positivos, número de atualizações, versão em uso e desempenho real
(quando lotes rotulados chegam via `/update`).

A **distribuição das probabilidades** é o sinal mais útil: um deslocamento nela
indica mudança nos dados de entrada ou no modelo *sem precisar esperar pelos
rótulos verdadeiros*. É o raciocínio por trás da detecção de drift, sem a
complexidade de implementar um detector.

### Por que não Prometheus + Grafana

Prometheus resolve agregação entre múltiplas instâncias, retenção de séries
temporais e alertas. Este projeto tem **uma instância, um worker e nenhum
alerta**. SQLite responde às mesmas perguntas com uma dependência que já vem no
Python, e é lido diretamente pelo dashboard. Quando houver múltiplas instâncias,
a migração é direta — as métricas já estão estruturadas.

### Simulação de produção

```bash
python -m training.simulate_production --start 2025-06-01 --days 120 --in-process
```

Percorre o dataset hora a hora chamando `/predict`, e a cada 14 dias envia o
lote rotulado para `/update`, reproduzindo o atraso real dos rótulos:

```
Simulando 2,852 horas (2025-06-01 -> 2025-09-28)

  2025-06-15  update com 321 obs  v1 -> v2  F1 no lote: 0.578
  2025-06-29  update com 336 obs  v2 -> v3  F1 no lote: 0.129
  2025-07-13  update com 336 obs  v3 -> v4  F1 no lote: 0.545
  ...
  2025-09-21  update com 336 obs  v8 -> v9  F1 no lote: 0.571

2,852 predições servidas  |  versão final: 9
desempenho acumulado: F1=0.506  precisão=0.529  recall=0.485
```

O F1 de 0,129 na segunda quinzena de junho não é um bug: junho marca o início da
estação chuvosa em Miami e a distribuição vira. É evidência de que métrica sobre
lote pequeno oscila muito — por isso o dashboard mostra o acumulado *e* o valor
por lote.

---

## MLflow

Usado para **tracking de experimentos**: parâmetros, métricas e artefatos de cada
run, com backend SQLite local.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

> O file store (`./mlruns`) entrou em modo de manutenção no MLflow 3.x. A maioria
> dos tutoriais ainda usa esse caminho; este projeto usa SQLite, que continua
> sendo um arquivo local, sem servidor.

**O Model Registry do MLflow não é usado**, por decisão de escopo. Ele exigiria
um servidor rodando e faria a API depender dele em tempo de execução — se o
MLflow cai, a API cai junto. Aqui o modelo em produção é um arquivo `.pt`, e as
perguntas relevantes (qual versão está no ar, quando foi criada, com que
desempenho) são respondidas por `models/registry.json`, legível por humanos.

Consequência prática: **MLflow não está no `requirements.txt` da API**, apenas no
`requirements-train.txt`. A imagem do serviço não carrega o que não usa.

---

## Como executar

### Com Docker (recomendado)

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

### Localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
```

```bash
# 1. preparar os dados (o CSV vai em data/)
python -m training.prepare_data --input data/MIA_2012_2025.csv --start-year 2021

# 2. treinar o modelo inicial
python -m training.train_initial

# 3. (opcional) rodar o experimento comparativo
python -m training.experiment

# 4. subir a API
uvicorn app.main:app --reload

# 5. gerar tráfego e ver o dashboard
python -m training.simulate_production --days 120 --in-process
streamlit run dashboard/app.py
```

### Testes

```bash
pytest tests/ -v      # 16 testes
```

---

## Estrutura

```
.
├── app/                      # o serviço
│   ├── constants.py          # contrato das features (sem pandas)
│   ├── features.py           # limpeza e engenharia de features
│   ├── main.py               # FastAPI
│   ├── metrics.py            # métricas de classificação
│   ├── model.py              # RainModel: fit, incremental_fit, save, load
│   ├── registry.py           # versionamento de modelo
│   ├── schemas.py            # validação Pydantic
│   └── storage.py            # SQLite de monitoramento
├── training/                 # offline
│   ├── prepare_data.py
│   ├── train_initial.py
│   ├── experiment.py         # static × retrain × incremental
│   └── simulate_production.py
├── dashboard/app.py          # Streamlit
├── tests/                    # 16 testes
├── models/                   # model.pt + registry.json
├── reports/                  # resultados e gráficos
├── notebooks/                # experimento acadêmico original
├── Dockerfile                # API (torch CPU-only)
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt          # dependências da API
└── requirements-train.txt    # + treino e experimentação
```

---

## Decisões técnicas

| Decisão | Alternativa descartada | Motivo |
|---|---|---|
| SQLite para monitoramento | Prometheus + Grafana | Uma instância, nenhum alerta. Dois contêineres a mais não se pagariam |
| `registry.json` | MLflow Model Registry | Evita dependência de runtime da API em um servidor |
| MLflow só offline | MLflow na imagem da API | Reduz a imagem; MLflow não é usado em runtime |
| `constants.py` separado | Importar de `features.py` | Tira pandas da API: ~50 MB de RAM a menos |
| torch CPU-only | `pip install torch` | A versão do PyPI passa de 2 GB por causa das libs CUDA |
| `--workers 1` | Múltiplos workers | O modelo é alterado in-place; workers teriam versões divergentes |
| Scaler congelado | Normalização adaptativa | Mudar a escala invalidaria os pesos aprendidos |
| Limiar calibrado em validação | Limiar fixo em 0,5 | Com 5% de positivos, 0,5 não é um corte razoável |

---

## Limitações

**As probabilidades não são calibradas em termos absolutos.** O `pos_weight`
desloca a escala: "0,84" não significa 84% de chance de chover. O modelo ordena
o risco muito bem (AUC 0,893), mas para leitura direta da probabilidade seria
necessária uma calibração posterior (Platt scaling ou isotônica).

**Uma única estação.** Miami tem um regime tropical bastante específico. Os
resultados não se transferem automaticamente para outros climas — e o resultado
negativo do incremental provavelmente também não.

**O ganho sobre a baseline trivial é modesto em F1** (0,535 vs 0,510). A
diferença real está no AUC (0,893 vs 0,743): a persistência só responde sim ou
não, o modelo entrega uma probabilidade que ordena o risco.

**Disco efêmero em deploy gratuito.** Em plataformas free tier o sistema de
arquivos é volátil: o modelo atualizado via `/update` volta ao estado do commit
a cada restart. Localmente os volumes do Docker resolvem; em produção real seria
necessário um volume persistente ou object storage.

**Um único worker.** Escalar exigiria separar leitura de escrita: vários workers
servindo `/predict` a partir de um checkpoint compartilhado e um processo único
responsável pelas atualizações.

**Métricas por lote pequeno oscilam muito.** Duas semanas sem chuva produzem F1
igual a zero por ausência de positivos, não por falha do modelo.

---

## Próximos passos

- Calibração de probabilidade (Platt/isotônica) para tornar a saída interpretável
- Múltiplas estações, para testar se o resultado do incremental se sustenta
- Separação leitura/escrita para permitir múltiplos workers
- CI no GitHub Actions rodando os testes a cada push
- Agendamento automático do `/update` conforme os rótulos ficam disponíveis

---

## Origem do projeto

Este trabalho partiu de um experimento acadêmico que comparava aprendizado
incremental, retreinamento periódico e treinamento único usando
River + Regressão Logística, em dados sujeitos a *concept drift*.

O que mudou na transformação em sistema de produção:

| | Experimento original | Este projeto |
|---|---|---|
| Modelo | River + Logistic Regression | MLP em PyTorch |
| Alvo | Chuva na hora corrente | Chuva na próxima hora |
| Features | 5 | 16 |
| Baseline | — | Persistência |
| Retrain | 1 época de SGD | Mesmo esforço do treino inicial |
| Entrega | Notebook | API + Docker + monitoramento |

O notebook original está em `notebooks/`.
