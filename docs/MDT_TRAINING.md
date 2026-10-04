# Introdução ao treinamento de MDT

**MDT — Modelo de Decisão Tipada** recebe um estado, uma pergunta e alternativas
definidas. O resultado é uma distribuição de probabilidades que permite obter
uma escolha, uma pontuação ordinal ou uma probabilidade de "sim".

Na Koemi-5HOM, o treinamento MDT usa a arquitetura Laya: encoder bidirecional,
cabeça transformer e pontuação das alternativas. Seu checkpoint é separado do
modelo causal HERM. O router MoE do HERM seleciona especialistas; o MDT responde
perguntas sobre o estado fornecido.

O caminho supervisionado usa diretamente os alvos rotulados. A perda combina
entropia cruzada com alvos probabilísticos e, para pontuações, um termo ordinal.
Isso oferece um comando de treinamento sem o termo de policy gradient. O Laya
também oferece aprendizado supervisionado em seu código-fonte; aqui o foco é
o fluxo independente, validado e integrado à Koemi.

## Preparar o ambiente e o checkpoint

Use Python 3.11 ou superior e um ambiente virtual. Na raiz do repositório:

```bash
python -m pip install -e ".[decisions]"
python -m koemi.training.laya_decisions --help
```

O extra instala `laya==0.3.27`. O treinador requer um checkpoint local completo:

```text
meu-checkpoint-laya/
  rl_agent_config.json
  model.safetensors
  encoder/config.json
  tokenizer/tokenizer_config.json
  tokenizer/...
```

A pasta do tokenizer precisa conter seus arquivos completos. Este repositório
distribui código e exemplos sintéticos; o comando não baixa pesos de modelos.

## Preparar os dados

Cada linha JSONL contém `state`, `questions` e `gold`. O estado pode ser texto,
uma lista ou um objeto. Toda pergunta deve ter uma distribuição alvo em `gold`.
Este é um exemplo formatado para leitura:

```json
{
  "state": "Please refund the duplicate invoice payment.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department handles this?",
      "criteria": {
        "billing": "payments and refunds",
        "technical": "software bugs"
      }
    }
  },
  "gold": {
    "department": {
      "probabilities": {"billing": 0.95, "technical": 0.05}
    }
  }
}
```

No arquivo de treinamento, grave cada objeto em uma única linha. Há quatro
estados sintéticos prontos em [`examples/typed_decisions.jsonl`](../examples/typed_decisions.jsonl).
Eles servem para verificar o fluxo; um modelo útil precisa de dados representativos
da tarefa. Rótulos certos podem usar probabilidades 1 e 0; alvos probabilísticos
podem expressar distribuições rotuladas ou produzidas por um professor.

| Tipo | Critérios da pergunta | Chaves das probabilidades alvo |
| --- | --- | --- |
| `choice` | Objeto com nome e descrição de cada opção | Os mesmos nomes das opções |
| `score` | Lista ordenada de descrições dos níveis | Índices como `"0"`, `"1"`, `"2"` |
| `noul` | Objeto opcional com descrições de `false` e `true` | `"false"` e `"true"` |

As probabilidades devem ser finitas, não negativas, somar 1 e nomear todas
as alternativas. O treinamento e a calibração exigem pelo menos dois estados
distintos. Estados repetidos e todas as suas perguntas permanecem na mesma
partição para evitar vazamento entre treino e calibração.

## Treinar e calibrar

```bash
python -m koemi.training.laya_decisions \
  --model-directory /caminho/meu-checkpoint-laya \
  --dataset examples/typed_decisions.jsonl \
  --output artifacts/meu-mdt \
  --epochs 4 --batch-size 8 --accumulation-steps 8 \
  --freeze-encoder --device cpu
```

No PowerShell, execute o comando em uma linha ou substitua as barras de
continuação por crases. A pasta de saída deve ser nova.

`--freeze-encoder` congela o encoder e treina os parâmetros supervisionados da
cabeça de decisões. Remova essa opção para ajustar também o encoder. Os controles
`--learning-rate` e `--encoder-learning-rate` definem taxas separadas.
`--accumulation-steps` acumula gradientes com ponderação pelo número de decisões,
inclusive quando o último grupo é menor. O caminho executa em FP32; `--device cuda`
requer um PyTorch compatível com a GPU.

Por padrão, 20% dos estados distintos são separados para ajustar temperaturas
após o treino. A temperatura é ajustada por tipo, dentro dos limites do Laya
instalado. O relatório inclui perda por época, passos do otimizador, número de
estados e NLL de calibração antes e depois. Essa calibração não substitui uma
avaliação em um terceiro conjunto independente.

O encoder recebe um orçamento de tokens. Truncamento do estado ou colapso das
alternativas é rejeitado; ajuste `--max-len` e `--head-max-len` quando necessário.
`--allow-state-truncation` permite explicitamente truncar o estado.

## Carregar e prever

O diretório exportado preserva os pesos safetensors, configuração do encoder,
tokenizer e configuração Laya, além de `koemi_training_report.json`.

```python
import json
from pathlib import Path

import laya

agent = laya.load("artifacts/meu-mdt", device="cpu")
row = json.loads(Path("examples/typed_decisions.jsonl").read_text(encoding="utf-8").splitlines()[0])
print(agent.predict(row["state"], row["questions"]))
```

Esse exemplo verifica carregamento e previsão. Para medir qualidade, use estados
novos e rotulados. O loader padrão acima foi verificado; a seleção explícita
de backend depende de um módulo ausente no wheel Laya 0.3.27.

A cabeça de ação/escalonamento do Laya é preservada e permanece sem nova
supervisão. As probabilidades das alternativas não garantem que a resposta
esteja correta. Ganhos de qualidade, custo e velocidade precisam ser medidos
na tarefa e no hardware escolhidos.

## Relação com o DeepGEMM

A inspiração no [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) está no caminho
MoE do HERM: separar seleção e execução, agrupando linhas por especialista.
A implementação nova usa PyTorch; não integra os kernels CUDA DeepGEMM.
O treinamento MDT é uma frente separada. Os contratos e a verificação local
estão em [`KOEMI_5HOM.md`](KOEMI_5HOM.md).
