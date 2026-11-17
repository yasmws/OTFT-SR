# OTFT-SR

Experimentos de regressão simbólica para curvas de transferência p-OTFT usando o pipeline LLM-SR. O repositório reúne adaptadores de dados, especificações, notebooks, resultados, logs e relatórios de auditoria para uma curva experimental da Org1 e conjuntos sintéticos das Org1–Org4.

## Escopo e interpretação

- As entradas são tensões em módulo (`|VGS|` e, nos conjuntos sintéticos, `|VDS|`); o alvo é `log10(|ID|/A)`.
- A curva experimental Org1 foi medida em uma única condição de `|VDS|` próxima de 50 V. Ela não representa um mapa bidimensional completo nem permite estimar separadamente parâmetros físicos que não constam nos dados.
- Os conjuntos sintéticos são gerados por um simulador OTFT. Seus resultados avaliam aproximação dos dados do gerador e não demonstram descoberta de uma lei física independente nem validação em dispositivos novos.
- Nos experimentos sintéticos, a divisão é feita por curva completa. `test_id` e `test_ood` são conjuntos reservados para avaliação; os runners usam apenas `train` na busca.

Consulte [`reports/AUDITORIA_CIENTIFICA_PIPELINE_OTFT.md`](reports/AUDITORIA_CIENTIFICA_PIPELINE_OTFT.md) para limitações, resultados e recomendações de interpretação.

## Conteúdo

- `prepare_dataset.py` e `run_experiment.py`: adaptação e execução da curva experimental.
- `prepare_synthetic_*.py` e `run_synthetic_*.py`: preparação e execução para Org1–Org4.
- `data/`: tabelas de treino e avaliação e manifestos das partições.
- `specifications/`: especificações de busca compatíveis com LLM-SR.
- `notebooks/`: notebooks de experimento e análise.
- `results/`, `logs/` e `audit/`: resultados persistidos, logs e verificações de auditoria.
- `reports/` e `relatorios/`: relatórios científicos, matrizes de achados e plano de correção.

## Dependências externas

Este repositório contém integralmente os arquivos da pasta `OTFT-SR`, mas alguns scripts também leem fontes que ficam fora dela no workspace original:

- o pipeline-base `LLM-SR/`;
- os dados de origem e metadados em `Model_OTFT/`.

Os adaptadores e runners calculam esses caminhos em relação ao diretório-pai do repositório. Para reproduzir a preparação dos dados originais e a busca completa, coloque os diretórios `LLM-SR`, `Model_OTFT` e `OTFT-SR` lado a lado. Os CSVs preparados já versionados em `data/` podem ser usados sem repetir a etapa de preparação. O modo `--smoke` verifica o encadeamento local sem servidor LLM.

A busca real requer o ambiente e as dependências do projeto LLM-SR, além de um servidor local de completions acessível em `http://127.0.0.1:5000/completions`. Os métodos clássicos opcionais estão listados em [`requirements-classic.txt`](requirements-classic.txt); PySR também requer runtime Julia.

## Uso

A partir da pasta do workspace que contém os três diretórios:

```bash
python OTFT-SR/prepare_dataset.py
python OTFT-SR/run_experiment.py --max-samples 100

python OTFT-SR/prepare_synthetic_dataset.py
python OTFT-SR/run_synthetic_experiment.py --max-samples 100
python OTFT-SR/prepare_synthetic_org2.py
python OTFT-SR/run_synthetic_org2.py --max-samples 100
python OTFT-SR/prepare_synthetic_org3.py
python OTFT-SR/run_synthetic_org3.py --max-samples 100
python OTFT-SR/prepare_synthetic_org4.py
python OTFT-SR/run_synthetic_org4.py --max-samples 100
```

Para executar a verificação determinística sem servidor LLM, use `--smoke --max-samples 3` nos runners. O modo smoke não mede descoberta via LLM. Os notebooks correspondentes estão em `notebooks/`.

## Licença e dados

Não foi incluído um arquivo de licença. Até que uma licença seja definida pelos titulares dos dados e do código, todos os direitos permanecem reservados. Verifique permissões e atribuição exigidas pelas fontes externas antes de reutilizar ou redistribuir esses materiais.
