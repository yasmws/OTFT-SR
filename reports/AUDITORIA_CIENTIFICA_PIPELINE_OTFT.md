# Auditoria científica do pipeline de descoberta de equações para OTFTs

**Data:** 16/11/2026  
**Escopo:** `OTFT-SR`, os módulos relacionados de `LLM-SR` e `Model_OTFT`, os quatro corpora sintéticos Org1–Org4, o conjunto experimental medido Org1, notebooks, logs e resultados persistidos.  
**Parecer:** **Não validado** para sustentar uma conclusão científica geral de descoberta física ou superioridade do método. Os resultados são evidência exploratória de ajuste de curvas sintéticas e de uma curva experimental específica, com limitações materiais descritas abaixo.

## A. Resumo executivo

O pipeline investigado adapta o LLM-SR para ajustar `log10(|ID|/A)` a partir de `|VGS|` e `|VDS|` (dados sintéticos) ou de `|VGS|` em uma curva de transferência experimental Org1 medida perto de `|VDS|=50 V`. A função de perda minimizada é MSE na escala logarítmica. Coeficientes numéricos são ajustados pelo próprio avaliador; a busca LLM altera a estrutura da equação. Para os dados sintéticos, a divisão é por curva completa e o runner passa apenas `train` à busca.

Não encontrei evidência de que `test_id`/`test_ood` tenham entrado no `pipeline.main` nem de sobreposição exata de linhas, curvas ou pares `(VGS,VDS)` entre as partições sintéticas inspecionadas. Portanto, **não identifiquei vazamento direto de teste no caminho atual dos runners**. Isso não demonstra ausência de vazamento contextual nas execuções históricas: os prompts e os identificadores/parâmetros do servidor LLM não foram arquivados.

Os artefatos de previsões persistidos permitem recalcular e reproduzir as métricas de cinco holdouts. O ajuste de uma única curva experimental tem RMSE 0,128 década, mas R² 0,103 em apenas 11 pontos contíguos de alta tensão. Nos resultados sintéticos, o OOD da Org4 tem RMSE 1,691 décadas e R² −0,073; na Org3, o PySR tem RMSE médio por curva OOD 0,587, abaixo do LLM-SR (0,843). Org1 e Org2 sintéticas apresentam erros menores, mas continuam sendo dados gerados por simulador e não uma validação externa de dispositivos.

Há também uma limitação de desenho: os quatro conjuntos sintéticos são gerados pelo modelo OTFT já presente em `Model_OTFT` e seus índices guardam os parâmetros de geração. Assim, os resultados avaliam aproximação de um gerador conhecido; não demonstram descoberta de uma lei física independente nem generalização a novos dispositivos. A comparação LLM-SR versus métodos clássicos foi feita com treino e avaliação correspondentes nos notebooks, mas orçamentos de busca, operadores, parametrização e tratamento de coeficientes não são pareados; deve ser tratada como exploratória, não como prova isolada de contribuição causal da LLM.

## B. Reconstrução metodológica

### Hipótese operacional reconstruída

O objetivo efetivamente implementado é encontrar uma função empírica compacta que aproxime a corrente de dreno em escala logarítmica, com coeficientes ajustáveis. No caso sintético, uma função bivariada `f(|VGS|, |VDS|) → log10(|ID|/A)` deve acomodar curvas de transferência e de saída. Na curva experimental, a tarefa é univariada e restrita a uma curva p-OTFT da Org1 com tensão de dreno nominal fixa.

O pipeline não implementa uma hipótese de extração de mobilidade, tensão de limiar ou outros parâmetros intrínsecos a partir da equação encontrada. A especificação da curva experimental declara que a equação é empírica e não deve ser interpretada como extração independente de parâmetros físicos (`OTFT-SR/specifications/otft_transfer_numpy.txt`). As especificações sintéticas também descrevem regressão MSE, coeficientes ajustáveis e o template inicial linear (`OTFT-SR/specifications/otft_synthetic*_numpy.txt`).

### Fluxo executado

1. Os adaptadores leem curvas CSV, tomam módulos de tensões/corrente, convertem corrente sintética de sua unidade declarada nos NPZ para ampères e calculam `log10(ID_A)`.
2. O adaptador sintético seleciona 150 posições aproximadamente uniformes por curva; a unidade de partição é o arquivo/curva completa. Os manifests registram arquivos, partições, unidades e parâmetros simulados.
3. O runner carrega `train.csv` e forma o dicionário de dados do LLM-SR. As especificações fazem `scipy.optimize.minimize(..., method="BFGS")` minimizar MSE e devolvem `-MSE` como score. A fórmula começa linear e até 12 parâmetros estão disponíveis nas especificações sintéticas.
4. O mecanismo ExperienceBuffer usa o score de treino para priorizar candidatos e compor prompts. A equação selecionada no notebook é a de maior score logado; seu parâmetro é reajustado em treino antes da avaliação reservada.
5. Os notebooks de análise ajustam PySR, gplearn e DEAP apenas com treino e avaliam os métodos nos conjuntos carregados. Exportam previsões da candidata LLM-SR para `test_ood`, métricas por curva, especificações e equações.

### O que os resultados permitem afirmar

Há evidência de que o método produz equações executáveis e aproxima algumas curvas sintéticas reservadas. Não há evidência de replicação de execuções históricas independentes, pois seed, amostra de LLM, ambiente exato e prompt enviado não estão ligados de modo completo a cada saída. Também não há demonstração de uma lei OTFT nova: os dados sintéticos provêm do modelo próprio de `Model_OTFT`, e a tarefa experimental cobre uma curva, não o mapa bidimensional completo.

## C. Auditoria dos dados e vazamento

### Proveniência e integridade

- A fonte medida usada na variante real é `Model_OTFT/datas/Org1/tipo p/org1_50VDS.csv`. Seu cabeçalho é `VDS,VGS,ID`; a linha seguinte declara `V,V,A`. A rotina `prepare_dataset.py` ignora essa linha, mantém apenas pontos finitos e positivos, exige módulo de VDS próximo de 50 V e transforma `ID` em `log10(ID)` (linhas 12–60). A conversão A é compatível com a unidade explícita no arquivo.
- As curvas sintéticas estão em `Model_OTFT/synthetic_inference_runs/Org{1,2,3,4}`. `_generate_synthetic_data.py` instancia o modelo de `Model_OTFT` e seleciona parâmetros por curva; os índices `index_generated.json` registram `params`, tensão fixa e caminho da curva. O adaptador lê `current_typic` dos NPZ e documenta a conversão (por exemplo, `prepare_synthetic_dataset.py`, linhas 58–72 e 107–111).
- Os manifests declaram 150 pontos por curva. O teste independente confirmou que as contagens e grupos observados correspondem aos manifests e que as colunas numéricas sintéticas são finitas. Não foi refeito o gerador contra cada arquivo completo porque os artifacts são amostrados, o estado aleatório/configuração de geração completa não está vinculado de forma rastreável ao resultado LLM e o propósito aqui foi preservar outputs originais.

### Partições e risco de vazamento

O código reserva curvas inteiras, com partições declaradas em `curve_split_manifest.json`; os runners usam o CSV de treino para montar a entrada ao LLM-SR. O teste independente conferiu, para Org1–Org4: 0 linhas duplicadas, 0 pares exatos `(VGS_abs_V,VDS_abs_V)` duplicados internamente e 0 pares compartilhados entre `train`, `test_id` e `test_ood`; nenhum `curve_id` se repete entre conjuntos.

| Tipo de risco | Diagnóstico | Evidência e limite |
|---|---|---|
| Mesmo ponto/curva em treino e avaliação sintética | Nenhum vazamento identificado nos testes realizados | `audit/tests/test_data_integrity.py`, `test_data_leakage.py`; 10 testes passaram. Verificação de igualdade exata nos CSVs persistidos. |
| Pré-processamento ajustado em todo o conjunto | Nenhum vazamento identificado nos caminhos inspecionados | Adaptadores aplicam `abs`, conversão de unidade e log por observação; não há scaler ou imputador ajustado globalmente. |
| Dados de avaliação no runner | Nenhum vazamento identificado no caminho do código | `run_synthetic_org{1,2,3,4}.py` constrói `dataset` com `train` apenas. |
| Feedback do teste à LLM/seleção | Nenhum vazamento identificado no caminho de código auditado; histórico não totalmente verificável | `LLM-SR/llmsr/pipeline.py` recebe somente `inputs`; teste não consta na entrada. Prompts de servidor e sessões históricas não foram salvos. |
| Correlação física entre pontos | Risco potencial presente, mas delimitado pela unidade curva | Pontos dentro de cada curva são correlacionados; por isso não devem ser tratados como réplicas independentes para intervalos/testes estatísticos. Curvas sintéticas também compartilham o gerador e podem compartilhar parâmetros, verificação de independência entre curvas ausente. |
| Resultado de seleção sobre treino | Risco de otimismo de treino, não vazamento de teste | Candidata “melhor” é escolhida pelo score de treino. Há avaliação reservada separada, mas não foram retidos todos os resultados por repetição/curva em todos os formatos. |
| Curva experimental Org1 | Split contíguo de pontos, dependentes na varredura | `prepare_dataset.py`, linhas 52–60: ordena por VGS e usa 80% iniciais/20% finais. Isso é apropriado para extrapolação ao longo dessa curva, mas não para estimar generalização a novos dispositivos/curvas; as 11 observações de teste são uma só curva correlacionada. |

O protocolo sintético testa polarizações fixas omitidas, mas o rótulo OOD é relativo à faixa de polarização fixa por família, não prova distribuição independente de dispositivos. Org3 possui curva de tensão nominal repetida com sufixos distintos e maior número de curvas de teste do que treino; o código separa arquivos, o que evita sobreposição de curva, mas a inferência estatística por ponto continua inadequada.

## D. Auditoria física e matemática

As variáveis confirmadas nos adaptadores são `VGS`, `VDS` e `ID`; os CSVs brutos sintéticos armazenam valores de sinal p-canal, o adaptador toma módulos e identifica as entradas como volts e alvo como `log10(|ID|/A)`. Isto é coerente como representação de magnitude, porém descarta a convenção de sinal e não permite conclusões sobre comportamento de polaridade, corrente assinada ou assimetria entre polarizações.

Nos dados experimentais Org1, as unidades no cabeçalho são volts e ampères; nos sintéticos, a corrente varia conforme `current_typic` no NPZ, que é convertida explicitamente. O teste independente conferiu a consistência das colunas A exportadas com a escala logarítmica. Não há normalizador a inverter. Não encontrei corrente nula nos pontos usados, pois a preparação rejeita esse caso.

As equações salvas foram verificadas quanto à sintaxe Python, nomes de entradas esperados e parâmetros numéricos. Isso não equivale a auditoria dimensional completa: os coeficientes ajustáveis são livres e podem carregar unidades implícitas, e a saída é logarítmica, portanto as expressões são regressões empíricas em coordenadas transformadas. As equações polinomiais com termos como `VGS²` e `VDS³` não têm interpretação dimensional declarada nem restrições de sinal/monotonicidade. Há expressões de grau alto nas soluções selecionadas, sem análise de estabilidade em um domínio de tensão mais amplo.

O modelo teórico OTFT na documentação do `Model_OTFT` inclui regiões de acumulação, cargas, fator de saturação, resistência e outros efeitos, mas o LLM-SR recebe apenas uma descrição curta e busca polinômios/operadores no template. Não encontrei validação por restrições físicas específicas do dispositivo (limites, derivadas, saturação, regime subthreshold, continuidade ou plausibilidade de parâmetros). As figuras e métricas avaliam ajuste em curvas amostradas; extrapolação do domínio completo não foi testada numericamente em grade além dos OOD selecionados. Logo, a coerência física das equações permanece **evidência insuficiente**.

## E. Auditoria de LLM e regressão simbólica

- As especificações são artefatos executáveis Python e o LLM preenche a função `equation`; coeficientes são otimizados numericamente durante avaliação. O objetivo registrado é `-MSE` no treino, não descoberta simbólica com custo explícito por complexidade.
- O ExperienceBuffer prioriza programas com base no score e disponibiliza candidatos anteriores no prompt (`LLM-SR/llmsr/buffer.py`, especialmente `get_prompt`/`_generate_prompt`). Esse mecanismo é adaptativo e adequado ao algoritmo, mas torna indispensável arquivar prompts e versões para rastreabilidade e comparação.
- Os notebooks configuram seed 7, 100 amostras máximas, 4 respostas por prompt, servidor local em localhost:5000 e um avaliador. Os resultados persistidos não registram identificador do modelo, hash/versão, temperatura, parâmetros de geração, prompt completo, respostas brutas nem checksum do dataset. A seed do cliente não garante repetibilidade do servidor.
- O baseline da análise tem parâmetros declarados: PySR 50 iterações × 6 populações × 30 indivíduos, gplearn 20 gerações × 300 e DEAP 20 × 300. O número de candidatos efetivamente avaliados e o custo total não é pareado com as 100 propostas máximas do LLM-SR. Os conjuntos de operadores também diferem. As diferenças de desempenho não isolam a contribuição da LLM.
- A seleção LLM-SR é feita com score de treino; isso está coerente com a separação de teste. A análise em resultados deve, contudo, preservar o registro de todos os candidatos, checkpoints, prompt, configuração de servidor e previsões para todos os splits. Os arquivos atuais incluem logs de amostras e checkpoints, mas a associação da amostra avaliada às previsões exportadas é incompleta.

## F. Auditoria estatística e métricas

Métricas recalculadas sem chamar funções originais, diretamente dos arquivos `*_best_candidate_holdout.csv` usando as definições padrão de MSE, RMSE, MAE e R²:

| Resultado persistido | N | RMSE em log10(ID) | MAE em log10(ID) | R² |
|---|---:|---:|---:|---:|
| Experimental Org1, curva única | 11 | 0,128 | 0,089 | 0,103 |
| Org1 sintética OOD | 900 | 0,294 | 0,242 | 0,971 |
| Org2 sintética OOD | 1.050 | 0,314 | 0,252 | 0,885 |
| Org3 sintética OOD | 600 | 0,956 | 0,479 | 0,907 |
| Org4 sintética OOD | 1.800 | 1,691 | 1,224 | −0,073 |

Os números de Org3 mostram por que R² alto não basta: RMSE próximo de uma década implica erro multiplicativo grande, apesar da variação ampla do alvo. No holdout experimental, RMSE é baixo em escala de log, mas R² próximo de zero e apenas 11 pontos não autorizam inferência robusta. Os fatores multiplicativos equivalentes a RMSE são aproximadamente `10**RMSE`, e erros logarítmicos simétricos não correspondem a erros simétricos em ampères.

Os resultados `metrics_by_curve.csv` permitem comparação por curva, sem independência entre pontos. RMSE médio por curva em OOD (LLM-SR, PySR, gplearn, DEAP) foi: Org1 `0,288 / 0,335 / 0,801 / 1,691`; Org2 `0,290 / 0,571 / 1,162 / 1,195`; Org3 `0,843 / 0,587 / 1,727 / 2,067`; Org4 `1,404 / 1,543 / 1,587 / 1,629`. Assim, a candidata LLM-SR não vence PySR em Org3 OOD. Org4 é fraca para todos os métodos. Uma média por curva não deve ser lida como estimativa populacional, pois há de 4 a 12 curvas e apenas uma execução registrada por organização.

Não há múltiplas seeds/replicações do benchmark com predições pareadas, intervalos de confiança ou teste estatístico por curva. Não calculei p-valores: com poucas curvas, dependência e seleção exploratória, isso não teria interpretação confiável. A análise por comparação inclui métricas no treino, `test_id` e `test_ood`, mas previsões row-level de `test_id` não foram exportadas para recálculo independente. O recálculo independente executado cobre cinco arquivos de previsões OOD, não todos os modelos.

## G. Rastreabilidade e reprodutibilidade

| Nível | Diagnóstico |
|---|---|
| Resultado reproduzido integralmente | Nenhum. Não rerodei busca LLM, PySR, gplearn ou DEAP: os experimentos são computacionalmente custosos e o serviço histórico/modelo não está identificado. |
| Resultado verificado por recálculo | Cinco arquivos `*_best_candidate_holdout.csv`; métricas acima recalculadas diretamente das predições. A relação `10**log10(ID)` bate com as colunas de corrente exportadas. |
| Compatível com evidências, não reproduzido | Contagens/manifests dos splits, equações/configurações e comparação por curva; consistentes com o que está salvo. |
| Não verificável | Modelo LLM e versão exatos, parâmetros de sampling do servidor, prompts completos, resposta original por request, reexecução do servidor, hash de dados usados, relação causal entre checkpoint e CSV final. |
| Contraditório | Uma alegação geral de superioridade ou validação física não é compatível com Org3 OOD (LLM pior que PySR), Org4 OOD (R² negativo) e ausência de validação experimental externa. |

Os arquivos `model_specifications.json` registram Python 3.11.7, configurações clássicas e seed; os logs de amostras guardam código/score/order. Checkpoints são pickle e não equivalem a um registro hermético do ambiente. Vários manifests preservam caminhos absolutos da máquina de origem que não existem nesta workspace; os adaptadores fazem rebase do nome do arquivo. Isso é manejável para uso local, mas diminui portabilidade e proveniência reproduzível.

## H. Matriz de achados

| ID | Severidade | Categoria | Achado e evidência | Impacto / recomendação |
|---|---|---|---|---|
| A-01 | **Alta** | Validade científica | Os dados sintéticos vêm do gerador e do modelo OTFT do próprio repositório; `index_generated.json` guarda parâmetros de geração. | Resultados não comprovam descoberta física independente nem generalização experimental. Apresentar como benchmark de recuperação/aproximação em simulador; validar depois em dispositivos independentes. |
| A-02 | **Alta** | Generalização / desempenho | OOD Org4: RMSE 1,691, R² −0,073; Org3 LLM-SR OOD RMSE médio por curva 0,843 versus PySR 0,587. Evidência em outputs e `metrics_by_curve.csv`. | Refuta alegação agregada de superioridade robusta/generalização universal. Corrigir conclusão e repetir com protocolo fixo e novos dispositivos. |
| A-03 | **Alta** | Física | Nenhum teste de dimensionalidade de coeficientes, monotonicidade, regiões, condições-limite ou estabilidade física foi implementado/arquivado; equações selecionadas são regressões empíricas em log. | Baixa interpretabilidade e risco de extrapolação não física. Adicionar validação física pré-registrada por família/regime e grade de domínio. |
| A-04 | **Alta** | Estatística | Uma execução por organização e poucos grupos independentes; métricas point-wise tratam varreduras correlacionadas, sem intervalos de confiança por curva. | Superioridade não tem sustentação estatística. Repetir seeds e inferir por curva/dispositivo, não por ponto. |
| A-05 | **Média** | Rastreabilidade LLM | Histórico sem modelo/versão, prompt completo, temperatura/parâmetros e respostas/request vinculados a resultados. | Resultados LLM não reproduzíveis integralmente. Registrar payload completo, checksum, runtime, seed do servidor, candidato e parâmetros ajustados. |
| A-06 | **Média** | Comparabilidade | Orçamentos de busca e espaços de operadores diferentes entre LLM-SR/PySR/gplearn/DEAP; quantidade efetiva/custo de propostas não pareada. | Diferenças não isolam efeito da LLM. Fazer ablação com mesmo orçamento de avaliações e controle de refinamento de coeficientes. |
| A-07 | **Média** | Teste de curva real | Holdout experimental são 11 pontos consecutivos da mesma curva; RMSE 0,128, mas R² 0,103. | Só demonstra ajuste local exploratório; não valida nova curva/dispositivo. Preferir leave-one-curve/device-out e medir mais dispositivos. |
| A-08 | **Baixa** | Métrica/artefatos | CSV de previsão row-level salvo apenas para candidata LLM-SR em OOD; `test_id` e previsões baseline não estão persistidos de modo pareado. | Recalculo independente e diagnóstico de erro incompletos. Exportar previsão/observado por ponto, método, split, curva e seed. |
| A-09 | **Baixa** | Proveniência/portabilidade | Manifestos preservam caminhos absolutos de host anterior; adaptadores os rebaseiam por nome de arquivo. | Pode haver ambiguidade de origem quando houver nomes repetidos. Usar caminhos relativos e hashes SHA-256 dos inputs. |
| A-10 | **Baixa** | Vazamento | Nenhum vazamento direto de curvas/pares de entrada/target observado nos splits sintéticos atuais; 10 testes passaram. Contexto histórico do servidor não está disponível. | Manter essa qualificação: ausência nos testes realizados, não prova universal de ausência de leakage. Arquivar requests e checksums de split. |

## I. Plano de correção priorizado

1. **Delimitar as alegações**: declarar que o resultado é benchmark sintético e curva experimental exploratória; retirar afirmação de descoberta física ou superioridade geral que exceda os dados.
2. **Criar protocolo congelado**: separar desenvolvimento/validação/teste por dispositivo e lote quando disponíveis; reservar teste uma única vez, sem uso em seleção/prompts; manter também o holdout de curvas.
3. **Executar benchmark repetido e pareado**: múltiplas seeds, orçamento igual em número de avaliações e tempo, operadores explicitados, mesmo ajuste de constantes; relatar média e dispersão por curva/dispositivo e incerteza hierárquica.
4. **Persistir tudo**: source/data hashes, manifest relativo, versão do código/ambiente, modelo e parâmetros LLM, prompt/resposta, candidatos/score, equação final e coeficientes, predições por ponto e métricas de todos os métodos/splits.
5. **Validar fisicamente**: manter `ID` assinado em análise paralela; incluir testes de limites, monotonicidade aplicável, singularidades, domínio, unidade e comportamento por regime; comparar as relações descobertas com modelo físico de referência e dados medidos externos.
6. **Ampliar avaliação externa**: novos dispositivos, condições e lotes não utilizados na geração; avaliação por curvas completas e teste de extrapolação definido por objetivo físico.
7. **Adequar inferência estatística**: usar curva/dispositivo como unidade experimental e não ponto; apresentar intervalos e variabilidade entre seeds e grupos.
8. **Completar os artefatos**: exportar `test_id` e previsões de todos os baselines; guardar configuração executada ao lado de cada execução, não apenas notebook com parâmetros correntes.

## J. Parecer científico final

**Classificação: Não validado.**

O escopo cobriu o código disponível de preparação, runners, especificações executáveis, mecanismo do pipeline LLM-SR, notebooks de experimento/análise, manifests, datasets processados, amostras/logs, métricas e CSVs de holdout. Foram executados 10 testes independentes de baixo custo, sem modificar código de produção ou dados originais; cinco conjuntos de métricas foram recalculados e as equações LLM sintéticas foram reavaliadas contra as previsões holdout persistidas.

O parecer “Não validado” refere-se às conclusões científicas amplas de descoberta de relações físicas confiáveis ou superioridade robusta. Há evidência verificável de execução, ajuste e alguns resultados promissores em curvas sintéticas; não há vazamento direto identificado no caminho atual das partições sintéticas. Entretanto, a proveniência sintética circular, falhas OOD (especialmente Org4), ausência de replicações/intervalos, falta de rastreabilidade dos prompts/modelos LLM e inexistência de validação física independente impedem sustentar conclusões gerais. A curva experimental Org1 não resolve essas lacunas: trata-se de um holdout contíguo curto da mesma curva, com R² baixo e sem generalização a novos dispositivos.

### Testes não executados

- Reexecução integral do gerador, do LLM-SR e dos métodos clássicos: não executada para evitar custo e porque o servidor/modelo histórico não está identificado.
- Recalcular métricas `test_id` e todos os baselines por ponto: não executado porque as previsões row-level não foram persistidas.
- GroupKFold/Leave-One-Device-Out: não aplicável aos datasets sintéticos atuais sem identificar réplicas/dispositivos independentes; necessário quando dados experimentais agrupados forem disponibilizados.
- Auditoria de temperatura/parâmetros e conteúdos reais enviados à LLM: não verificável sem logs/request payload do servidor.
- Varredura física além do domínio persistido e comparação numérica completa com o modelo gerador: não executada; exigiria avaliação independente do modelo e fixação explícita do protocolo físico.
