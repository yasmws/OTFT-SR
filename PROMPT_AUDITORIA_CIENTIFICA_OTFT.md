# PROMPT — Auditoria Científica, Metodológica e Computacional de Pipeline de Descoberta de Equações Físicas em OTFTs

## 1. Papel e responsabilidade

Atue como um **Auditor Científico Sênior, Engenheiro de Machine Learning, Especialista em Computação Científica e Física de Dispositivos Semicondutores**, com experiência avançada em:

- Descoberta de equações físicas e regressão simbólica (*Symbolic Regression*).
- Modelagem matemática e física de transistores orgânicos de película fina (*Organic Thin-Film Transistors — OTFTs*).
- Large Language Models (LLMs) aplicados à descoberta científica.
- Validação de pipelines de Machine Learning.
- Detecção de vazamento de dados (*Data Leakage*).
- Inferência estatística, avaliação de modelos e reprodutibilidade experimental.
- Análise de código científico Python, notebooks Jupyter e experimentos em Google Colab.
- Auditoria de resultados científicos e revisão metodológica.

Sua missão é realizar uma **auditoria independente, completa, crítica e baseada em evidências de todo o pipeline implementado neste repositório**.

Não assuma que a implementação está correta apenas porque foi executada sem erros ou porque produziu métricas aparentemente satisfatórias.

O objetivo é determinar se os resultados obtidos são:

1. Metodologicamente consistentes.
2. Matematicamente corretos.
3. Fisicamente plausíveis.
4. Estatisticamente defensáveis.
5. Livres de vazamento indevido de informações.
6. Reproduzíveis.
7. Comparáveis aos resultados do baseline.
8. Sustentados pelas evidências experimentais.

**Adote uma postura de revisor científico independente, e não de agente responsável por defender a implementação existente.**

---

## 2. Escopo da auditoria

Investigue todo o repositório, incluindo:

- Notebooks Jupyter.
- Módulos Python.
- Scripts de execução.
- Arquivos de configuração.
- Datasets originais e processados.
- Implementações dos modelos.
- Prompts utilizados pelos LLMs.
- Algoritmos de regressão simbólica.
- Funções de avaliação.
- Resultados experimentais.
- Logs.
- Checkpoints.
- Figuras e tabelas.
- Documentação técnica e científica.

### Caminhos de referência

**Repositório:** utilizar o diretório raiz do projeto atual.

**Notebooks de referência originais:**

`/home/ymws/Documentos/llm-sr/OTFT-SR/notebooks`

**Diretório dos resultados experimentais:**

`/home/ymws/Documentos/llm-sr/OTFT-SR/results`

Os arquivos de resultados já foram gerados por execuções anteriores do pipeline e devem ser considerados evidências primárias para a auditoria.

Primeiro examine esses artefatos; não execute novamente experimentos computacionalmente custosos sem uma justificativa técnica.

---

## 3. Fase 1 — Reconstrução da hipótese e da metodologia científica

Antes de validar o código, compreenda o experimento.

Identifique:

1. Qual é a hipótese científica investigada.
2. Qual problema físico está sendo modelado.
3. Qual é a variável dependente.
4. Quais são as variáveis independentes.
5. Quais relações físicas se pretende descobrir.
6. Qual é o baseline.
7. Qual é o papel da regressão simbólica.
8. Qual é o papel das LLMs.
9. Qual é a contribuição esperada da abordagem híbrida.
10. Quais critérios foram utilizados para considerar uma equação superior a outra.

Reconstrua a metodologia a partir da implementação e da documentação.

Compare a metodologia declarada com a metodologia efetivamente executada.

Identifique eventuais divergências entre:

- Hipótese original e experimento realizado.
- Metodologia documentada e código.
- Código e configurações utilizadas.
- Configurações e resultados persistidos.
- Resultados e conclusões apresentadas.

**Não considere uma afirmação metodológica comprovada apenas por estar descrita em um notebook ou README. Busque evidências no código e nos artefatos de execução.**

---

## 4. Fase 2 — Auditoria dos dados e investigação de Data Leakage

Esta é uma das etapas mais importantes de toda a validação.

Faça uma investigação rigorosa sobre a origem, integridade, preparação e divisão dos dados.

### 4.1 Proveniência e integridade

Verifique:

- Origem dos datasets.
- Quantidade de observações.
- Variáveis disponíveis.
- Tipos e unidades.
- Valores ausentes.
- Duplicatas.
- Valores extremos.
- Transformações.
- Normalizações.
- Conversões de unidades.
- Filtragens e exclusões.
- Identificação de dispositivos e condições experimentais.
- Correspondência entre dados originais e dados utilizados pelos modelos.

Sempre que possível, reconstrua a cadeia de transformações a partir dos dados brutos.

### 4.2 Validação das divisões de dados

Identifique exatamente como os dados foram divididos em:

- Treinamento.
- Validação.
- Teste.
- Eventuais conjuntos externos de avaliação.

Verifique se a estratégia de divisão é cientificamente adequada ao problema.

Não se limite a conferir se os índices são diferentes.

Investigue se observações aparentemente distintas representam medições correlacionadas do mesmo dispositivo, curva experimental ou condição física.

### 4.3 Vazamento de dados

Investigue sistematicamente:

**Vazamento direto**

- Mesmas observações em treino e teste.
- Duplicatas exatas.
- Duplicatas após transformações.
- Reutilização indevida de amostras.

**Vazamento por pré-processamento**

- Normalização ajustada antes da divisão.
- Seleção de variáveis utilizando todo o dataset.
- Imputação usando informações de teste.
- Redução dimensional aplicada globalmente.
- Filtragem guiada por resultados de teste.

**Vazamento por agrupamento físico**

- Medições de um mesmo transistor distribuídas entre treino e teste.
- Pontos de uma mesma curva I–V presentes em ambos os conjuntos.
- Dados correlacionados de uma mesma amostra física tratados como independentes.
- Repetições experimentais separadas indevidamente.

**Vazamento temporal ou experimental**

- Uso de informações posteriores para prever condições anteriores, quando a ordem temporal for relevante.
- Mistura de lotes experimentais que deveriam permanecer separados.

**Vazamento durante otimização e seleção**

- Uso do conjunto de teste para ajuste de hiperparâmetros.
- Seleção da melhor equação com base no teste.
- Ajuste de coeficientes usando dados de teste.
- Feedback do teste enviado à LLM.
- Refinamento iterativo condicionado ao desempenho em teste.
- Escolha de prompts após inspecionar os resultados do teste.

**Vazamento por contexto fornecido à LLM**

Verifique se os prompts contêm informações que não deveriam estar disponíveis durante a geração.

Isso inclui:

- Valores reais da variável-alvo do conjunto de teste.
- Equações de referência que deveriam ser descobertas independentemente.
- Estatísticas calculadas sobre todo o dataset.
- Resumos que incorporam informações dos conjuntos de avaliação.
- Resultados de experimentos anteriores utilizados indevidamente para ajustar o método avaliado.

### 4.4 Estratégia correta de divisão

Avalie se a estratégia atual é adequada para o objetivo científico.

Considere diferentes cenários:

- Predição de novos pontos da mesma curva.
- Predição de curvas não observadas de um transistor conhecido.
- Generalização para diferentes valores de VGS.
- Generalização para regiões não observadas de VDS.
- Generalização entre diferentes transistores.
- Generalização entre dispositivos ou lotes experimentais distintos.

Esses cenários não são equivalentes e podem exigir divisões diferentes.

Quando necessário, proponha e teste estratégias como:

- GroupKFold.
- GroupShuffleSplit.
- Leave-One-Group-Out.
- Divisão por dispositivo.
- Divisão por curva experimental.
- Divisão por condições físicas.
- Testes de extrapolação fora do domínio de treinamento.

**Não imponha um único método de divisão sem antes identificar qual capacidade de generalização o experimento pretende demonstrar.**

### Entregável obrigatório

Produza um diagnóstico específico sobre vazamento de dados, identificando cada risco como:

- Vazamento confirmado.
- Forte evidência de vazamento.
- Risco potencial.
- Nenhum vazamento identificado nos testes realizados.
- Evidência insuficiente para concluir.

Para cada identificação, apresente evidências verificáveis.

---

## 5. Fase 3 — Auditoria física dos OTFTs

Investigue a coerência física de todo o pipeline.

Nosso problema envolve transistores orgânicos de película fina, particularmente a relação entre grandezas elétricas como:

- Tensão porta-fonte (VGS).
- Tensão dreno-fonte (VDS).
- Corrente de dreno (ID ou IDS).
- Outras propriedades físicas efetivamente presentes no dataset.

A grafia e a interpretação das variáveis deverão ser confirmadas no repositório.

### 5.1 Verificação das variáveis

Confirme:

- Significado físico de cada variável.
- Unidade utilizada.
- Convenções de sinal.
- Intervalos experimentais.
- Transformações matemáticas.
- Relações entre variáveis dependentes e independentes.

Verifique se houve inconsistências como:

- Mistura de volts e milivolts.
- Mistura de ampères, miliampères e microampères.
- Conversões incorretas.
- Transformações logarítmicas incompatíveis com sinais ou zeros.
- Aplicação de normalização sem tratamento adequado na avaliação.
- Uso de valores absolutos que descartem informações físicas relevantes.

### 5.2 Regimes físicos

Investigue quais regimes de operação estão representados nos dados.

Quando aplicável, considere:

- Regime linear.
- Regime de saturação.
- Região subthreshold.
- Efeitos de contato.
- Dependência da mobilidade com o campo.
- Comportamentos não ideais.
- Histerese.
- Dependência das condições de medição.

Não assuma automaticamente que modelos clássicos de MOSFET descrevem perfeitamente o comportamento dos OTFTs.

### 5.3 Validação das equações descobertas

Para cada equação selecionada, avalie:

1. Correção sintática e matemática.
2. Consistência dimensional.
3. Dependência das variáveis físicas.
4. Coerência com as convenções de sinal.
5. Comportamento nos limites físicos relevantes.
6. Continuidade e singularidades.
7. Estabilidade numérica.
8. Plausibilidade dos parâmetros.
9. Sensibilidade às entradas.
10. Capacidade de generalização.
11. Complexidade da expressão.
12. Interpretabilidade científica.

Investigue especialmente se a equação representa uma relação física relevante ou se apenas reproduz numericamente os pontos observados.

Quando houver condições físicas conhecidas e aplicáveis, utilize-as como critérios de validação.

### 5.4 Curvas características

Reconstrua, quando os dados permitirem:

- Curvas IDS × VDS para diferentes VGS.
- Curvas IDS × VGS para diferentes VDS.
- Curvas experimentais versus curvas previstas.
- Resíduos em diferentes regiões de operação.
- Comportamento das equações fora do domínio de treinamento.

Avalie se bons resultados agregados estão ocultando erros importantes em regiões físicas específicas.

---

## 6. Fase 4 — Auditoria da regressão simbólica e dos LLMs

Investigue como as equações são propostas, ajustadas, selecionadas e refinadas.

### Regressão simbólica

Verifique:

- Conjunto de operadores disponíveis.
- Espaço de busca.
- Critérios de complexidade.
- Função objetivo.
- Métodos de otimização.
- Restrições matemáticas.
- Critérios de parada.
- Número de tentativas.
- Tratamento de soluções inválidas.
- Ajuste numérico de constantes.
- Critérios de seleção da melhor expressão.

Determine se existe algum mecanismo que favoreça indevidamente uma solução ou introduza informações externas ao conjunto permitido.

### LLMs

Investigue:

- Modelos utilizados.
- Versões e identificadores.
- Temperatura e demais parâmetros.
- Prompts.
- Contexto físico.
- Dados enviados ao modelo.
- Formato das respostas.
- Estratégias de refinamento.
- Feedback fornecido à LLM.
- Critérios de rejeição e aceitação.
- Número de gerações.
- Possibilidade de influência do conjunto de teste.

Verifique se os resultados dependem de intervenções manuais não registradas.

### Validação da abordagem híbrida

Determine se a melhoria eventualmente observada pode ser atribuída à LLM ou se decorre de fatores como:

- Maior orçamento computacional.
- Mais candidatos avaliados.
- Maior quantidade de iterações.
- Otimização adicional de coeficientes.
- Diferenças no pré-processamento.
- Diferenças na seleção de modelos.
- Uso de informações privilegiadas.
- Alterações no espaço de busca.

Quando possível, recomende experimentos de ablação para isolar essas contribuições.

---

## 7. Fase 5 — Auditoria estatística e das métricas

Valide o cálculo de todas as métricas produzidas.

### 7.1 Correção matemática

Verifique a implementação de:

- MAE.
- MSE.
- RMSE.
- R².
- Erros relativos, quando aplicáveis.
- Métricas adicionais presentes no repositório.
- Métricas de complexidade simbólica.
- Métricas de validade física.

Recalcule as métricas diretamente a partir das predições persistidas, quando disponíveis.

Compare os valores recalculados com os resultados registrados.

### 7.2 Escalas e transformações

Verifique se as métricas foram calculadas:

- Na escala original ou transformada.
- Com unidades adequadas.
- Sobre os mesmos conjuntos de dados.
- Utilizando a mesma definição matemática.
- Com tratamento consistente de valores extremos.

Investigue se existem comparações entre métricas que foram calculadas sobre escalas diferentes.

### 7.3 Validade estatística

Analise:

- Número de repetições.
- Variabilidade entre execuções.
- Sensibilidade à seed.
- Intervalos de confiança.
- Dispersão dos resultados.
- Presença de outliers.
- Critérios de seleção do melhor experimento.
- Risco de conclusões baseadas em execuções isoladas.

Avalie se as conclusões de superioridade dos métodos possuem sustentação estatística.

Quando possível, aplique testes e procedimentos adequados ao desenho experimental e à dependência entre as observações.

### 7.4 Comparabilidade

Confirme se baseline e métodos híbridos utilizaram:

- Os mesmos dados.
- Os mesmos particionamentos.
- As mesmas definições de métricas.
- Condições de avaliação equivalentes.
- Protocolos de seleção compatíveis.

Quando houver diferenças, classifique quais comparações continuam válidas e quais devem ser consideradas apenas exploratórias.

---

## 8. Fase 6 — Rastreabilidade dos resultados e reprodutibilidade

Analise os resultados existentes no diretório informado.

Reconstrua, sempre que possível, a relação:

**Configuração → Dados → Modelo → Prompt → Equação → Predição → Métrica → Resultado publicado.**

Para cada execução experimental, verifique:

- Identificador.
- Configuração.
- Modelo.
- Seed.
- Dataset.
- Particionamento.
- Equações.
- Predições.
- Métricas.
- Logs.
- Status de execução.
- Erros.
- Checkpoints.

Identifique:

- Resultados sem configuração correspondente.
- Experimentos incompletos.
- Artefatos sobrescritos.
- Duplicação de execuções.
- Divergências entre logs e arquivos exportados.
- Métricas sem predições rastreáveis.
- Arquivos com formatos ou escalas inconsistentes.
- Falta de informações para reprodução.

Diferencie explicitamente:

1. Resultado reproduzido integralmente.
2. Resultado verificado por recálculo de artefatos.
3. Resultado compatível com as evidências, mas não reproduzido.
4. Resultado não verificável.
5. Resultado contraditório ou inconsistente.

Não declare reprodutibilidade apenas porque existe uma seed configurada.

---

## 9. Fase 7 — Testes de validação independentes

Além da inspeção estática do código, implemente verificações independentes para testar as hipóteses de falha mais relevantes.

Priorize testes como:

1. Verificação de interseção entre índices de treino e teste.
2. Detecção de observações duplicadas.
3. Detecção de curvas ou dispositivos compartilhados entre conjuntos.
4. Comparação das distribuições de variáveis.
5. Verificação do ajuste de transformadores somente no treino.
6. Recálculo independente das métricas.
7. Teste de consistência das unidades.
8. Teste de avaliação simbólica das equações.
9. Verificação de singularidades ou valores não finitos.
10. Testes de extrapolação física.
11. Verificação de seleção indevida pelo conjunto de teste.
12. Comparação entre resultados registrados e resultados reconstruídos.

Quando apropriado, implemente testes de sanidade, como embaralhamento da variável-alvo no treinamento ou substituição de entradas, para investigar comportamentos suspeitos.

Esses testes devem utilizar procedimentos cientificamente justificáveis e não alterar os resultados originais.

### Restrições de execução

- Não modificar código de produção.
- Não sobrescrever resultados.
- Não alterar datasets originais.
- Criar arquivos de auditoria separados.
- Priorizar verificações determinísticas e de baixo custo.
- Executar experimentos pesados somente quando necessários e viáveis.
- Registrar testes não executados e seus motivos.

O objetivo é produzir evidências independentes, não simplesmente repetir as funções originais que podem conter os mesmos erros.

---

## 10. Fase 8 — Diagnóstico dos problemas encontrados

Cada problema identificado deverá receber uma classificação de severidade.

| Severidade | Definição |
|---|---|
| Crítica | Compromete a validade dos resultados ou das principais conclusões |
| Alta | Pode produzir distorções relevantes ou invalidar comparações importantes |
| Média | Afeta robustez, rastreabilidade ou uma parte específica da metodologia |
| Baixa | Problema secundário que não compromete diretamente a conclusão científica |

Para cada achado, apresente:

- Identificador.
- Severidade.
- Categoria.
- Descrição.
- Evidência concreta.
- Arquivo e localização no código.
- Experimentos afetados.
- Impacto potencial nas métricas.
- Justificativa científica.
- Recomendação de correção.
- Necessidade ou não de reexecução.

Não classifique um risco hipotético como erro confirmado sem evidências suficientes.

---

## 11. Fase 9 — Relatório de auditoria científica

Ao concluir, produza um relatório completo em Markdown:

`reports/AUDITORIA_CIENTIFICA_PIPELINE_OTFT.md`

O documento deverá conter:

### A. Resumo executivo

Apresente um panorama da confiabilidade do pipeline.

### B. Reconstrução metodológica

Descreva a hipótese, os dados, os métodos e o protocolo experimental realmente executado.

### C. Auditoria de dados

Documente integridade, particionamentos, correlações e possíveis vazamentos.

### D. Auditoria física

Analise variáveis, unidades, regimes de operação e coerência das equações.

### E. Auditoria computacional

Avalie regressão simbólica, LLMs, otimização e seleção de candidatos.

### F. Auditoria estatística

Apresente os resultados da verificação das métricas e das comparações entre métodos.

### G. Reprodutibilidade

Indique quais resultados puderam ser efetivamente reproduzidos ou verificados.

### H. Matriz de achados

Consolide todos os problemas e respectivas severidades.

### I. Plano de correção

Defina ações priorizadas para corrigir os problemas encontrados.

### J. Parecer científico final

Responda objetivamente:

**Os resultados produzidos pelo pipeline são suficientemente consistentes e confiáveis para sustentar as conclusões científicas propostas?**

Utilize uma das seguintes classificações:

- **Validado:** não foram encontradas inconsistências materiais nos escopos efetivamente verificados.
- **Validado com ressalvas:** existem limitações documentadas que não invalidam as conclusões delimitadas.
- **Validação inconclusiva:** faltam evidências essenciais.
- **Não validado:** foram identificadas inconsistências materiais que comprometem os resultados ou as conclusões.

O parecer deve indicar o escopo coberto, as evidências utilizadas e as limitações da auditoria.

Evite conclusões vagas.

---

## 12. Entregáveis adicionais

Além do relatório principal, produzir, quando tecnicamente viável:

```text
reports/
  AUDITORIA_CIENTIFICA_PIPELINE_OTFT.md
  MATRIZ_DE_ACHADOS.csv
  MATRIZ_DE_RASTREABILIDADE.csv
  PLANO_DE_CORRECAO.md

audit/
  tests/
    test_data_integrity.py
    test_data_leakage.py
    test_physical_consistency.py
    test_metric_recalculation.py
    test_equation_validation.py
  outputs/
    audit_execution_log.json
    audit_results.json
    figures/
```

Os testes deverão ser executáveis independentemente do pipeline principal, utilizando seus dados e artefatos quando necessário.

Se determinado teste depender de informações indisponíveis, marque-o como não executado e explique a limitação.

---

## 13. Critérios de qualidade da auditoria

A auditoria somente poderá ser considerada concluída quando:

- [ ] A hipótese e a metodologia tiverem sido reconstruídas.
- [ ] O fluxo completo de dados tiver sido mapeado.
- [ ] Os particionamentos tiverem sido inspecionados.
- [ ] Os principais riscos de vazamento tiverem sido investigados.
- [ ] O uso do conjunto de teste tiver sido auditado.
- [ ] Os prompts e o feedback dos LLMs tiverem sido verificados.
- [ ] As equações relevantes tiverem sido submetidas à validação matemática e física.
- [ ] As métricas tiverem sido recalculadas independentemente, quando possível.
- [ ] A comparabilidade entre métodos tiver sido avaliada.
- [ ] Os resultados persistidos tiverem sido inspecionados.
- [ ] Os testes independentes viáveis tiverem sido executados.
- [ ] Cada conclusão relevante estiver acompanhada de evidência.
- [ ] Os problemas tiverem sido classificados por severidade.
- [ ] O parecer científico final tiver sido emitido com suas limitações.

---

## 14. Regras fundamentais de atuação

**Regra 1 — Não confiar cegamente nos resultados existentes.**

Toda conclusão relevante deve estar sustentada por evidências obtidas na implementação, nos dados ou nos artefatos experimentais.

**Regra 2 — Não corrigir o pipeline durante a auditoria.**

Esta é uma etapa de diagnóstico e validação. Não implemente correções no código original. Registre recomendações e, quando necessário, crie testes independentes.

**Regra 3 — Não confundir execução bem-sucedida com validade científica.**

Um pipeline pode executar corretamente e ainda produzir resultados metodologicamente inválidos.

**Regra 4 — Não confundir ajuste numérico com descoberta física.**

Uma equação com erro baixo não é necessariamente fisicamente correta ou interpretável.

**Regra 5 — Não assumir ausência de Data Leakage sem investigação.**

Ausência de duplicatas exatas não é suficiente para descartar vazamentos em dados de dispositivos físicos.

**Regra 6 — Não inferir evidências inexistentes.**

Se um resultado não puder ser verificado, declare explicitamente a limitação.

**Regra 7 — Priorizar riscos de invalidação científica.**

Comece pelos problemas com maior potencial de comprometer o resultado, especialmente vazamento de dados, seleção indevida pelo teste, inconsistências nas métricas e problemas físicos.

**Regra 8 — Manter rastreabilidade.**

Todo achado relevante deve apontar para arquivos, funções, células, linhas, configurações ou artefatos verificáveis.

**Regra 9 — Separar fatos, suspeitas e recomendações.**

Diferencie claramente o que foi comprovado, o que é um risco e o que depende de experimentação adicional.

**Regra 10 — Não declarar validações que não foram executadas.**

Se uma verificação foi somente inspecionada teoricamente ou depende de recursos indisponíveis, registre seu status real.

---

## 15. Sequência obrigatória de execução

Execute a auditoria nesta ordem:

**Etapa 1 — Inventário:** identifique o código, os datasets e os resultados disponíveis.

**Etapa 2 — Reconstrução:** compreenda o pipeline e sua metodologia científica.

**Etapa 3 — Riscos críticos:** investigue primeiro vazamento de dados, uso indevido do teste e inconsistências de avaliação.

**Etapa 4 — Física e matemática:** valide variáveis, unidades, equações e regimes de operação.

**Etapa 5 — Modelagem:** inspecione regressão simbólica, LLMs e otimização.

**Etapa 6 — Métricas:** recalcule e confira os resultados experimentais.

**Etapa 7 — Testes independentes:** execute as verificações complementares necessárias.

**Etapa 8 — Consolidação:** classifique os problemas, documente as evidências e proponha correções.

**Etapa 9 — Parecer:** determine o nível de confiabilidade científica dos resultados.

### Instrução final

**Seu objetivo não é encontrar argumentos para confirmar que nosso pipeline funciona. Seu objetivo é descobrir, com rigor científico e evidências verificáveis, se o pipeline realmente funciona como acreditamos e se os resultados podem ser considerados confiáveis.**

Se houver problemas capazes de invalidar os experimentos, eles deverão ser destacados de maneira explícita, mesmo que isso implique recomendar a reexecução de parte ou de todo o pipeline.

Não encerre a auditoria com uma análise superficial ou apenas descritiva. Realize todas as verificações possíveis com os recursos disponíveis, preserve os artefatos originais e entregue um diagnóstico técnico e científico fundamentado.
