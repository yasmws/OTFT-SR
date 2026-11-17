# Plano de correção priorizado

1. **Reescrever as conclusões para corresponder à evidência atual.** Descrever os dados sintéticos como benchmark gerado pelo simulador do próprio repositório. Restringir a curva experimental a um estudo exploratório de uma curva Org1 e retirar conclusões gerais de descoberta física/superioridade.
2. **Congelar protocolo de avaliação externa.** Definir dispositivo, lote e curva como unidades de generalização distintas. Quando houver IDs de dispositivo/lote, usar GroupKFold ou Leave-One-Device-Out; manter `test_id` para interpolação de polarizações e `test_ood` para extrapolação declarada. O teste externo não participa de prompt, seleção nem ajuste.
3. **Parear os métodos.** Igualar o orçamento por avaliações de candidatos e tempo; documentar operadores, restrições, otimização de coeficientes e critérios de parada. Separar o ganho da LLM do ganho de otimização/refinamento por ablações sem LLM e com mesmas propostas.
4. **Repetir rodadas.** Executar múltiplas seeds por método, persistir todas as métricas por curva e resumir com incerteza entre curvas/dispositivos, sem tratar pontos de uma mesma varredura como réplicas independentes.
5. **Adicionar validação física.** Declarar unidades e domínio; avaliar sinal em paralelo, comportamento nas bordas, continuidade, singularidades e derivadas/regimes físicos aplicáveis. Marcar expressões sem interpretação dimensional como modelos empíricos em escala log, não equações constitutivas.
6. **Completar rastreabilidade.** Em cada execução salvar hashes do CSV, split manifest e código; versão do Python/dependências; ID e versão do LLM; parâmetros do sampling; prompt e resposta completos; score; código candidato; coeficientes finais; predições de treino, validação e teste de todos os métodos.
7. **Fortalecer validação experimental.** Avaliar várias curvas de transferência e saída em diferentes dispositivos/lotes independentes e comparar contra o modelo OTFT de referência. Não reutilizar conjuntos usados para calibrar gerador ou seleção de expressão.
8. **Tornar partições portáveis.** Substituir caminhos absolutos por caminhos relativos e gravar SHA-256 de origem de cada CSV e NPZ.

## Verificações já disponíveis

Execute `python OTFT-SR/audit/run_audit_checks.py`. Os scripts são somente leitura para dados e código de produção; saídas ficam em `OTFT-SR/audit/outputs/`.
