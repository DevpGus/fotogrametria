# Fossil Photogrammetry: 3D Reconstruction via Focus & Zoom

Este projeto implementa uma abordagem híbrida de **Depth from Focus (DFF)** e **Shape from Zoom** para a reconstrução 3D de fósseis em alta precisão. Utilizando um sistema de trilho motorizado (eixo Z) com encoder, o algoritmo estima a profundidade de cada pixel baseando-se na variação da nitidez (Laplaciano) e na escala geométrica da imagem.

---

## 📐 Base Matemática

A profundidade inicial de cada pixel ($Z_0$) é calculada através da relação entre o deslocamento físico do trilho ($\Delta Z$) e o fator de escala observado ($s$):

$$Z_0 = \frac{s \cdot \Delta Z}{s - 1}$$

Onde:
* **$s > 1$**: Representa a ampliação acumulada da imagem $i$ em relação à imagem base (Imagem 0).
* **$\Delta Z$**: É a distância total percorrida pelo motor (medida via encoder) entre a captura da imagem 0 e a imagem $i$.



---

## 🚀 Metodologia

O fluxo de trabalho consiste em quatro etapas principais:

1.  **Alinhamento e Pré-processamento:** As imagens capturadas são alinhadas para remover micro-vibrações mecânicas, garantindo que o centro óptico seja o pivô da expansão.
2.  **Estimativa de Escala ($s$):** Minimização do erro (MSE/RMSE/NCC) entre pares de imagens para determinar o fator de escala acumulado.
3.  **Medida de Foco:** Aplicação do operador de **Laplace** através da pilha de imagens para identificar o índice da imagem de máxima nitidez por pixel.
4.  **Reconstrução Geométrica:** Aplicação da fórmula projetiva para converter movimento e escala em distância absoluta ($Z$) e posterior inversão para mapa de relevo ($H$).

---

## 🛠️ Funcionalidades

* **Minimização de Erro:** Algoritmo iterativo para busca da escala ótima entre quadros.
* **Métricas Estatísticas:** Suporte para MSE, RMSE e NCC (Normalized Cross-Correlation).
* **Mascaramento de Borda (ROI):** Tratamento de artefatos de transformação para evitar erros de borda no cálculo do erro.
* **Depth Mapping:** Geração de mapas de profundidade e índices de foco totalmente vetorizados em NumPy.
* **Visualização Diagnóstica:** Gráficos de funções de custo (curva em U) e mapas residuais térmicos para validação de cada par.

---

## 📂 Estrutura das Funções

| Função | Descrição |
| :--- | :--- |
| `algorithm()` | Orquestra o processamento da lista e gera os logs de escala e gráficos gerais. |
| `analyze_keypoint_density()` | Detecta Keypoints (AKAZE) em todas as camadas e analisa sua distribuição. |
| `calculate_depth_map()` | Processa o cubo de dados (Laplaciano) e aplica a fórmula de profundidade. |
| `estimate()` | Implementações das métricas de comparação de imagens. |

---

## 📦 Requisitos

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* Pandas
* Matplotlib

---

## 📊 Visualização de Resultados

O algoritmo fornece saídas visuais para garantir a integridade dos dados:
* **Gráfico de Convergência:** Identificação do mínimo global da função de custo.
* **Mapas de Índices:** Representação visual de quais camadas do fóssil estão em foco.
* **Mapas de Profundidade:** Representação visual da profundidade (Z) estimada para cada pixel.
* **Gráfico de Keypoints** Visualização da distribuição de keypoints e sua correlação com as escalas estimadas.