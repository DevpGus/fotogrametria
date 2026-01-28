from utils.align_images import load_images_from_folder, align_images_ecc, is_empty, save_results
from utils.estimate_scales import algorithm
import pandas as pd
import numpy as np
import cv2
import os

os.system('cls')

# Constantes.
INPUT_ALIGNED_PATH = './images/aligned'
INPUT_RAW_PATH = './images/ordered'
INTERVAL = np.linspace(1.00, 1.20, 500)

# Download e Ordenação das Imagens.
if is_empty(INPUT_ALIGNED_PATH):
    raw_images = load_images_from_folder(INPUT_RAW_PATH)
    aligned_imgs, scales, accumulated_scales = align_images_ecc(raw_images)
    save_results(aligned_imgs, scales, accumulated_scales)

# Imagens Alinhadas.
images = load_images_from_folder(INPUT_ALIGNED_PATH)

control = input("\nRealizar Estimativa das Escalas? Sim (1)\n")
if control == '1':
    CUSTO = input('\nDigite a Função de Custo que será utilizada (MSE, RMSE ou NCC): ')

    os.system('cls')

    # Estimativa das Escalas.
    step_scales, accumulated_scales = algorithm(images, CUSTO, INTERVAL, debug=False)

    # DataFrames.
    imgs_idx = np.arange(0, len(step_scales), 1)

    df = pd.DataFrame(columns=['imagens', 'escalas', 'escalas_acumuladas'])
    df['imagens'] = imgs_idx
    df['escalas'] = step_scales
    df['escalas_acumuladas'] = accumulated_scales

    resume = df.describe()

    # Salva em disco
    if CUSTO == 'MSE':
        df_path = f"./results/scales/mse/escalas.csv"
        resume_path = f"./results/scales/mse/resume.csv"

    elif CUSTO == 'RMSE':
        df_path = f"./results/scales/rmse/escalas.csv"
        resume_path = f"./results/scales/rmse/resume.csv"

    else:
        df_path = f"./results/scales/ncc/escalas.csv"
        resume_path = f"./results/scales/ncc/resume.csv"

    df.to_csv(df_path, index=False)
    resume.to_csv(resume_path, index=True)

# Calcular Profundidade.