from utils.align_images import load_images_from_folder
from utils.estimate_scales import algorithm
import pandas as pd
import numpy as np
import cv2
import os

os.system('cls')

# Constantes.
INPUT_PATH = './images/aligned'

INTERVAL = np.linspace(1.00, 1.20, 500)
CUSTO = 'NCC'

# Download e Ordenação das Imagens.
images = load_images_from_folder(INPUT_PATH)


if images:
    # Estimativa das Escalas.
    step_scales, accumulated_scales = algorithm(images, CUSTO, INTERVAL, debug=False)

    # DataFrames.
    imgs_idx = np.arange(0, len(step_scales), 1)

    df_step = pd.DataFrame(columns=['imagens', 'escalas'])
    df_step['imagens'] = imgs_idx
    df_step['escalas'] = step_scales

    df_acmd = pd.DataFrame(columns=['imagens', 'escalas acumuladas'])
    df_acmd['imagens'] = imgs_idx
    df_acmd['escalas acumuladas'] = accumulated_scales

    resume_step = df_step.describe()
    resume_acmd = df_acmd.describe()

    # Salva em disco
    if CUSTO == 'MSE':
        df_step_path = f"./results/scales/mse/escalas_passo.csv"
        resume_step_path = f"./results/scales/mse/resume_passo.csv"
        df_acmd_path = f"./results/scales/mse/escalas_acumuladas.csv"
        resume_acmd_path = f"./results/scales/mse/resume_acumulado.csv"

    elif CUSTO == 'RMSE':
        df_step_path = f"./results/scales/rmse/escalas_passo.csv"
        resume_step_path = f"./results/scales/rmse/resume_passo.csv"
        df_acmd_path = f"./results/scales/rmse/escalas_acumuladas.csv"
        resume_acmd_path = f"./results/scales/rmse/resume_acumulado.csv"

    else:
        df_step_path = f"./results/scales/ncc/escalas_passo.csv"
        resume_step_path = f"./results/scales/ncc/resume_passo.csv"
        df_acmd_path = f"./results/scales/ncc/escalas_acumuladas.csv"
        resume_acmd_path = f"./results/scales/ncc/resume_acumulado.csv"

    df_step.to_csv(df_step_path, index=False)
    resume_step.to_csv(resume_step_path, index=True)

    df_acmd.to_csv(df_acmd_path, index=False)
    resume_acmd.to_csv(resume_acmd_path, index=True)