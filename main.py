from utils.compute_depth import analyze_keypoint_density, compute_focus_stacking, save_focus_results, calculate_depth_map, plot_depth
from utils.align_images import load_images_from_folder, align_images_ecc, is_empty, save_results
from utils.estimate_scales import algorithm
import pandas as pd
import numpy as np
import cv2
import os

os.system('cls')

# Constantes.
MOTOR_STEP_MM = 2
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

# Descrição de PATHs.
CUSTO = input('\nDigite qual método será utilizado para calcular as escalas (MSE, RMSE, NCC ou ECC): ')

if CUSTO == 'MSE':
        df_path = f"./results/scales/mse/escalas.csv"
        resume_path = f"./results/scales/mse/resume.csv"
elif CUSTO == 'RMSE':
    df_path = f"./results/scales/rmse/escalas.csv"
    resume_path = f"./results/scales/rmse/resume.csv"
elif CUSTO == 'ECC':
    df_path = f"./results/scales/ecc/escalas.csv"
    resume_path = f"./results/scales/ecc/resume.csv"
else:
    df_path = f"./results/scales/ncc/escalas.csv"
    resume_path = f"./results/scales/ncc/resume.csv"

# Operar Estimativas.
step1 = input("\nRealizar estimativa das escalas? Sim (1)\n")
if step1 == '1' and CUSTO != 'ECC':
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
    df.to_csv(df_path, index=False)
    resume.to_csv(resume_path, index=True)

else:
    if not is_empty(df_path):
        df = pd.read_csv(df_path)
    else:
        print("ERRO: Certifique-se de calcular as escalas.")

# Análise de Keypoints e Escalas.
kp_stats = analyze_keypoint_density(aligned_images=images, scales=df['escalas'])

# Calcular Profundidade.
step2 = input("\nRealizar estimativa da profundidade? Sim (1)\n")

if step2 == '1':
    depth_map, index_map = calculate_depth_map(images, df['escalas_acumuladas'], MOTOR_STEP_MM)
    plot_depth(depth_map, index_map)