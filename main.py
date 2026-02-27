from utils.compute_depth import keypoint_analysis, depth_map, plot_depth, save_point_cloud
from utils.align_images import load_images, align_images, is_empty, save_results, focus_analysis
from utils.estimate_scales import algorithm
import pandas as pd
import numpy as np
import cv2
import os

os.system('cls')

print("---- INICIALIZANDO ALGORITMO DE RECONSTRUÇÃO 3D ----")

# Constantes.
MOTOR_STEP_MM = 0.001
INPUT_ALIGNED_PATH = './images/aligned'
INPUT_RAW_PATH = './images/ordered'
INTERVAL = np.linspace(1.00, 1.20, 500)

# Fase 1: Ordenação, Alinhamento e Recorte de Imagens (ECC).
if is_empty(INPUT_ALIGNED_PATH):
    raw_images = load_images(INPUT_RAW_PATH)

    aligned_imgs, scales, accumulated_scales = align_images(raw_images)
    save_results(aligned_imgs, scales, accumulated_scales)
images = load_images(INPUT_ALIGNED_PATH)


# Fase 2: Estimativa das Escalas por Função de Custo.
CUSTO = input('\nDefina a Função de Custo para a estimativa de escalas (RMSE, MSE, NCC, ECC ou None): ')

if CUSTO == 'MSE':
    dir_path = "./results/scales/mse"
    df_path = f"./results/scales/mse/escalas.csv"
    resume_path = f"./results/scales/mse/resume.csv"
elif CUSTO == 'RMSE':
    dir_path = "./results/scales/rmse"
    df_path = f"./results/scales/rmse/escalas.csv"
    resume_path = f"./results/scales/rmse/resume.csv"
elif CUSTO == 'ECC':
    dir_path = "./results/scales/ecc"
    df_path = f"./results/scales/ecc/escalas.csv"
    resume_path = f"./results/scales/ecc/resume.csv"
elif CUSTO == 'NCC':
    dir_path = "./results/scales/ncc"
    df_path = f"./results/scales/ncc/escalas.csv"
    resume_path = f"./results/scales/ncc/resume.csv"
else:
    quit()

if CUSTO in ['MSE', 'RMSE', 'NCC']:
    os.system('cls')

    # Estimativa das Escalas.
    if is_empty(dir_path):
        step_scales, accumulated_scales = algorithm(images, CUSTO, INTERVAL, debug=True)

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
        df = pd.read_csv(df_path)

# Fase 3: Análise de Keypoints e Escalas.
kp_stats, color_idx = keypoint_analysis(aligned_images=images, scales=df['escalas'])
kp_stats.to_csv('./results/keypoints/kp_stats.csv', index=False)
# Fase 4: Calcular Profundidade.
step2 = input("\n[OS] Prosseguir para cálculo da profundidade? [Enter] ")

if step2 == '':
    depth_final, depth_raw, index_map = depth_map(images, df['escalas_acumuladas'], MOTOR_STEP_MM, agg_window=3, d=5, h_thr=0.21, px_thr=20)
plot_depth(depth_final, index_map)

# Fase 5: Reconstrução 3D.
img = images[color_idx]
save_point_cloud(depth_final, img, filename="./results/models/model.ply")