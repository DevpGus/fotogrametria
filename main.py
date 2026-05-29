from utils.compute_depth import keypoint_analysis, depth_map, plot_depth, save_point_cloud
from utils.align_images import load_images, align_images, is_empty, save_results, focus_analysis
from utils.estimate_scales import algorithm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os
import sys

os.system('cls')

print("---- INICIALIZANDO ALGORITMO DE RECONSTRUÇÃO 3D ----")

# Constantes.
MOTOR_STEP_MM = 0.001
INPUT_ALIGNED_PATH = './images/aligned'
INPUT_RAW_PATH = './images/ordered'
INTERVAL = np.linspace(1.00, 1.20, 500)

#



#



#



# Fase 1: Ordenação, Alinhamento e Recorte de Imagens (ECC).
if is_empty(INPUT_ALIGNED_PATH):
    raw_images = load_images(INPUT_RAW_PATH)

    aligned_imgs, scales, accumulated_scales = align_images(raw_images)

    save_results(aligned_imgs, scales, accumulated_scales)
images = load_images(INPUT_ALIGNED_PATH) # Por que eu retomo as imagens com cores?

# Método avaliativo - Diferença entre as imagens.
rmse = []
print("\n")
for i in range(0, len(images)-1):

    ref = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY).astype(np.float32)
    warp = cv.cvtColor(images[i+1], cv.COLOR_BGR2GRAY).astype(np.float32)

    width, height = ref.shape[1], ref.shape[0]
    E = (ref - warp)
    N = width*height

    frobenius_norm = np.linalg.norm(E, ord='fro')

    rmse.append(round(float(frobenius_norm/np.sqrt(N)), 2))

    print(f"[OS] Avaliação da Transformação da Imagem ({i+1}) para ({0}): {round(float(frobenius_norm/np.sqrt(N)), 2)}")

print(f"\n[OS] Método Avaliativo da Aproximação das Imagens: {rmse[:3]} ... {rmse[-3:]}")

# Avaliação Gráfica 

for i in range(0, len(images)-1):
    img1 = images[0]
    img2 = images[i+1]

# Análise dos pontos de interesse.

    ref = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    warp = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    akaze = cv.AKAZE_create(threshold=0.001)

    kp1, des1 = akaze.detectAndCompute(ref, None)
    kp2, des2 = akaze.detectAndCompute(warp, None)


    matcher = cv.BFMatcher(cv.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)


    # Razão de Lowe (Filtragem de outliers)
    good_matches = []
    ratio_threshold = 0.70 
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    
    print(f"\nKeypoints Imagem (0): {len(kp1)}")
    print(f"Keypoints Imagem ({i}): {len(kp2)}")
    print(f"Correspondências robustas: {len(good_matches)}")
    
    ref_kp = cv.drawKeypoints(ref, kp1, None, (0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    warp_kp = cv.drawKeypoints(warp, kp2, None, (0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img_matches = cv.drawMatches(ref, kp1, warp, kp2, good_matches, None, 
                                  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    


# Mapa Binário
    href, wref = ref.shape[:2]
    hwarp, wwarp = warp.shape[:2]

    mask_ref = np.zeros((href, wref), dtype=np.uint8)
    mask_warp = np.zeros((hwarp, wwarp), dtype=np.uint8)

    for n in range(0, len(kp1)-1):
        point = kp1[n]

        x, y = point.pt
        ix, iy = int(x), int(y)
        
        if 0 <= ix < wref and 0 <= iy < href:
            # mask_ref[iy, ix] = 255
            cv.circle(mask_ref, (ix, iy), radius=3, color=1, thickness=-1)

    for n in range(0, len(kp2)-1):
        point = kp2[n]

        x, y = point.pt
        ix, iy = int(x), int(y)

        if 0 <= ix < wwarp and 0 <= iy < hwarp:
            # mask_warp[iy, ix] = 255
            cv.circle(mask_warp, (ix, iy), radius=3, color=1, thickness=-1)


# Visualização

    fig = plt.figure(figsize=(12, 5), layout="constrained")

    plot1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    plot2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    gs = fig.add_gridspec(nrows=2, ncols=4)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(plot1)
    ax1.set_title('Imagem de Referência (0)')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(plot2)
    ax2.set_title(f'Imagem Transformada ({i+1})')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(ref_kp)
    ax3.set_title(f"Pontos de Interesse: {len(kp1)} pts.")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(warp_kp)
    ax4.set_title(f"Pontos de Interesse: {len(kp2)} pts.")

    ax5 = fig.add_subplot(gs[0, 2:])
    ax5.imshow(img_matches)
    ax5.set_title(f'Pontos Correspondentes: {len(good_matches)} pts.')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(mask_ref)
    ax6.set_title(f'Máscara Binária (0)')
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 3])
    ax7.imshow(mask_warp)
    ax7.set_title(f'Máscara Binária ({i+1})')
    ax7.axis('off')

    plt.tight_layout()
    plt.show()

#
#
#
#
#


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

elif CUSTO == 'ECC':
    df = pd.read_csv(df_path)
    
else:
    sys.exit()

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