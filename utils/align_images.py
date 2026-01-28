import pandas as pd
import numpy as np
import cv2
import os
import re

def is_empty(dir_path):
    """ Verifica se uma pasta é vazia usando os.scandir(). """

    if not os.path.isdir(dir_path):
        return False
    with os.scandir(dir_path) as entries:
        return next(entries, None) is None

def load_images_from_folder(folder):
    """
    Ordena um conjunto de imagens a partir de uma dataset contendo ordenação 
    númerica crescente a patir da localização da pasta de origem.
    
    """
    
    images = []
    
    try:
        all_files = os.listdir(folder)
    except FileNotFoundError:
        print(f"Erro: A pasta '{folder}' não foi encontrada.")
        return []

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0 

    sorted_files = sorted(image_files, key=extract_number)
    
    print(f"\nOrdem de carregamento detectada: {sorted_files[:5]} ...")

    for filename in sorted_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            images.append(img)
        else:
            print(f"Aviso: Não foi possível ler {filename}")

    return images

def align_images_ecc(images):
    """
    1. Alinha usando ECC.
    2. Rastreia as bordas pretas criadas pelo zoom/shift.
    3. Recorta o resultado final para eliminar áreas sem dados.
    """

    if not images: return [], []
    
    print(f"\nIniciando Alinhamento ECC com Recorte ({len(images)} imagens)")
    
    # Referência - Ground Truth.
    ref_img = images[0]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = ref_img.shape[:2]
    
    aligned_images = [ref_img]
    scales = [1.0]
    acummulated_scales = scales.copy()
    common_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Configuração ECC (Critérios de parada).
    warp_mode = cv2.MOTION_AFFINE
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    for i in range(1, len(images)):
        img_curr = images[i]
        curr_gray = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        try:
            # Calcula passo ECC
            step_matrix = np.eye(2, 3, dtype=np.float32)
            cc, step_matrix = cv2.findTransformECC(ref_gray, curr_gray, step_matrix, warp_mode, criteria)
            
            # Acumula Matriz (dot product).
            step_3x3 = np.vstack([step_matrix, [0, 0, 1]])
            warp_3x3_accum = np.vstack([warp_matrix, [0,0,1]])
            combined = np.dot(step_3x3, warp_3x3_accum)
            warp_matrix = combined[:2, :]
            
            # Alinha a Imagem.
            img_aligned = cv2.warpAffine(img_curr, warp_matrix, (width, height), 
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
            aligned_images.append(img_aligned)
            
            # Atualiza a Máscara de Validade.
            mask_curr = np.ones((height, width), dtype=np.uint8) * 255            
            mask_warped = cv2.warpAffine(mask_curr, warp_matrix, (width, height), 
                                         flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            common_mask = cv2.bitwise_and(common_mask, mask_warped)
            
            # Calcula Escala.
            s_x = np.sqrt(warp_matrix[0,0]**2 + warp_matrix[1,0]**2)
            scales.append(s_x)
            acummulated_scales.append(acummulated_scales[-1]*s_x)
            
            print(f"Img {i} Alinhada (ECC). Escala: {s_x:.4f}")

            # Atualiza Referência.
            ref_gray = curr_gray
            
        except cv2.error:
            print(f"Falha ECC na img {i}. Usando anterior.")
            aligned_images.append(aligned_images[-1])
            scales.append(scales[-1])
            acummulated_scales.append(acummulated_scales[-1]**2)
            

    # Corte
    contours, _ = cv2.findContours(common_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        print(f"\nÁrea válida detectada: \nX: {x}, Y: {y}, Largura: {w}, Altura: {h}")
        print(f"Pixels perdidos nas bordas: \nHorizontal: {width-w}, Vertical: {height-h}")
        
        final_images = []
        for img in aligned_images:
            # Recorta todas as imagens no mesmo retângulo seguro.
            cropped = img[y:y+h, x:x+w]
            final_images.append(cropped)
            
        return final_images, scales, acummulated_scales
    else:
        print("Erro Crítico: A máscara comum ficou vazia! (Overlap insuficiente). Retornando originais.")
        return aligned_images, scales, acummulated_scales
    
def save_results(aligned_images, scales, acummulated_scales):
    """
    Salva as imagens processadas e um arquivo CSV com os dados de escala/zoom.
    """

    output_folder="./images/aligned"
    csv_path = "./results/scales/ecc/escalas.csv"
    resume_csv_path = "./results/scales/ecc/resume.csv"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Pasta '{output_folder}' criada.")
    print(f"\nSalvando {len(aligned_images)} imagens e dados em '{output_folder}'...")
    
    with open(csv_path, mode='w', newline='') as file:
        for i, _ in enumerate(scales):
            filename = f"aligned_{i:04d}.jpg"
            filepath = os.path.join(output_folder, filename)

            cv2.imwrite(filepath, aligned_images[i])

    imgs_idx = np.arange(0, len(scales), 1)
    df = pd.DataFrame(columns=['imagens', 'escalas', 'escalas_acumuladas'])
    df['imagens'] = imgs_idx
    df['escalas'] = scales
    df['escalas_acumuladas'] = acummulated_scales
    resume = df.describe()

    df.to_csv(csv_path, index=False)
    resume.to_csv(resume_csv_path, index=True)
        
    print("Concluído! Todos os arquivos foram salvos.")
    return csv_path