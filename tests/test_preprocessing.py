import cv2
import numpy as np
from pathlib import Path
from src.preprocessing import DocumentPreprocessor


def generate_synthetic_invoice(output_path: Path) -> None:
    """Génère un document de test incliné avec du bruit et un gradient d'ombre."""
    # Fond blanc 800x600
    img = np.ones((800, 600, 3), dtype=np.uint8) * 240

    # Simulation d'un texte de connaissement / facture logistique
    cv2.putText(img, "AGL LOGISTICS - BON DE LIVRAISON", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "CONTAINER ID : MSCU4829104", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "POIDS BRUT   : 24500 KG", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "DATE         : 14/03/2026", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.putText(img, "MONTANT TTC  : 1 450 000 XOF", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    # 1. Incliner volontairement l'image de 6 degrés pour tester le deskewing
    M = cv2.getRotationMatrix2D((300, 400), 6.0, 1.0)
    skewed = cv2.warpAffine(img, M, (600, 800), borderValue=(220, 220, 220))

    # 2. Ajouter du bruit Gaussien (simulation d'un mauvais capteur photo)
    noise = np.random.normal(0, 15, skewed.shape).astype(np.int16)
    noisy = np.clip(skewed.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), noisy)


if __name__ == "__main__":
    sample_in = Path("data/raw/sample_bl.png")
    sample_out = Path("data/processed/sample_bl_cleaned.png")

    generate_synthetic_invoice(sample_in)

    preprocessor = DocumentPreprocessor(target_width=1000)
    result = preprocessor.run_pipeline(sample_in, apply_binarization=True)

    cv2.imwrite(str(sample_out), result["processed_image"])
    print(f"Dimension originale : {result['original_shape']}")
    print(f"Dimension finale    : {result['processed_shape']}")
    print(f"Angle corrigé       : {result['deskew_angle_deg']}°")
    print(f"Image nettoyée sauvegardée dans : {sample_out}")