"""Genera capturas PNG de cada pestaña de la app para el informe (docs/app/capturas/).

Requiere la app en ejecución y Playwright:
    .venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium
    .venv/bin/streamlit run experimento_dl/app_streamlit.py --server.headless true &
    .venv/bin/python experimento_dl/tools/capturas_app.py [--url http://localhost:8501]

En el visor se sube CT_small.dcm (un corte de TC real incluido en pydicom) o, con --dicom-dir,
los .dcm de un estudio real de RSNA.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "app" / "capturas"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--dicom-dir", default=None, help="carpeta con los .dcm de un estudio")
    a = ap.parse_args()
    if a.dicom_dir:
        files = sorted(str(p) for p in Path(a.dicom_dir).glob("*.dcm"))
    else:
        from pydicom.data import get_testdata_file
        files = [get_testdata_file("CT_small.dcm")]
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        page = p.chromium.launch().new_page(viewport={"width": 1440, "height": 2400}, device_scale_factor=1.25)
        page.goto(a.url)
        page.wait_for_selector("[role=tab]", timeout=60000)
        page.wait_for_timeout(4000)
        tabs = page.locator("[role=tab]")
        for i in range(tabs.count()):
            tabs.nth(i).click()
            page.wait_for_timeout(3000)
            if i == 2:  # visor: subir DICOM real y calcular la entrada del modelo
                page.locator("input[type=file]").first.set_input_files(files)  # 1.er uploader = visor
                page.wait_for_timeout(6000)
                page.get_by_text("Calcular la entrada del modelo").first.click()
                page.wait_for_timeout(6000)
            name = tabs.nth(i).inner_text().split("·")[-1].strip().lower()
            name = name.replace(" ", "_").replace("é", "e").replace("ó", "o").replace("ó", "o")
            path = OUT / f"{i + 1:02d}_{name}.png"
            page.screenshot(path=str(path), full_page=True)
            print("captura:", path.relative_to(REPO))
            if any(page.locator("[data-testid=stException]").all()):
                print("  ¡ATENCIÓN! la pestaña muestra una excepción")


if __name__ == "__main__":
    main()
