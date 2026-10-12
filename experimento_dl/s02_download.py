"""Descarga autorizada y parcial vía API oficial de Kaggle.

Requisitos: aceptar las reglas de la competencia y definir KAGGLE_API_TOKEN (variable de entorno
o variable de usuario de Windows). El token nunca se imprime ni se registra.

No se descarga el dataset completo: para cada estudio de la muestra se
1) obtiene el número de cortes N con búsqueda binaria sobre la existencia de
   train_images/<UID>/<n>.dcm (N se guarda en results/slice_counts.csv para no repetirlo), y
2) se descargan solo K cortes (common.select_slice_numbers).

Robustez (v2, tras recibir HTTP 429 de Kaggle con 12 workers sin límite de ritmo):
- limitador global de peticiones compartido entre hilos (--min-interval s);
- ante 429: pausa GLOBAL con backoff creciente (60 s -> 600 s); se aborta limpiamente si la
  pausa acumulada supera --max-throttle-min;
- errores de un estudio no detienen el proceso: se registran con status="error";
- log escrito tras cada estudio; archivos ya presentes no se vuelven a descargar;
- orden: prueba -> validación -> entrenamiento (las particiones no cambian; así, si la descarga
  no termina, prueba y validación quedan completas).
Uso: python s02_download.py [--workers 4] [--min-interval 1.0] [--max-gb 10]
"""
import argparse
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from common import COMPETITION, DICOM_DIR, RESULTS_DIR, select_slice_numbers

LOG_PATH = RESULTS_DIR / "download_log.csv"
COUNTS_PATH = RESULTS_DIR / "slice_counts.csv"
_local = threading.local()


def load_token():
    """Carga KAGGLE_API_TOKEN desde el entorno o, en Windows, desde las variables de usuario del
    registro (por si se definió con setx después de abrir la terminal). Nunca se imprime."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                os.environ["KAGGLE_API_TOKEN"] = winreg.QueryValueEx(k, "KAGGLE_API_TOKEN")[0]
                return True
        except OSError:
            pass
    return False


def redact(msg: str) -> str:
    tok = os.environ.get("KAGGLE_API_TOKEN")
    return msg.replace(tok, "***") if tok else msg


def api():
    if not hasattr(_local, "api"):
        load_token()
        from kaggle.api.kaggle_api_extended import KaggleApi
        a = KaggleApi()
        a.authenticate()
        _local.api = a
    return _local.api


class Throttle:
    """Ritmo global de peticiones + pausa global ante HTTP 429."""

    def __init__(self, min_interval, max_throttle_s):
        self.lock = threading.Lock()
        self.min_interval = min_interval
        self.next_t = 0.0
        self.backoff = 60.0
        self.total_pause = 0.0
        self.max_throttle_s = max_throttle_s
        self.n_429 = 0
        self.n_requests = 0
        self.n_ok = 0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next_t)
            self.next_t = t + self.min_interval
            self.n_requests += 1
        time.sleep(max(0.0, t - time.time()))

    def hit_429(self, retry_after=None):
        """Kaggle aplica una cuota por ventana de tiempo y devuelve Retry-After (s);
        si viene, se respeta exactamente (+30 s); si no, backoff 60 -> 600 s."""
        with self.lock:
            self.n_429 += 1
            now = time.time()
            pause = float(retry_after) + 3 if retry_after else self.backoff
            if self.next_t < now + pause * 0.5:  # otro hilo aún no inició la pausa
                self.min_interval = min(self.min_interval * 1.25, 15.0)  # ritmo adaptativo
                self.next_t = now + pause
                self.total_pause += pause
                print(f"  HTTP 429 ({time.strftime('%H:%M:%S')}): pausa global de {pause / 60:.1f} min "
                      f"(Retry-After={retry_after}; acumulado {self.total_pause / 60:.1f} min; "
                      f"peticiones={self.n_requests}, exitosas={self.n_ok}, nuevo intervalo={self.min_interval:.1f}s)", flush=True)
                if not retry_after:
                    self.backoff = min(self.backoff * 2, 600.0)
            if self.total_pause > self.max_throttle_s:
                raise SystemExit("Rate-limit persistente: se detiene la descarga (reanudable).")

    def ok(self):
        with self.lock:
            self.n_ok += 1
            self.backoff = max(60.0, self.backoff * 0.75)


THR: Throttle = None
STOP = threading.Event()


def _normalize(path: Path):
    """Si Kaggle entregó el archivo comprimido en zip, se extrae el .dcm."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"PK\x03\x04":
        with zipfile.ZipFile(path) as z:
            data = z.read(z.namelist()[0])
        path.write_bytes(data)


def fetch(uid: str, n: int, keep: bool):
    """Descarga train_images/uid/n.dcm. True si existe, False si 404."""
    dst = DICOM_DIR / uid / f"{n}.dcm"
    if dst.exists() and dst.stat().st_size > 0:
        return True
    other_errors = 0
    while not STOP.is_set():
        THR.wait()
        tmp = Path(tempfile.mkdtemp(dir=DICOM_DIR))
        try:
            api().competition_download_file(COMPETITION, f"train_images/{uid}/{n}.dcm",
                                             path=str(tmp), quiet=True, force=True)
            files = [p for p in tmp.iterdir() if p.is_file() and not p.name.endswith("partial")]
            if not files or files[0].stat().st_size == 0:
                partial = [p.read_bytes()[:150] for p in tmp.iterdir() if p.name.endswith("partial")]
                raise RuntimeError(f"descarga vacía; parcial={partial!r}")
            _normalize(files[0])
            if keep:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(files[0]), dst)
            THR.ok()
            return True
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            msg = redact(str(e))
            if "404" in msg or "Not Found" in msg:
                THR.ok()  # respuesta válida (el corte no existe)
                return False
            if "429" in msg or "Too Many" in msg:
                resp = getattr(e, "response", None)
                THR.hit_429(resp.headers.get("retry-after") if resp is not None else None)
                continue
            if "403" in msg or "401" in msg:
                STOP.set()
                raise SystemExit(f"Acceso denegado ({msg[:120]}). ¿Reglas aceptadas / token válido?")
            other_errors += 1
            print(f"  error ({time.strftime('%H:%M:%S')}) {uid}/{n} intento {other_errors}: "
                  f"{type(e).__name__}: {msg[-220:]}", flush=True)
            _local.__dict__.pop("api", None)  # p. ej. conexión caída tras una pausa larga: nueva sesión
            if other_errors >= 5:
                raise RuntimeError(f"{uid}/{n}: {msg[:150]}")
            time.sleep(5 * other_errors)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    raise RuntimeError("detenido")


def count_slices(uid: str) -> int:
    """Búsqueda exponencial + binaria del mayor n con n.dcm existente (asume 1..N contiguo)."""
    if not fetch(uid, 1, keep=False):
        return 0
    lo, hi = 1, 128
    while fetch(uid, hi, keep=False):
        lo, hi = hi, hi * 2
        if hi > 4096:
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fetch(uid, mid, keep=False):
            lo = mid
        else:
            hi = mid
    return lo


_io_lock = threading.Lock()


def process(uid: str, counts: dict):
    t0 = time.time()
    try:
        n = counts.get(uid) or count_slices(uid)
        if n == 0:
            return dict(StudyInstanceUID=uid, n_slices=0, n_downloaded=0, mb=0, status="sin_1.dcm",
                        seconds=round(time.time() - t0, 1), error="")
        with _io_lock:
            counts[uid] = n
            pd.DataFrame(sorted(counts.items()), columns=["StudyInstanceUID", "n_slices"]).to_csv(
                COUNTS_PATH, index=False)
        sel = select_slice_numbers(n)
        ok = sum(fetch(uid, s, keep=True) for s in sel)
        mb = sum(p.stat().st_size for p in (DICOM_DIR / uid).glob("*.dcm")) / 1e6
        return dict(StudyInstanceUID=uid, n_slices=n, n_downloaded=ok, mb=round(mb, 2),
                    status="ok" if ok == len(sel) else "incompleto",
                    seconds=round(time.time() - t0, 1), error="")
    except RuntimeError as e:
        return dict(StudyInstanceUID=uid, n_slices=counts.get(uid, -1), n_downloaded=-1, mb=0,
                    status="error", seconds=round(time.time() - t0, 1), error=redact(str(e))[-250:])


def main():
    global THR
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-interval", type=float, default=1.0, help="s mínimos entre peticiones")
    ap.add_argument("--max-gb", type=float, default=10.0)
    ap.add_argument("--max-throttle-min", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=0, help="solo los primeros N estudios (prueba)")
    args = ap.parse_args()
    THR = Throttle(args.min_interval, args.max_throttle_min * 60)

    splits = pd.read_csv(RESULTS_DIR / "splits.csv")
    order = {"test": 0, "val": 1, "train": 2}
    splits = splits.assign(o=splits["split"].map(order)).sort_values("o", kind="stable")
    uids = splits["StudyInstanceUID"].tolist()[: args.limit or None]

    log = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=["StudyInstanceUID"])
    counts = (pd.read_csv(COUNTS_PATH).set_index("StudyInstanceUID")["n_slices"].to_dict()
              if COUNTS_PATH.exists() else {})
    for r in log.to_dict("records"):  # N conocido de ejecuciones anteriores
        if r.get("n_slices", 0) and r["n_slices"] > 0:
            counts.setdefault(r["StudyInstanceUID"], int(r["n_slices"]))
    rows = {r["StudyInstanceUID"]: r for r in log.to_dict("records") if r.get("status") == "ok"}
    # Estudios con todos sus cortes ya en disco (completados en una ejecución interrumpida)
    for u, n in counts.items():
        if u not in rows and n > 0:
            sel = select_slice_numbers(int(n))
            if all((DICOM_DIR / u / f"{s}.dcm").exists() for s in sel):
                mb = sum(p.stat().st_size for p in (DICOM_DIR / u).glob("*.dcm")) / 1e6
                rows[u] = dict(StudyInstanceUID=u, n_slices=int(n), n_downloaded=len(sel), mb=round(mb, 2),
                               status="ok", seconds=0, error="")
    todo = [u for u in uids if u not in rows]
    print(f"{len(todo)} estudios por descargar ({len(rows)} ya completos); "
          f"workers={args.workers}, intervalo={args.min_interval}s", flush=True)

    t0, done = time.time(), 0
    total_mb = sum(float(r.get("mb", 0) or 0) for r in rows.values())
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, u, counts) for u in todo]
        retried = set()
        try:
            pending = list(futs)
            while pending:
                fut = next(as_completed(pending))
                pending.remove(fut)
                r = fut.result()
                if r["status"] == "error" and r["StudyInstanceUID"] not in retried and not STOP.is_set():
                    retried.add(r["StudyInstanceUID"])  # un único reintento por estudio, al final de la cola
                    print(f"  reintento programado: {r['StudyInstanceUID']} ({r['error'][-120:]})", flush=True)
                    nf = ex.submit(process, r["StudyInstanceUID"], counts)
                    futs.append(nf); pending.append(nf)
                    continue
                rows[r["StudyInstanceUID"]] = r
                done += 1
                total_mb += r["mb"]
                pd.DataFrame(rows.values()).to_csv(LOG_PATH, index=False)
                el = time.time() - t0
                print(f"[{done}/{len(todo)}] {r['status']} N={r['n_slices']} {r['seconds']}s | "
                      f"{total_mb / 1e3:.2f} GB | {el / 60:.1f} min | ETA {el / done * (len(todo) - done) / 60:.1f} min"
                      f" | req={THR.n_requests} ok={THR.n_ok} 429={THR.n_429}" + (f" | {r['error']}" if r["error"] else ""),
                      flush=True)
                if total_mb / 1e3 > args.max_gb:
                    print("Límite de GB alcanzado; deteniendo.", flush=True)
                    STOP.set()
                    break
        except SystemExit as e:
            print(str(e), flush=True)
            STOP.set()
        finally:
            STOP.set() if STOP.is_set() else None
            for f in futs:
                f.cancel()
    pd.DataFrame(rows.values()).to_csv(LOG_PATH, index=False)
    print("Fin. ok:", sum(r["status"] == "ok" for r in rows.values()), flush=True)


if __name__ == "__main__":
    main()
