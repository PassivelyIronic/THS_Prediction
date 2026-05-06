#!/usr/bin/env python3
"""
02b_waveform_extraction.py — WERSJA V3
──────────────────────────────────────
Kluczowe zmiany względem V2:
  1. Okno poszukiwań: −4h (zamiast −1h) → omijamy szczyt interwencji
  2. Scoring kandydatów: jakość × bliskość do reference_time
  3. Flat-line guard: szybkie odrzucanie płaskich segmentów
  4. Multi-method R-peak detection: 3 algorytmy, bierzemy najlepszy
  5. RR-interval cleaning: usuwamy fizjologicznie niemożliwe wartości
     przed obliczaniem HRV (nie odrzucamy całego sygnału!)
"""

import sys, logging, warnings
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ─── Stałe ────────────────────────────────────────────────────────────────────

PHYSIONET_DB_PATH = 'mimic3wdb-matched/1.0'

# V3: szukamy 4h wstecz (zamiast 1h) — omijamy szczyt interwencji
SEARCH_HOURS      = 4
SEGMENT_SEC       = 300    # 5 minut sygnału
MIN_VALID_SEC     = 60     # minimum do analizy

# Progi fizjologiczne interwałów RR (ms)
RR_MIN_MS         = 300    # 200 BPM — ekstremalna tachykardia
RR_MAX_MS         = 2000   # 30  BPM — głęboka bradykardia

MIN_VALID_RR      = 5      # minimum RR do obliczeń HRV

# Flat-line detection — sygnał z std < progu → odrzuć natychmiast
FLAT_LINE_STD     = 0.02   # mV / ADU (typowo ≥0.1 dla normalnego EKG)

ECG_LEAD_PRIORITY = ['II', 'MLII', 'I', 'MLI', 'V5', 'V']
CHECKPOINT_EVERY  = 50

# Kolumny wyjściowe — MUSZĄ być spójne z config.ECG_COLS!
ECG_OUT_COLS = ['ECG_RMSSD', 'ECG_SDNN', 'ECG_pNN50', 'ECG_LF_HF']
_NAN_FEATS   = {c: np.nan for c in ECG_OUT_COLS}


# ─── Pomocnicze ──────────────────────────────────────────────────────────────

def _subject_rel_dir(subject_id: int) -> str:
    padded = f'p{subject_id:06d}'
    return f'{padded[:3]}/{padded}'


def _pick_ecg_channel(sig_names: list[str]) -> int:
    upper = [s.upper().strip() for s in sig_names]
    for lead in ECG_LEAD_PRIORITY:
        if lead in upper:
            return upper.index(lead)
    for i, n in enumerate(upper):
        if 'ECG' in n or 'EKG' in n:
            return i
    return -1


def _fetch_master_records(subject_id: int) -> list[str]:
    """Zwraca 'master records' (multi-segment, bez 'n', bez '_')."""
    sub_rel = _subject_rel_dir(subject_id)
    url = (f'https://physionet.org/files/'
           f'{PHYSIONET_DB_PATH}/{sub_rel}/RECORDS')
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            lines = [ln.decode('utf-8').strip() for ln in resp]
        return [ln for ln in lines
                if ln and not ln.endswith('n') and '_' not in ln]
    except Exception:
        return []


def _is_flat(signal: np.ndarray) -> bool:
    """True gdy sygnał jest płaski (odłączona elektroda / pełen szum DC)."""
    if len(signal) == 0:
        return True
    std = np.std(signal)
    return std < FLAT_LINE_STD


# ─── Ekstrakcja segmentu EKG ─────────────────────────────────────────────────

def extract_ecg_segment(
    subject_id: int,
    reference_time: pd.Timestamp,
) -> tuple[np.ndarray | None, float | None]:
    """
    V3 — Adaptacyjna selekcja segmentu.

    Strategia:
    1. Przeszukujemy okno [reference_time − 4h, reference_time).
    2. Dla każdego pasującego segmentu obliczamy score:
         score = signal_quality_proxy × (1 / (1 + hours_to_ref))
       Im bliżej reference_time i im lepsza jakość, tym wyższy score.
    3. Zwracamy najlepszy segment.

    GWARANCJA BRAKU LEAKAGE:
    sampto = próbka odpowiadająca dokładnie reference_time.
    Nigdy nie pobieramy danych po reference_time.
    """
    search_start = reference_time - pd.Timedelta(hours=SEARCH_HOURS)

    master_records = _fetch_master_records(subject_id)
    if not master_records:
        return None, None

    sub_rel = _subject_rel_dir(subject_id)
    candidates: list[tuple[float, np.ndarray, float]] = []  # (score, ecg, fs)

    for m_rec in master_records:
        try:
            hdr = wfdb.rdheader(m_rec, pn_dir=f'{PHYSIONET_DB_PATH}/{sub_rel}')
        except Exception:
            continue

        if hdr.base_datetime is None or not hasattr(hdr, 'seg_name'):
            continue

        current_time = pd.Timestamp(hdr.base_datetime)
        fs = float(hdr.fs)

        for seg_idx, seg_name in enumerate(hdr.seg_name):
            seg_len  = hdr.seg_len[seg_idx]
            seg_dur  = seg_len / fs
            seg_start = current_time
            seg_end   = current_time + pd.Timedelta(seconds=seg_dur)
            current_time = seg_end

            if seg_name == '~' or seg_len == 0:
                continue

            # Overlap z naszym oknem poszukiwań
            ov_start = max(seg_start, search_start)
            ov_end   = min(seg_end,   reference_time)   # ← NIGDY poza reference_time
            overlap_sec = (ov_end - ov_start).total_seconds()

            if overlap_sec < MIN_VALID_SEC:
                continue

            # Wybór kanału EKG
            try:
                seg_hdr = wfdb.rdheader(
                    seg_name, pn_dir=f'{PHYSIONET_DB_PATH}/{sub_rel}'
                )
            except Exception:
                continue

            ch_idx = _pick_ecg_channel(seg_hdr.sig_name)
            if ch_idx < 0:
                continue

            # Zakres próbek: bierzemy do 5 minut, kończąc NA ov_end
            grab_end   = ov_end
            grab_start = max(ov_start,
                             grab_end - pd.Timedelta(seconds=SEGMENT_SEC))

            samp_from = max(0, int((grab_start - seg_start).total_seconds() * fs))
            samp_to   = min(seg_len,
                            int((grab_end   - seg_start).total_seconds() * fs))

            if samp_to - samp_from < int(fs * MIN_VALID_SEC):
                continue

            # Pobierz sygnał
            try:
                sig_raw, _ = wfdb.rdsamp(
                    seg_name,
                    pn_dir=f'{PHYSIONET_DB_PATH}/{sub_rel}',
                    channels=[ch_idx],
                    sampfrom=samp_from,
                    sampto=samp_to,
                )
                ecg = sig_raw.flatten().astype(float)
                ecg = ecg[np.isfinite(ecg)]
            except Exception:
                continue

            if len(ecg) < int(fs * MIN_VALID_SEC):
                continue

            # ── Flat-line guard ───────────────────────────────────────────────
            if _is_flat(ecg):
                continue   # Odrzuć — odłączona elektroda lub artefakt DC

            # ── Score: jakość × bliskość do reference_time ───────────────────
            # signal_quality_proxy: IQR znormalizowane (większy = więcej dynamiki EKG)
            iqr = float(np.percentile(ecg, 75) - np.percentile(ecg, 25))
            hours_to_ref = (reference_time - grab_end).total_seconds() / 3600
            score = iqr / (1.0 + hours_to_ref)

            candidates.append((score, ecg, fs))

    if not candidates:
        return None, None

    # Najlepszy segment wg score
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_ecg, best_fs = candidates[0]
    return best_ecg, best_fs


# ─── Obliczanie HRV ──────────────────────────────────────────────────────────

def _detect_r_peaks_robust(signal: np.ndarray, fs_int: int) -> np.ndarray | None:
    """
    Próbuje 3 metody detekcji R-peaków, zwraca zestaw z największą
    liczbą fizjologicznie poprawnych interwałów RR.
    """
    methods = ['pantompkins1985', 'hamiltonze', 'engzee']
    best_peaks = None
    best_valid_count = 0

    for method in methods:
        try:
            clean = nk.ecg_clean(signal, sampling_rate=fs_int, method=method)
            _, info = nk.ecg_peaks(clean, sampling_rate=fs_int, method=method)
            peaks = info['ECG_R_Peaks']

            # Policz tylko fizjologicznie sensowne interwały RR
            if len(peaks) > 1:
                rr = np.diff(peaks) / fs_int * 1000  # ms
                valid = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]
                if len(valid) > best_valid_count:
                    best_peaks = peaks
                    best_valid_count = len(valid)
        except Exception:
            continue

    return best_peaks if best_valid_count >= MIN_VALID_RR else None


def _clean_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Zwraca NN-intervals [ms] po usunięciu wartości poza zakresem
    fizjologicznym i ekstremalnych outlierów (IQR × 2.5).
    """
    rr = np.diff(peaks) / fs * 1000  # ms
    rr = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]
    if len(rr) < 3:
        return rr
    q1, q3 = np.percentile(rr, 25), np.percentile(rr, 75)
    iqr = q3 - q1
    return rr[(rr >= q1 - 2.5 * iqr) & (rr <= q3 + 2.5 * iqr)]


def compute_hrv_features(signal: np.ndarray | None, fs: float | None) -> dict:
    """
    V3 — Oblicza cechy HRV na podstawie oczyszczonych interwałów NN.

    Zwracane cechy:
      ECG_RMSSD  [ms]     — Root Mean Square of Successive Differences
      ECG_SDNN   [ms]     — Standard Deviation of NN intervals
      ECG_pNN50  [%]      — Frakcja NN50 / n_intervals × 100
      ECG_LF_HF  [ratio]  — LF/HF power ratio (wymaga ≥120 s sygnału)

    KLUCZOWA RÓŻNICA względem V2:
    Czyścimy interwały RR przed obliczeniami (nie odrzucamy całego sygnału).
    Sygnał z artefaktami → mniej ważnych interwałów → HRV z tego co zostało.
    """
    if signal is None or fs is None:
        return _NAN_FEATS

    fs_int = int(round(fs))

    if len(signal) < int(fs * MIN_VALID_SEC):
        return _NAN_FEATS

    if _is_flat(signal):
        return _NAN_FEATS

    # Detekcja R-peaków (multi-method)
    peaks = _detect_r_peaks_robust(signal, fs_int)
    if peaks is None:
        return _NAN_FEATS

    # Oczyszczone interwały NN
    nn_ms = _clean_rr_intervals(peaks, fs_int)
    if len(nn_ms) < MIN_VALID_RR:
        return _NAN_FEATS

    feats = dict(_NAN_FEATS)  # kopia, żeby nie mutować globalu

    # ── HRV domena czasu (zawsze) ────────────────────────────────────────────
    diff_nn = np.diff(nn_ms)

    feats['ECG_RMSSD'] = float(np.sqrt(np.mean(diff_nn ** 2)))
    feats['ECG_SDNN']  = float(np.std(nn_ms, ddof=1))

    nn50 = np.sum(np.abs(diff_nn) > 50)
    feats['ECG_pNN50'] = float(nn50 / len(diff_nn) * 100) if len(diff_nn) > 0 else np.nan

    # Odrzuć fizjologicznie absurdalne wartości
    if feats['ECG_RMSSD'] > 2000 or feats['ECG_RMSSD'] < 0:
        feats['ECG_RMSSD'] = np.nan
    if feats['ECG_SDNN']  > 2000 or feats['ECG_SDNN']  < 0:
        feats['ECG_SDNN']  = np.nan

    # ── HRV domena częstotliwości
    if len(nn_ms) >= 20:
        try:
            hrv_freq = nk.hrv_frequency(
                peaks, sampling_rate=fs_int,
                vlf_min=0.003, lf_min=0.04,
                hf_min=0.15,  hf_max=0.40,
            )
            lf = hrv_freq.get('HRV_LF',  pd.Series([np.nan])).iloc[0]
            hf = hrv_freq.get('HRV_HF',  pd.Series([np.nan])).iloc[0]
            if float(hf) > 1e-9:            # unikamy dzielenia przez zero
                feats['ECG_LF_HF'] = float(lf) / float(hf)
        except Exception:
            pass

    return feats


# ─── main() ──────────────────────────────────────────────────────────────────

def main():
    out_path = config.PROCESSED_DIR / 'waveform_features.csv'
    chk_path = config.PROCESSED_DIR / '_waveform_checkpoint.csv'

    # Zbierz zadania ze wszystkich okien czasowych
    tasks_list = []
    for T in config.TIME_WINDOWS:
        fpath = config.WINDOWS_DIR / f'features_T{T}h_VS_RB_BG.csv'
        if not fpath.exists():
            log.warning(f'Brak {fpath.name} — pomijam T={T}h')
            continue
        df = pd.read_csv(
            fpath,
            usecols=['HADM_ID', 'SUBJECT_ID', 'TIME_WINDOW', 'REFERENCE_TIME'],
        )
        df['REFERENCE_TIME'] = pd.to_datetime(df['REFERENCE_TIME'])
        tasks_list.append(df)

    if not tasks_list:
        log.error('Brak plików wejściowych. Uruchom 02_preprocessing.ipynb.')
        return

    all_tasks = (
        pd.concat(tasks_list, ignore_index=True)
        .drop_duplicates(subset=['HADM_ID', 'TIME_WINDOW', 'REFERENCE_TIME'])
        .reset_index(drop=True)
    )
    log.info(f'Zadań do przetworzenia: {len(all_tasks):,}')

    # Wczytaj checkpoint
    if chk_path.exists():
        done_df = pd.read_csv(chk_path)
        done_df['REFERENCE_TIME'] = pd.to_datetime(done_df['REFERENCE_TIME'])
        done_keys = set(
            zip(done_df['HADM_ID'].astype(int),
                done_df['TIME_WINDOW'].astype(float),
                done_df['REFERENCE_TIME'].astype(str))
        )
        log.info(f'Checkpoint: {len(done_df):,} gotowych.')
    else:
        done_df, done_keys = pd.DataFrame(), set()

    results = []

    for _, row in tqdm(all_tasks.iterrows(), total=len(all_tasks), desc='HRV'):
        key = (
            int(row['HADM_ID']),
            float(row['TIME_WINDOW']),
            str(row['REFERENCE_TIME']),
        )
        if key in done_keys:
            continue

        sig, fs = extract_ecg_segment(
            subject_id     = int(row['SUBJECT_ID']),
            reference_time = row['REFERENCE_TIME'],
        )
        feats = compute_hrv_features(sig, fs)

        results.append({
            'HADM_ID':        int(row['HADM_ID']),
            'TIME_WINDOW':    float(row['TIME_WINDOW']),
            'REFERENCE_TIME': row['REFERENCE_TIME'],
            **feats,
        })

        if len(results) % CHECKPOINT_EVERY == 0:
            batch = pd.DataFrame(results)
            pd.concat([done_df, batch], ignore_index=True).to_csv(chk_path, index=False)

    final_df = pd.concat([done_df, pd.DataFrame(results)], ignore_index=True)
    final_df.to_csv(out_path, index=False)

    # Podsumowanie diagnostyczne
    n_total = len(final_df)
    for col in ECG_OUT_COLS:
        nan_pct = final_df[col].isna().mean() * 100
        log.info(f'  {col:15s}: {nan_pct:5.1f}% NaN ({n_total - int(final_df[col].isna().sum()):,} rekordów z wartością)')

    if chk_path.exists():
        chk_path.unlink()

    log.info(f'✅ → {out_path.name}')


if __name__ == '__main__':
    main()