import pandas as pd
import numpy as np

wf = pd.read_csv('processed/waveform_features.csv')

# Usuń ECG_LF_HF — 100% NaN, brak wartości informacyjnej
wf = wf.drop(columns=['ECG_LF_HF'], errors='ignore')

# Filtr fizjologiczny: RMSSD=0 to flat-line, RMSSD>300 to luźna elektroda
# Zakres fizjologiczny dla populacji ICU: 2–300 ms
mask_bad = (wf['ECG_RMSSD'].notna()) & (
    (wf['ECG_RMSSD'] <= 0) | (wf['ECG_RMSSD'] > 300)
)
print(f"Odrzucam {mask_bad.sum()} artefaktów → zamieniam na NaN")
for col in ['ECG_RMSSD', 'ECG_SDNN', 'ECG_pNN50']:
    wf.loc[mask_bad, col] = np.nan

# Zapisz nadpisując
wf.to_csv('processed/waveform_features.csv', index=False)

# Podsumowanie
for col in ['ECG_RMSSD', 'ECG_SDNN', 'ECG_pNN50']:
    nn = wf[col].notna().sum()
    print(f"  {col}: {nn} clean ({nn/len(wf)*100:.1f}%)")