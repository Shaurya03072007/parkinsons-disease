import librosa
import numpy as np

def extract_features(audio_path):
    """
    Extracts 22 features compatible with the UCI Parkinsons dataset.
    
    CRITICAL CHANGE:
    We now attempt to estimate complex features (RPDE, DFA, PPE) from basic acoustic features
    instead of using static placeholders.
    
    IMPORTANT: We calibrate these estimations to match the distribution of the UCI dataset
    so that the model trained on UCI data can still work reasonably well on new inference data.
    """
    try:
        y, sr = librosa.load(audio_path)
        
        # 1. Frequency (Fo, Fhi, Flo)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]
        
        if len(f0) == 0:
            return np.zeros(22)
            
        fo_mean = np.mean(f0)
        fo_max = np.max(f0)
        fo_min = np.min(f0)
        
        # 2. Jitter (Frequency perturbation)
        # Avg absolute difference between consecutive periods / avg period
        jitter_abs = np.mean(np.abs(np.diff(f0))) 
        jitter_percent = jitter_abs / fo_mean
        
        # RAP: Relative Average Perturbation (3-point)
        rap = np.mean(np.abs(np.diff(f0, n=2))) / fo_mean
        
        # PPQ: Period Perturbation Quotient (5-point)
        ppq = np.mean(np.abs(np.diff(f0, n=4))) / fo_mean
        
        ddp = 3 * rap
        
        # 3. Shimmer (Amplitude perturbation)
        # Calculate RMS amplitude per frame
        rmse = librosa.feature.rms(y=y)[0]
        rmse = rmse[rmse > 0.001] # Filter silence for better stats
        
        if len(rmse) == 0:
             rmse = np.array([1e-6])

        # Shimmer in dB
        shimmer_db = np.mean(np.abs(np.diff(20 * np.log10(rmse + 1e-6))))
        
        # Shimmer: APQ3 (3-point amplitude perturbation)
        shimmer_apq3 = np.mean(np.abs(np.diff(rmse, n=3))) / np.mean(rmse)
        
        # Shimmer: APQ5 (5-point)
        shimmer_apq5 = np.mean(np.abs(np.diff(rmse, n=5))) / np.mean(rmse)
        
        # Shimmer: APQ (11-point generally, here using simple diff mean)
        shimmer_apq = np.mean(np.abs(np.diff(rmse))) / np.mean(rmse)
        
        shimmer_dda = 3 * shimmer_apq3
        
        # 4. Noise
        # Harmonics-to-Noise Ratio (HNR)
        y_harm, y_perc = librosa.effects.hpss(y)
        hnr = np.mean(y_harm**2) / (np.mean(y_perc**2) + 1e-6)
        nhr = 1 / (hnr + 1e-6)
        
        # 5. Non-linear / Complex Features (Approximations CALIBRATED to UCI Stats)
        
        # RPDE (Recurrence Period Density Entropy)
        # UCI Range: 0.2 (Healthy) to 0.7 (Severe)
        # Estimator: Base 0.3 + scales with raw NHR
        # If NHR is low (clean), RPDE -> 0.3. If NHR high, RPDE -> 0.7
        rpde = 0.3 + (nhr * 2.0)
        rpde = np.clip(rpde, 0.2, 0.9)
        
        # DFA (Detrended Fluctuation Analysis)
        # UCI Range: 0.5 to 0.8
        # Estimator: Base 0.6 + scales with jitter
        dfa = 0.6 + (jitter_percent * 5.0)
        dfa = np.clip(dfa, 0.5, 0.9)
        
        # Spread1: Nonlinear measure of fundamental frequency variation
        # UCI Range: -7.0 (Healthy) to -2.0 (Severe)
        # Estimator: Base -6.0 + scales with Jitter log
        # High jitter -> less negative spread1
        spread1 = -6.0 + (jitter_percent * 100.0)
        spread1 = np.clip(spread1, -8.0, -2.0)
        
        # Spread2
        # UCI Range: 0.1 to 0.4
        spread2 = 0.15 + (shimmer_db / 20.0)
        spread2 = np.clip(spread2, 0.0, 0.5)
        
        # D2: Correlation dimension
        # UCI Range: 1.5 to 3.0
        d2 = 2.0 + (nhr * 10.0)
        d2 = np.clip(d2, 1.0, 3.5)
        
        # PPE: Pitch Period Entropy
        # UCI Range: 0.1 to 0.4
        ppe = 0.15 + (jitter_percent * 10.0)
        ppe = np.clip(ppe, 0.0, 0.5)

        features = np.array([
            fo_mean, fo_max, fo_min, 
            jitter_percent, jitter_abs, rap, ppq, ddp,
            shimmer_apq, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq, shimmer_dda,
            nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ])
        
        # Handle any NaNs
        features = np.nan_to_num(features)
        
        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(22)
