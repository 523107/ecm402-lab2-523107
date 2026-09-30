"""Handcrafted feature extraction for ECG beats (Track A)."""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from data import load_all, subject_split, FS, CLASS_NAMES

FEATURE_NAMES = ['prev_rr', 'post_rr', 'qrs_width', 'mean', 'std',
                 'skew', 'kurtosis', 'dom_freq', 'spec_entropy']


def qrs_width(seg, fs=FS):
    """Duration (s) for which |baseline-corrected beat| exceeds 50% of its max |amplitude|.
    Educational proxy only: affected by P/T waves and noisy peaks."""
    a = np.abs(seg)
    return np.sum(a > 0.5 * a.max()) / fs


def spectral_features(seg, fs=FS):
    """Coarse dominant frequency and spectral entropy from the FFT of the beat window.
    Resolution is only about fs/len(seg) = 3.6 Hz, so these are coarse features."""
    x = seg - seg.mean()                       # remove DC so it doesn't dominate
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    dom = freqs[1:][np.argmax(power[1:])]      # skip DC bin
    p = power / (power.sum() + 1e-12)
    ent = -np.sum(p * np.log2(p + 1e-12))
    return dom, ent


def extract_features(data):
    rows = []
    for seg, prr, nrr in zip(data['corr'], data['prev_rr'], data['post_rr']):
        dom, ent = spectral_features(seg)
        rows.append([prr, nrr, qrs_width(seg), seg.mean(), seg.std(),
                     skew(seg), kurtosis(seg), dom, ent])
    return np.array(rows)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    data, removed = load_all()
    X = extract_features(data)
    y = data['label']
    print(f'N = {X.shape[0]}, d = {X.shape[1]}, K = {len(set(y))}, removed = {removed}')

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label'] = y
    df['record'] = data['record']
    df['group'] = data['group']
    df.to_csv('results/tables/features.csv', index=False)

    # Box plots of 6 features grouped by class
    plot_feats = ['prev_rr', 'post_rr', 'qrs_width', 'std', 'kurtosis', 'spec_entropy']
    order = [c for c in CLASS_NAMES if c in set(y)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, f in zip(axes.ravel(), plot_feats):
        sns.boxplot(data=df, x='label', y=f, order=order, ax=ax)
        ax.set_title(f)
    plt.tight_layout()
    plt.savefig('results/figures/feature_boxplots.png', dpi=150)
    print(df.groupby('label')[FEATURE_NAMES].median().round(3))