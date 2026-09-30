"""Data loading, AAMI label mapping, baseline correction, and subject-wise split (Track A)."""
import numpy as np
import wfdb
from scipy.signal import medfilt

FS = 360
RECORDS = ['100', '101', '103', '106', '108', '109', '111', '112', '114', '115']
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

# Each of these records comes from a different subject.
# (If you ever add 201/202, map BOTH to the same subject ID, e.g. '201': 'S201', '202': 'S201'.)
SUBJECT_ID = {r: 'S' + r for r in RECORDS}

# Original MIT-BIH beat symbols -> AAMI superclass. Everything else (+, [, ], ~ ...) is ignored.
AAMI_MAP = {}
for sym in ['N', 'L', 'R', 'e', 'j']: AAMI_MAP[sym] = 'N'
for sym in ['A', 'a', 'J', 'S']:      AAMI_MAP[sym] = 'S'
for sym in ['V', 'E']:                AAMI_MAP[sym] = 'V'
AAMI_MAP['F'] = 'F'
for sym in ['/', 'f', 'Q']:           AAMI_MAP[sym] = 'Q'

PRE = int(0.100 * FS)    # 36 samples before R-peak
POST = int(0.180 * FS)   # 64 samples after R-peak  -> window = 101 samples

# Suggested split (justify in your report): test set has beats of every kind we can get.
TEST_RECORDS = ['109', '112', '114']


def baseline_correct(sig, fs=FS):
    """Two-stage median filter (200 ms then 600 ms) estimates the baseline; subtract it.
    Replace with your Lab 1 method if you want."""
    k1 = int(0.2 * fs) // 2 * 2 + 1
    k2 = int(0.6 * fs) // 2 * 2 + 1
    baseline = medfilt(medfilt(sig, k1), k2)
    return sig - baseline


def load_record(name):
    """Return list-of-beats data for one record."""
    rec = wfdb.rdrecord(name, pn_dir='mitdb')
    ann = wfdb.rdann(name, 'atr', pn_dir='mitdb')

    # Record 114 has its channels swapped, so find the MLII channel by name.
    ch = rec.sig_name.index('MLII') if 'MLII' in rec.sig_name else 0
    raw = rec.p_signal[:, ch]
    corrected = baseline_correct(raw)

    # keep only beat annotations
    keep = [i for i, s in enumerate(ann.symbol) if s in AAMI_MAP]
    peaks = np.array([ann.sample[i] for i in keep])
    labels = [AAMI_MAP[ann.symbol[i]] for i in keep]

    out = dict(raw=[], corr=[], prev_rr=[], post_rr=[], label=[], record=[], peak=[])
    removed = 0
    for j in range(len(peaks)):
        # first and last beat have no previous / next beat -> discard
        if j == 0 or j == len(peaks) - 1:
            removed += 1
            continue
        p = peaks[j]
        if p - PRE < 0 or p + POST + 1 > len(raw):   # window falls off the signal
            removed += 1
            continue
        out['raw'].append(raw[p - PRE:p + POST + 1])
        out['corr'].append(corrected[p - PRE:p + POST + 1])
        out['prev_rr'].append((p - peaks[j - 1]) / FS)
        out['post_rr'].append((peaks[j + 1] - p) / FS)
        out['label'].append(labels[j])
        out['record'].append(name)
        out['peak'].append(p)
    return out, removed


def load_all(records=RECORDS):
    """Load all records and concatenate. Returns dict of arrays + total removed count."""
    data = dict(raw=[], corr=[], prev_rr=[], post_rr=[], label=[], record=[], peak=[])
    total_removed = 0
    for r in records:
        d, removed = load_record(r)
        total_removed += removed
        for k in data:
            data[k].extend(d[k])
        print(f'Record {r}: {len(d["label"])} beats kept, {removed} removed')
    data = {k: np.array(v) for k, v in data.items()}
    data['group'] = np.array([SUBJECT_ID[r] for r in data['record']])
    return data, total_removed


def subject_split(data, test_records=TEST_RECORDS):
    """Split by SUBJECT, never by individual beat."""
    test_subjects = {SUBJECT_ID[r] for r in test_records}
    test_mask = np.isin(data['group'], list(test_subjects))
    train_subjects = sorted(set(data['group'][~test_mask]))
    assert not (set(train_subjects) & test_subjects), 'Subject overlap!'
    print('Train subjects:', train_subjects)
    print('Test subjects :', sorted(test_subjects))
    return ~test_mask, test_mask


if __name__ == '__main__':
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data, removed = load_all()
    print('Total samples removed (first/last beats):', removed)

    counts = pd.Series(data['label']).value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
    print(counts)
    counts.plot(kind='bar', title='Class frequencies (AAMI)')
    plt.ylabel('Number of beats'); plt.tight_layout()
    plt.savefig('results/figures/class_frequencies.png', dpi=150)

    train_mask, test_mask = subject_split(data)
    print('Train beats:', train_mask.sum(), ' Test beats:', test_mask.sum())
    print('Train class counts:\n', pd.Series(data['label'][train_mask]).value_counts())
    print('Test class counts:\n', pd.Series(data['label'][test_mask]).value_counts())
    