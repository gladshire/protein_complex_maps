import itertools
from sklearn.metrics import precision_recall_curve, average_precision_score

def prep_reference(ref, detected):
    out = [set(c) & set(detected) for c in ref]
    return [c for c in out if len(c) >= 2]

def overlap_score(a, b):
    i = len(a & b)
    return float(i * i) / (len(a) * len(b))

def complex_level(pred, ref, thr=0.25):
    pred = [set(p) for p in pred if len(p) >= 2]
    if not pred or not ref:
        return 0.0, 0.0
    pred_hit = [any(overlap_score(p, r) >= thr for r in ref) for p in pred]
    ref_hit  = [any(overlap_score(p, r) >= thr for p in pred) for r in ref]
    return sum(pred_hit) / float(len(pred)), sum(ref_hit) / float(len(ref))

def pair_set(clusters):
    return set(frozenset(p) for c in clusters
               for p in itertools.combinations(sorted(set(c)), 2))

def pairwise(pred, ref):
    P, R = pair_set(pred), pair_set(ref)
    tp = len(P & R)
    return (tp / float(len(P)) if P else 0.0), tp / float(len(R))

def pair_curve(pair_scores, ref):
    pos = pair_set(ref)
    prots = sorted(set(p for c in ref for p in c))
    universe = [frozenset(p) for p in itertools.combinations(prots, 2)]
    y = [1 if p in pos else 0 for p in universe]
    s = [pair_scores.get(p, 0.0) for p in universe]
    return precision_recall_curve(y, s), average_precision_score(y, s)

def scores_from_clusters(clusters, cluster_scores):
    out = {}
    for c, sc in zip(clusters, cluster_scores):
        for p in pair_set([c]):
            out[p] = max(out.get(p, 0.0), sc)
    return out

def prep_reference_named(ref_dict, detected):
    names, sets = [], []
    for k in sorted(ref_dict):
        s = set(ref_dict[k]) & set(detected)
        if len(s) >= 2:
            names.append(k)
            sets.append(s)
    return names, sets

def cluster_stats(cluster, ref, names):
    c = set(cluster)
    out = dict(match="none", size=len(c), shared=0, precision=0.0, recall=0.0, f1=0.0, pair_precision=0.0)
    if not c or not ref:
        return out
    j = max(range(len(ref)), key=lambda k: overlap_score(c, ref[k]))
    shared = len(c & ref[j])
    if shared == 0:
        return out
    p, r = shared / float(len(c)), shared / float(len(ref[j]))
    pairs = pair_set([c])
    out.update(match=str(names[j]), shared=shared, precision=p, recall=r, f1=2 * p * r / (p + r),
               pair_precision=(len(pairs & pair_set(ref)) / float(len(pairs))) if pairs else 0.0)
    return out

def _clean(s, maxlen=24):
    """Printable single-line ASCII only: old matplotlib PDF backends choke on control/odd characters."""
    s = "%s" % (s,)
    out = []
    for ch in s:
        if ch in "\t\r\n":
            out.append(" ")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out).strip()[:maxlen]

def plot_cluster_stats(stats_nmf, stats_mc, savefilename, title, stats_nmf_raw=None, stats_mc_raw=None):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    keys = ["precision", "recall", "f1", "pair_precision"]
    labels = ["precision", "recall", "F1", "pair prec."]
    # Type 42 embedding + the bundled font avoid a font-embedding crash seen in old matplotlib builds
    rc = {"pdf.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]}
    with matplotlib.rc_context(rc):
        x, w = np.arange(len(keys)), 0.2
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.bar(x - 3*w / 2, [stats_nmf[k] for k in keys], w, label="NMF seed (n=%d)" % stats_nmf["size"], color="#b8b8b8")
        ax.bar(x + w / 2, [stats_mc[k] for k in keys], w, label="NMF + MC (n=%d)" % stats_mc["size"], color="#1f77b4")
        if stats_nmf_raw and stats_mc_raw:
            ax.bar(x - w / 2, [stats_nmf_raw[k] for k in keys], w, label="NMF seed (from raw counts) (n=%d)" % stats_nmf_raw["size"], color="#5c5c5c")
            ax.bar(x + 3*w / 2, [stats_mc_raw[k] for k in keys], w, label="NMF + MC (from raw counts) (n=%d)" % stats_mc_raw["size"], color="#19c47a")

        ax.set_ylim(0, 1.3); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title("%s | match: %s / %s" % (_clean(title), _clean(stats_nmf["match"]), _clean(stats_mc["match"])), fontsize=8)
        ax.legend(fontsize=7, loc="upper center", ncol=2)
        fig.tight_layout()
        try:
            fig.savefig(savefilename)
        except Exception as e:
            # never let a plotting problem kill a long run: fall back to PNG next to the intended PDF
            fallback = savefilename.rsplit(".", 1)[0] + ".png"
            print(("WARNING: could not save %s (%s); wrote %s instead" % (savefilename, e, fallback)))
            fig.savefig(fallback, dpi=150)
        plt.close(fig)

def write_stats_table(rows, filename):
    cols = ["match", "size", "shared", "precision", "recall", "f1", "pair_precision"]
    with open(filename, "w") as f:
        f.write("\t".join(["component", "version"] + cols) + "\n")
        for comp, ver, st in rows:
            f.write("\t".join([str(comp), ver] + [("%.3f" % st[c]) if isinstance(st[c], float) else str(st[c]) for c in cols]) + "\n")
