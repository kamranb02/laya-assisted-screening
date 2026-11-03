"""Stage 5 - publication-quality figures (matplotlib only, colour-blind-safe Okabe-Ito palette)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analyze import METHOD_LABELS, SCALE_LABELS, STORAGE_LABELS  # noqa: E402
from .lexicon import GROUPS  # noqa: E402

PAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#7F7F7F"]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 100,
                     "savefig.bbox": "tight", "legend.frameon": False})


def cfg_thr(cfg):
    return float(cfg["decision"]["raw_include_threshold"])


class Figs:
    def __init__(self, cfg, paths, log):
        self.cfg, self.paths, self.log = cfg, paths, log
        self.fmts = cfg["analysis"]["formats"]
        self.dpi = int(cfg["analysis"]["dpi"])
        self.made = []

    def save(self, fig, name):
        for f in self.fmts:
            fig.savefig(self.paths.figures / f"{name}.{f}", dpi=self.dpi)
        plt.close(fig)
        self.made.append(name)

    def pretty(self, s):
        return str(s).replace("_", " ")

    # ------------------------------------------------------------- primitives
    def hbar(self, s, name, title, xlabel="Number of papers", color=PAL[0], labelmap=None):
        if s is None or len(s) == 0:
            return
        s = s.sort_values()
        fig, ax = plt.subplots(figsize=(8, max(3, .38 * len(s) + 1.2)))
        labs = [(labelmap or {}).get(i, self.pretty(i)) if not isinstance(i, str) or len(str(i)) < 60 else str(i)[:57] + "…"
                for i in s.index]
        ax.barh(labs, s.values, color=color)
        for i, v in enumerate(s.values):
            ax.text(v, i, f" {int(v)}", va="center", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        self.save(fig, name)

    def heat(self, m: pd.DataFrame, name, title, cmap="YlGnBu", xlabel="", ylabel="", fmt="{:.0f}", figsize=None,
             xmap=None, ymap=None):
        if m is None or m.empty or float(np.nansum(m.values.astype(float))) == 0:
            return
        fig, ax = plt.subplots(figsize=figsize or (max(6, .8 * m.shape[1] + 3), max(3.5, .48 * m.shape[0] + 2)))
        v = m.values.astype(float)
        im = ax.imshow(v, cmap=cmap, aspect="auto")
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels([(xmap or {}).get(c, self.pretty(c)) for c in m.columns], rotation=45, ha="right")
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels([(ymap or {}).get(c, self.pretty(c)) for c in m.index])
        ax.grid(False)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if not np.isnan(v[i, j]):
                    r_, g_, b_, _ = im.cmap(im.norm(v[i, j]))
                    lum = .299 * r_ + .587 * g_ + .114 * b_
                    ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=8,
                            color="white" if lum < .5 else "black")
        fig.colorbar(im, ax=ax, shrink=.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.save(fig, name)

    def stacked_year(self, ct: pd.DataFrame, name, title, labelmap=None):
        if ct is None or ct.empty:
            return
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        bottom = np.zeros(len(ct))
        for i, c in enumerate(ct.columns):
            ax.bar(ct.index.astype(str), ct[c].values, bottom=bottom, label=(labelmap or {}).get(c, c), color=PAL[i % len(PAL)])
            bottom += ct[c].values
        for x, t in zip(range(len(ct)), bottom):
            ax.text(x, t, int(t), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Included papers")
        ax.set_title(title, loc="left")
        ax.legend(ncol=2, fontsize=8, loc="upper left")
        ax.grid(axis="x", visible=False)
        self.save(fig, name)

    # ---------------------------------------------------------------- figures
    def prisma(self, P):
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)

        def box(x, y, w, h, txt, fc="#EAF2FA", fs=9):
            ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#333", lw=1.2))
            ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

        def arrow(x1, y1, x2, y2):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
        box(2.0, 10.6, 5.2, 1.1, f"Records identified (query hits, with overlap)\nn = {P['query_hits_total_with_overlap']:,}")
        box(2.0, 8.9, 5.2, 1.1, f"Unique OpenAlex records\nn = {P['unique_records']:,}")
        box(7.4, 7.5, 2.5, 1.0, f"Duplicates removed\nn = {P['duplicates_removed']:,}", "#FBEFD9")
        box(2.0, 7.3, 4.4, 1.1, f"Records after de-duplication\nn = {P['unique_after_dedup']:,}")
        reasons = "\n".join(f"{k}: {v:,}" for k, v in P["auto_excluded_reasons"].items()) or "-"
        nr = max(1, len(P["auto_excluded_reasons"]))
        box(7.0, 5.0, 2.9, 0.9 + 0.32 * nr, f"Auto-excluded (rules)\nn = {P['auto_excluded']:,}\n{reasons}", "#FBEFD9", fs=8)
        box(2.0, 5.5, 4.4, 1.1, f"Screened by Laya (title/abstract)\nn = {P['screened_by_laya']:,}"
                                f"\npending n = {P['pending_screening']:,}")
        box(7.0, 3.0, 2.9, 1.3, f"Excluded by model\n(score < {P['threshold']})\nn = {P['model_excluded']:,}"
                                f"\nhuman-removed from included: {P['human_removed']:,}", "#FBEFD9", fs=8)
        box(2.0, 3.3, 4.4, 1.3, f"Included in UHS corpus (final)\n({'validated ' if P['validated'] else 'NOT VALIDATED '}{P['scorer']} >= {P['threshold']})"
                                f"\nn = {P['included']:,}  [model {P['model_included']:,}; human-added {P['human_added']:,}; "
                                f"human-removed {P['human_removed']:,}]", "#DDF0E4", fs=8)
        box(2.0, 1.2, 4.4, 1.5, f"of which high-confidence n = {P['included_high_confidence']:,}\nborderline n = {P['included_borderline']:,}\n"
                                f"title-only n = {P['included_title_only']:,}\nmanual review queue n = {P['review_queue_size']:,}", "#F3F3F3")
        arrow(4.6, 10.6, 4.6, 10.0); arrow(4.6, 8.9, 4.6, 8.4); arrow(6.4, 7.85, 7.4, 8.0)
        arrow(4.2, 7.3, 4.2, 6.6); arrow(4.4, 7.0, 7.0, 6.0); arrow(4.2, 5.5, 4.2, 4.6); arrow(4.4, 5.2, 7.0, 4.0); arrow(4.2, 3.3, 4.2, 2.7)
        ax.set_title("Record flow (PRISMA-style)", loc="left")
        self.save(fig, "fig01_prisma_flow")

    def yearly(self, Y):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        yrs = Y.publication_year.astype(str) + np.where(Y.partial_year, "*", "")
        ax.bar(yrs, Y.records - Y.auto_excluded, color="#C9D6E3", label="Eligible records")
        ax.bar(yrs, Y.included, color=PAL[0], label="Included (UHS)")
        top = float((Y.records - Y.auto_excluded).max())
        for x, v in enumerate(Y.included):
            ax.text(x, v + top * 0.015, int(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2 = ax.twinx()
        ax2.plot(yrs, Y.included_per_month, color=PAL[3], marker="o", label="Included per month")
        for x, v in enumerate(Y.included_per_month):
            ax2.annotate(f"{v:.0f}/mo", (x, v), textcoords="offset points", xytext=(0, 9), ha="center", color=PAL[3], fontsize=8)
        ax2.set_ylabel("Included papers per month")
        ax2.set_ylim(0, float(Y.included_per_month.max()) * 1.35)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax.set_ylabel("Papers per (partial) year")
        ax.set_title("UHS publications  (* = partial year; compare the per-month line)", loc="left")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
        ax.grid(axis="x", visible=False)
        self.save(fig, "fig02_publications_per_year")

    def monthly(self, Mo):
        if Mo is None or len(Mo) == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 4.4))
        x = np.arange(len(Mo))
        ax.bar(x, Mo.records - Mo.auto_excluded, color="#C9D6E3", label="Eligible records")
        ax.bar(x, Mo.included, color=PAL[0], label="Included (UHS)")
        ax.set_xticks(x)
        ax.set_xticklabels(Mo.pub_month, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Papers per month")
        ax.set_title("Monthly UHS publications (recent months are under-indexed in OpenAlex)", loc="left")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        self.save(fig, "fig24_monthly")

    def topic_shift(self, S):
        if S is None or len(S) == 0:
            return
        cols = [c for c in S.columns if c.startswith("pct_")]
        d = S.reindex(S.diff_pp.abs().sort_values(ascending=False).index).head(14).sort_values("diff_pp")
        fig, ax = plt.subplots(figsize=(8, max(3.5, .38 * len(d) + 1.4)))
        colors = [PAL[2] if v > 0 else PAL[3] for v in d.diff_pp]
        ax.barh([t.replace("_", " ") for t in d.topic], d.diff_pp, color=colors)
        for i, (v, p) in enumerate(zip(d.diff_pp, d.p_adj_bonferroni)):
            ax.text(v, i, f" {v:+.1f}{'*' if p < 0.05 else ''}", va="center", fontsize=8)
        ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel(f"Change in share of papers, {cols[0][4:]} -> {cols[-1][4:]} (percentage points; * = Bonferroni p<0.05)")
        ax.set_title("Topic shift between the first and last year", loc="left")
        ax.grid(axis="y", visible=False)
        self.save(fig, "fig25_topic_shift")

    def relevance_hist(self, df, thr, band):
        s = df.score_final.dropna()
        if s.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(s, bins=40, color=PAL[0], alpha=.85)
        ax.axvspan(band[0], band[1], color=PAL[1], alpha=.15, label=f"Review band {band[0]}–{band[1]}")
        ax.axvline(thr, color=PAL[3], lw=2, label=f"Include threshold {thr}")
        ax.set_yscale("log")
        ax.set_xlabel("Decision score (P(paper is about UHS))")
        ax.set_ylabel("Papers (log)")
        ax.set_title("Distribution of Laya relevance scores", loc="left")
        ax.legend()
        self.save(fig, "fig03_relevance_distribution")

    def threshold(self, ts, thr):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(ts.threshold, ts.included_by_model, marker="o", color=PAL[0])
        ax.axvline(thr, color=PAL[3], ls="--", label=f"current = {thr}")
        ax.set_xlabel("Inclusion threshold on the decision score")
        ax.set_ylabel("Included papers (model decisions)")
        ax2 = ax.twinx()
        ax2.plot(ts.threshold, ts["rule_strong_or_likely_among_included_%"], color=PAL[2], marker="s", ls=":",
                 label="% confirmed by keyword rules")
        ax2.set_ylabel("% of included also rule-positive"); ax2.grid(False); ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left")
        ax.set_title("Sensitivity of corpus size to the inclusion threshold", loc="left")
        self.save(fig, "fig04_threshold_sensitivity")

    def confidence(self, inc):
        cols = [c for c in ("laya_storage_type_conf", "laya_method_conf", "laya_scale_conf") if c in inc and inc[c].notna().any()]
        if not cols:
            return
        fig, ax = plt.subplots(figsize=(8, 4.3))
        for i, c in enumerate(cols):
            ax.hist(inc[c].dropna(), bins=25, alpha=.55, label=c.replace("laya_", "").replace("_conf", ""), color=PAL[i])
        ax.axvline(self.cfg["screening"]["min_choice_conf"], color="k", ls="--", lw=1)
        ax.set_xlabel("Probability of the chosen label"); ax.set_ylabel("Papers")
        ax.set_title("Laya classification confidence (included papers)", loc="left")
        ax.legend()
        self.save(fig, "fig21_laya_confidence")

    def performance(self, perf: pd.DataFrame):
        if perf is None or perf.empty:
            return
        perf = perf.copy()
        perf["timestamp"] = pd.to_datetime(perf["timestamp"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for st, g in perf.groupby("stage"):
            axes[0].plot(g.timestamp, g.papers_per_sec.rolling(10, min_periods=1).mean(), label=f"stage {st}")
        axes[0].set_ylabel("Papers / second (10-batch mean)"); axes[0].legend(); axes[0].tick_params(axis="x", rotation=30)
        axes[0].set_title("Laya throughput", loc="left")
        tot = perf.groupby("stage").agg(n=("n_papers", "sum"), s=("seconds", "sum"))
        axes[1].bar([f"stage {i}" for i in tot.index], tot.s / 3600, color=PAL[:len(tot)])
        for i, (n, s) in enumerate(zip(tot.n, tot.s)):
            axes[1].text(i, s / 3600, f"{int(n):,} papers\n{s/3600:.1f} h", ha="center", va="bottom", fontsize=9)
        axes[1].set_ylabel("Wall-clock hours"); axes[1].set_title("Compute time by stage", loc="left")
        self.save(fig, "fig22_laya_performance")

    def query_yield(self, qy):
        if qy is None or qy.empty or "hits" not in qy:
            return
        q = qy.dropna(subset=["hits"]).sort_values("query_id")
        fig, ax = plt.subplots(figsize=(9, max(3.5, .38 * len(q) + 1.5)))
        y = np.arange(len(q))
        ax.barh(y + .2, q.hits, .4, color="#C9D6E3", label="Records retrieved")
        ax.barh(y - .2, q.included, .4, color=PAL[0], label="Included")
        ax.barh(y - .2, q.unique_included, .4, color=PAL[3], label="Included & found ONLY by this query")
        ax.set_yticks(y); ax.set_yticklabels(q.query_id); ax.invert_yaxis()
        ax.set_xlabel("Papers"); ax.set_title("Search-query yield", loc="left"); ax.legend(loc="lower right")
        self.save(fig, "fig23_query_yield")

    def citations(self, inc, oa):
        if inc.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].hist(np.log10(inc.cited_by_count.fillna(0) + 1), bins=30, color=PAL[0])
        axes[0].set_xlabel("log10(citations + 1)"); axes[0].set_ylabel("Papers")
        axes[0].set_title("Citation distribution", loc="left")
        axes[1].barh(oa.oa_status.astype(str), oa.n, color=PAL[2])
        axes[1].invert_yaxis(); axes[1].set_xlabel("Papers"); axes[1].set_title("Open-access status", loc="left")
        self.save(fig, "fig17_citations_oa")

    # ------------------------------------------------------------ orchestrate
    def all(self, df, T, P, perf=None, only=None):
        sc = self.cfg["screening"]
        thr = df.attrs.get("threshold", cfg_thr(self.cfg))
        hw = df.attrs.get("halfwidth", float(self.cfg["decision"]["review_halfwidth"]))
        band = [max(0.0, thr - hw), min(1.0, thr + hw)]
        inc = df[df.decision == "included"].copy()
        jobs = [
            ("prisma", lambda: self.prisma(P)),
            ("yearly", lambda: self.yearly(T["yearly"])),
            ("monthly", lambda: self.monthly(T["monthly"])),
            ("shift", lambda: self.topic_shift(T.get("topic_shift"))),
            ("relevance", lambda: self.relevance_hist(df[df.score_final.notna() & df.dup_of.isna() & df.eligibility.eq("eligible")], thr, band)),
            ("threshold", lambda: self.threshold(T["threshold_sensitivity"], thr)),
            ("storage", lambda: self.hbar(T["storage_type_counts"].set_index("storage_type").n, "fig05_storage_setting",
                                          "Storage setting (Laya)", color=PAL[0])),
            ("storage_year", lambda: self.stacked_year(T["storage_type_by_year"].set_index("publication_year"),
                                                       "fig05b_storage_by_year", "Storage setting by year")),
            ("method", lambda: self.hbar(T["method_counts"].set_index("method").n, "fig06_method",
                                         "Research method (Laya)", color=PAL[1])),
            ("method_year", lambda: self.stacked_year(T["method_by_year"].set_index("publication_year"),
                                                      "fig06b_method_by_year", "Research method by year")),
            ("scale", lambda: self.hbar(T["scale_counts"].set_index("scale").n, "fig07_scale",
                                        "Scale of investigation (Laya)", color=PAL[2])),
            ("s_x_m", lambda: self.heat(T["storage_x_method"].set_index("storage_type_lab"), "fig08_storage_x_method",
                                        "Storage setting × research method", xlabel="Method", ylabel="Setting")),
            ("s_x_sc", lambda: self.heat(T["storage_x_scale"].set_index("storage_type_lab"), "fig09_storage_x_scale",
                                         "Storage setting × scale", xlabel="Scale", ylabel="Setting")),
            ("topics", lambda: self._topics(T)),
            ("trend", lambda: self._trend(T)),
            ("coverage", lambda: self.heat(T["coverage_matrix"].set_index("setting"), "fig12_coverage_matrix",
                                           "Research coverage: storage setting × physical process (abstract mentions)",
                                           cmap="viridis", figsize=(15, 4.5))),
            ("cooc", lambda: self.heat(T["tag_cooccurrence"].set_index("tag"), "fig13_tag_cooccurrence",
                                       "Topic co-occurrence (top 20 tags)", cmap="Blues", figsize=(10, 8))),
            ("journals", lambda: self.hbar(T["top_journals"].head(15).set_index("journal").n, "fig14_top_journals",
                                           "Top journals", color=PAL[4])),
            ("countries", lambda: self.hbar(T["top_countries"].head(15).set_index("country").n, "fig15_top_countries",
                                            "Top countries (authors' affiliations, whole counting)", color=PAL[5])),
            ("keywords", lambda: self.hbar(T["top_keywords"].head(20).set_index("keyword").n, "fig16_top_keywords",
                                           "Top OpenAlex keywords", color=PAL[2])),
            ("cit", lambda: self.citations(inc, T["oa_status"])),
            ("lith", lambda: self._group_bar(T, ("lithology", "fluid"), "fig18_lithology_fluids", "Lithology, seal and fluid mentions")),
            ("tools", lambda: self.hbar(T["top_tools"].set_index("tool").n, "fig19_simulation_tools",
                                        "Simulation / modelling software mentioned in abstracts", color=PAL[3])),
            ("agree", lambda: self._agreement(T)),
            ("conf", lambda: self.confidence(inc)),
            ("perf", lambda: self.performance(perf)),
            ("qy", lambda: self.query_yield(T["query_yield"])),
        ]
        for name, fn in jobs:
            if only and name not in only:
                continue
            try:
                fn()
            except Exception as e:  # one broken figure must never stop the report
                self.log.warning("figure '%s' failed: %s", name, e)
        self.log.info("Figures written: %d", len(self.made))
        return self.made

    def _topics(self, T):
        tp = T["topic_prevalence"]
        lex = tp[tp.source == "Lexicon"].sort_values("pct", ascending=False).head(25).set_index("topic").pct
        self.hbar(lex, "fig10_topic_prevalence", "Topic prevalence in included papers (% with abstract mention)",
                  xlabel="% of included papers", color=PAL[0])

    def _trend(self, T):
        t = T["tag_pct_by_year"].set_index("publication_year")
        top = t.mean().sort_values(ascending=False).head(18).index
        self.heat(t[top].T, "fig11_topic_trends", "Topic prevalence by year (% of included papers)", cmap="magma_r",
                  fmt="{:.0f}", figsize=(9, 7), xlabel="Year")

    def _group_bar(self, T, groups, name, title):
        tp = T["topic_prevalence"]
        s = tp[(tp.source == "Lexicon") & tp.group.isin(groups)].set_index("topic").n
        self.hbar(s, name, title, color=PAL[1])

    def _agreement(self, T):
        a = T["agreement_uhs"].set_index("rule_uhs")
        order = [x for x in ("strong", "likely", "unlikely", "none") if x in a.index]
        self.heat(a.loc[order], "fig20_laya_vs_rules", "Laya decision vs keyword-rule class (screened papers)",
                  cmap="Purples", figsize=(7, 4))
