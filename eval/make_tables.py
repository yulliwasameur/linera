"""Turn results/*.json into manuscrit/results_macros.tex and manuscrit/tab_*.tex.
No number in the paper is typed by hand: the prose uses the macros defined here.
Usage: python3 -I eval/make_tables.py --res results --tex ../manuscrit
"""
import argparse, json, os

def load(d, f):
    p = os.path.join(d, f)
    return json.load(open(p)) if os.path.exists(p) else None

D2W = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
def mname(s): return "".join(D2W.get(c, c) for c in s if c.isalnum())
def pct(x, d=1): return f"{100 * x:.{d}f}\\%"
def num(x): return f"{x:,}".replace(",", "{,}") if isinstance(x, int) else str(x)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="results"); ap.add_argument("--tex", default="../manuscrit")
    a = ap.parse_args(); M = {}
    def m(k, v): M[mname(k)] = v
    L = load(a.res, "results_locomo.json")
    if L:
        me = L["meta"]
        m("LocoConvs", num(len(me["conversations"]))); m("LocoSessions", num(me["sessions"])); m("LocoSources", num(me["sources"]))
        m("LocoDerived", num(me["derived"])); m("LocoLLMHours", f"{me['llm_seconds'] / 3600:.1f}")
        m("LocoTokensPrompt", num(me["llm_tokens_prompt"])); m("LocoTokensCompletion", num(me["llm_tokens_completion"]))
        c = L["E1_E2_canary"]; N = c["canaries"]
        m("LocoCanaries", num(N)); m("LocoCanPropExact", num(c["canaries_propagated_to_derived"]))
        m("LocoCanPropTopic", num(c["canaries_topic_propagated_to_derived"]))
        names = [("plain_logical", "Logical deletion + soft-deleted HNSW"), ("per_object", "Per-object keys (no lineage)"),
                 ("epoch", "Epoch keys"), ("linera_attributed", "\\sys{}, attributed rule"), ("linera_sound", "\\sys{}, sound rule")]
        rows = []
        for k, lab in names:
            r = c["schemes"][k]
            rows.append(f"{lab} & {r['recovered_by_decryption']} & {r['recovered_from_raw_bytes']} & {r['identified_from_vectors']} & "
                        f"{r['topic_leak_in_descendants']} & \\textbf{{{r['any_channel']}}} & {num(r['objects_lost'])}\\\\")
            for f in ("recovered_by_decryption", "recovered_from_raw_bytes", "identified_from_vectors", "topic_leak_in_descendants", "any_channel", "objects_lost"):
                m(f"Can{k}{f}", num(r[f]))
            m(f"Can{k}WriteMs", f"{r['write_ms_per_object']:.2f}"); m(f"Can{k}ForgetMs", f"{r['forget_ms_per_request']:.1f}")
        open(os.path.join(a.tex, "tab_canary.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{Erasure of planted secrets after a real LLM wrote the memory (LoCoMo, "
            f"{me['sessions']} sessions, {N} canary turns forgotten). The attacker holds every file and the current secret state. "
            "Columns count canaries (out of " f"{N}" ") recovered through each channel.}\\label{tab:canary}\n"
            "\\begin{tabular}{@{}lcccccc@{}}\\toprule\n"
            "Scheme & Exact secret in & Exact secret in & Identified from & Topic in a & Any & Objects\\\\\n"
            " & readable records & raw bytes & vectors (1 of 100) & descendant & channel & erased\\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
        e5 = L["E5_lineage"]
        for rule in ("sound", "attributed"):
            for f, v in e5[rule].items(): m(f"Lin{rule}{f}", num(v) if isinstance(v, int) else (f"{v:.1f}" if isinstance(v, float) else v))
        for rule in ("whatif_session_local", "whatif_flat"):
            for f, v in e5[rule].items(): m(f"Lin{rule}{f}", num(v) if isinstance(v, int) else f"{v:.1f}")
        for k, v in e5["derived_lost_at_rates"].items(): m(f"Rad{k}", pct(v))
        rr = []
        for lab, key, ks in [("Sound (as run, retrieves summaries)", "sound", "sound"), ("Attributed (top-3 context items)", "attributed", "attributed"),
                             ("Flat derivation$^\\dagger$", "whatif_flat", "flat"), ("Session-local$^\\dagger$", "whatif_session_local", None)]:
            s = e5[key]
            mp = f"{s['mean_parents']:.1f}" if "mean_parents" in s else "--"
            RAD = load(a.res, "results_radius.json")
            if ks: rad = " & ".join(pct(e5["derived_lost_at_rates"][f"{ks}@{p}"]) for p in (0.01, 0.05, 0.1))
            elif RAD: rad = " & ".join(pct(RAD["session_writer_conv0-4"]["session_local"][f"erased@{p}"]) for p in (0.01, 0.05, 0.1))
            else: rad = "-- & -- & --"
            rr.append(f"{lab} & {mp} & {s['single_forget_descendants_mean']:.1f} & {s['single_forget_descendants_p95']:.0f} & {s['single_forget_descendants_max']} & {rad}\\\\")
        open(os.path.join(a.tex, "tab_lineage.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{Lineage statistics and forgetting radius on the LoCoMo log. Descendants: derived objects "
            "erased by forgetting one turn. Radius: share of derived memories erased when a random fraction of turns is forgotten "
            "(mean of 3 seeds). $^\\dagger$Structural what-if on the same log: the parent sets that the alternative pipeline would record.}"
            "\\label{tab:lineage}\n\\begin{tabular}{@{}lccccccc@{}}\\toprule\n"
            "Lineage rule & Mean & \\multicolumn{3}{c}{Descendants of one turn} & \\multicolumn{3}{c}{Derived memories erased}\\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(l){6-8} & parents & mean & p95 & max & 1\\% & 5\\% & 10\\%\\\\\\midrule\n"
            + "\n".join(rr) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
        u = L.get("E3_retrieval")
        if u:
            R = u["results"]; m("LocoQuestions", num(u["questions"]))
            for k, v in R.items(): m("Ret" + k.replace("p=", "p").replace("|", ""), num(int(v["mean"])) if "n_questions" in k else pct(v["mean"]))
    QA = load(a.res, "results_qa_locomo.json"); QL = load(a.res, "results_qa_lme.json"); LM = load(a.res, "results_lme.json")
    if L and L.get("E3_retrieval"):
        R = L["E3_retrieval"]["results"]
        def g(p, c, v): return pct(R[f"p={p}|{c}|{v}"]["mean"]) if f"p={p}|{c}|{v}" in R else "--"
        rows = [f"No forgetting & \\multicolumn{{3}}{{c}}{{{g(0.0,'none','facts_only')}}} & \\multicolumn{{3}}{{c}}{{{g(0.0,'none','hybrid')}}}\\\\"]
        for c, lab in [("no_lineage", "No lineage (logical, per-object, epoch)"), ("attributed", "\\sys{}, attributed"), ("sound", "\\sys{}, sound")]:
            rows.append(f"{lab} & " + " & ".join(g(p, c, "facts_only") for p in (0.01, 0.05, 0.1)) + " & "
                        + " & ".join(g(p, c, "hybrid") for p in (0.01, 0.05, 0.1)) + "\\\\")
        open(os.path.join(a.tex, "tab_retrieval.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{Retrieval utility on LoCoMo after forgetting: share of retained questions for which "
            "a top-10 hit comes from an evidence session (mean of 3 seeds). Facts-only: the store keeps derived memories only (mem0-like); "
            "hybrid: raw turns are kept too.}\\label{tab:retrieval}\n\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{@{}lcccccc@{}}\\toprule\n"
            "& \\multicolumn{3}{c}{Facts-only} & \\multicolumn{3}{c}{Hybrid}\\\\\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
            "Condition & 1\\% & 5\\% & 10\\% & 1\\% & 5\\% & 10\\%\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if QA or QL:
        rows = []
        labs = [("none@0.0", "No forgetting"), ("no_lineage@0.01", "No lineage, 1\\%"), ("attributed@0.01", "\\sys{} attributed, 1\\%"),
                ("sound@0.01", "\\sys{} sound, 1\\%"), ("sound_rd@0.01", "\\sys{} sound + re-derivation, 1\\%"),
                ("no_lineage@0.1", "No lineage, 10\\%"), ("attributed@0.1", "\\sys{} attributed, 10\\%"),
                ("sound@0.1", "\\sys{} sound, 10\\%"), ("sound_rd@0.1", "\\sys{} sound + re-derivation, 10\\%")]
        for k, lab in labs:
            cells = []
            for Q, tag in ((QA, "Lo"), (QL, "Lme")):
                if Q and k in Q["conditions"]:
                    r = Q["conditions"][k]
                    cells += [f"{100*r['f1']:.1f}", f"{100*r['unknown_rate']:.0f}", str(r["rederive_llm_calls"]) if r["rederive_llm_calls"] else "--"]
                    nm = mname(k.replace("@", "at").replace(".", "p"))
                    m(f"QA{tag}{nm}Fone", f"{100*r['f1']:.1f}"); m(f"QA{tag}{nm}Unk", f"{100*r['unknown_rate']:.0f}\\%")
                    m(f"QA{tag}{nm}Alive", num(r["derived_alive"])); m(f"QA{tag}{nm}Calls", num(r["rederive_llm_calls"]))
                    m(f"QA{tag}{nm}SD", f"{100*r['f1_sd_bootstrap']:.1f}")
                    if "rederive_seconds" in r: m(f"QA{tag}{nm}Min", f"{r['rederive_seconds']/60:.0f}")
                else:
                    cells += ["--", "--", "--"]
            rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
        for Q, tag in ((QA, "Lo"), (QL, "Lme")):
            if Q: m(f"QA{tag}N", num(Q["meta"]["questions_scored"])); m(f"QA{tag}Eligible", num(Q["meta"]["questions_eligible"]))
        open(os.path.join(a.tex, "tab_qa.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{Answer quality from the surviving memory (token F1, \\%), share of ``unknown'' answers (\\%) "
            "and number of sessions re-summarised. Same LLM for writing, re-deriving and answering; only questions whose evidence "
            "survives are scored.}\\label{tab:qa}\n\\setlength{\\tabcolsep}{2.5pt}\\begin{tabular}{@{}lcccccc@{}}\\toprule\n"
            "& \\multicolumn{3}{c}{LoCoMo (conv.~0--1)} & \\multicolumn{3}{c}{LongMemEval-S subset}\\\\\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
            "Condition & F1 & unk. & re-der. & F1 & unk. & re-der.\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    if LM:
        m("LmeSessions", num(LM["meta"]["sessions"])); m("LmeSources", num(LM["meta"]["sources"])); m("LmeDerived", num(LM["meta"]["derived"]))
        for k, v in LM["E5_lineage"]["derived_lost_at_rates"].items(): m(f"LmeRad{k}", pct(v))
        if LM.get("E3_retrieval"):
            m("LmeQuestions", num(LM["E3_retrieval"]["questions"]))
            for k, v in LM["E3_retrieval"]["results"].items(): m("LmeRet" + k.replace("p=", "p").replace("|", ""), num(int(v["mean"])) if "n_questions" in k else pct(v["mean"]))
    WL = load(a.res, "results_locomo_win4.json")
    if WL and WL.get("E1_E2_canary"):
        c = WL["E1_E2_canary"]; m("WinCanaries", num(c["canaries"])); m("WinCanPropExact", num(c["canaries_propagated_to_derived"]))
        for k, r in c["schemes"].items():
            for f in ("any_channel", "objects_lost", "recovered_by_decryption"): m(f"WinCan{k}{f}", num(r[f]))
    RA = load(a.res, "results_radius.json")
    if RA:
        for blk, d in RA.items():
            for rule, v in d.items():
                if isinstance(v, dict):
                    for f, x in v.items():
                        if f == "fan_hist": continue
                        m(f"Rr{blk}{rule}{f}", pct(x) if "@" in f else f"{x:.1f}")
                else: m(f"Rr{blk}{rule}", num(v))
    QW = load(a.res, "results_qa_locomo_win4.json")
    if QW:
        for k, r in QW["conditions"].items():
            nm = mname(k.replace("@", "at").replace(".", "p"))
            m(f"QAWin{nm}Fone", f"{100*r['f1']:.1f}"); m(f"QAWin{nm}Calls", num(r["rederive_llm_calls"]))
            m(f"QAWin{nm}Alive", num(r["derived_alive"])); m(f"QAWin{nm}Orig", num(r["derived_original"]))
            if "rederive_seconds" in r: m(f"QAWin{nm}Min", f"{r['rederive_seconds']/60:.0f}")
    # ---- writer-context ablation and second model family (conversations 0-1)
    RA = load(a.res, "results_radius.json")
    if RA:
        QA01 = load(a.res, "results_qa_locomo.json")
        rows = []
        spec = [("Session writer (Qwen2.5-1.5B)", "session_writer_conv0-1", "results_locomo_c01.json", "results_qa_locomo.json", None)]
        spec += [(f"{w}-turn windows (Qwen2.5-1.5B)", f"window{w}_writer_conv0-1", f"results_locomo_win{w}.json", f"results_qa_locomo_win{w}.json", w) for w in (16, 8, 4, 2)]
        spec += [("Session writer (Llama-3.2-3B)", "llama3b_session_writer_conv0-1", "results_locomo_llama3b.json", "results_qa_locomo_llama3b.json", "L")]
        for lab, key, cf, qf, w in spec:
            if key not in RA: continue
            r = RA[key]["sound"]; C = load(a.res, cf); Q = load(a.res, qf)
            calls = RA[key].get("llm_calls", "--") if w not in (None, "L") else (
                sum(1 for _ in open("runs/locomo_log.jsonl") if '"conv": 0,' in _ or '"conv": 1,' in _) if w is None
                else (C["meta"]["sessions"] if C else "--"))
            der = RA[key].get("derived", "--")
            can = "--"
            if C and "E1_E2_canary" in C:
                cs = C["E1_E2_canary"]
                if True:
                    can = f"{cs['schemes']['per_object']['any_channel']}/{cs['canaries']} \\,|\\, {cs['schemes']['linera_sound']['any_channel']}/{cs['canaries']}"
            def f1(k):
                return f"{100*Q['conditions'][k]['f1']:.1f}" if Q and k in Q["conditions"] else "--"
            rows.append(f"{lab} & {calls} & {der} & {r['mean_source_ancestors']:.1f} & {pct(r['erased@0.01'])} & {pct(r['predicted@0.01'])} & "
                        f"{pct(r['erased@0.1'])} & {f1('none@0.0')} & {f1('sound@0.01')} & {f1('sound_rd@0.01')} & {f1('sound@0.1')} & {can}\\\\")
            tag = mname(key)
            for kk in ("erased@0.01", "erased@0.1", "predicted@0.01", "mean_source_ancestors"):
                m(f"Ab{tag}{kk}", pct(r[kk]) if "@" in kk else f"{r[kk]:.1f}")
            for kk in ("none@0.0", "sound@0.01", "sound_rd@0.01", "sound@0.1"):
                m(f"Ab{tag}Fone{kk}", f1(kk))
            if C and "E1_E2_canary" in C:
                cs = C["E1_E2_canary"]; m(f"Ab{tag}Canaries", num(cs["canaries"]))
                for sch, v in cs["schemes"].items(): m(f"Ab{tag}Can{sch}", num(v["any_channel"]))
                m(f"Ab{tag}CanProp", num(cs["canaries_propagated_to_derived"]))
        open(os.path.join(a.tex, "tab_ablation.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{Writer context size and model family (LoCoMo conversations 0--1, sound rule). "
            "Fan-in: mean number of source turns behind a derived memory. Erased: share of derived memories erased when a random 1\\% or 10\\% "
            "of turns is forgotten (3 seeds); predicted: fan-in law. F1: answers from the facts-only store (60 questions); rd: after re-derivation. "
            "Leaks: forgotten canaries recovered by any channel with per-object keys \\textbar{} with \\sys{}.}\\label{tab:ablation}\n"
            "\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{@{}lrrrrrrrrrrc@{}}\\toprule\n"
            "Writer & LLM & Derived & Fan-in & \\multicolumn{2}{c}{Erased at 1\\%} & Erased & \\multicolumn{4}{c}{F1} & Leaks\\\\\n"
            "\\cmidrule(lr){5-6}\\cmidrule(lr){8-11} & calls & memories & & measured & predicted & at 10\\% & none & 1\\% & 1\\%+rd & 10\\% & \\\\\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    # ---- extended campaign: Qwen2.5-7B-Instruct (vLLM, one H100), all 10 LoCoMo conversations + LongMemEval-S (100 q.)
    RA = load(a.res, "results_radius.json") or {}
    Q7 = [("LoCoMo, session writer", "Ses", "results_locomo_qwen7b.json", "results_qa_locomo_qwen7b.json", "qwen7b_session_writer_conv0-9"),
          ("LoCoMo, 4-turn windows", "Win", "results_locomo_qwen7b_win4.json", "results_qa_locomo_qwen7b_win4.json", "qwen7b_window4_writer_conv0-9"),
          ("LongMemEval-S, session writer", "Lme", "results_lme_qwen7b.json", "results_qa_lme_qwen7b.json", None)]
    rows7 = []
    for lab, tag, cf, qf, rk in Q7:
        C = load(a.res, cf); Q = load(a.res, qf)
        if not C: continue
        me = C["meta"]; t = f"QSeven{tag}"
        m(f"{t}Sessions", num(me["sessions"])); m(f"{t}Sources", num(me["sources"])); m(f"{t}Derived", num(me["derived"]))
        m(f"{t}Convs", num(len(me["conversations"]))); m(f"{t}TokensPrompt", num(me["llm_tokens_prompt"]))
        m(f"{t}LLMMin", f"{me['llm_seconds'] / 60:.0f}")
        so = C["E5_lineage"]["sound"]; at = C["E5_lineage"]["attributed"]
        m(f"{t}Fanin", f"{so['mean_source_ancestors']:.1f}"); m(f"{t}AttrFanin", f"{at['mean_source_ancestors']:.1f}")
        m(f"{t}Parents", f"{so['mean_parents']:.1f}"); m(f"{t}DescMean", f"{so['single_forget_descendants_mean']:.1f}")
        for k, v in C["E5_lineage"]["derived_lost_at_rates"].items(): m(f"{t}Rad{k}", pct(v))
        lk = ["--"] * 4
        if "E1_E2_canary" in C:
            cs = C["E1_E2_canary"]; S_ = cs["schemes"]
            m(f"{t}Canaries", num(cs["canaries"])); m(f"{t}CanProp", num(cs["canaries_propagated_to_derived"]))
            m(f"{t}CanPropTopic", num(cs["canaries_topic_propagated_to_derived"]))
            for sch, v in S_.items():
                for f in ("recovered_by_decryption", "recovered_from_raw_bytes", "identified_from_vectors", "topic_leak_in_descendants", "any_channel", "objects_lost"):
                    m(f"{t}Can{sch}{f}", num(v[f]))
            po, ep = S_["per_object"]["any_channel"], S_["epoch"]["any_channel"]
            lk = [str(S_["plain_logical"]["any_channel"]), str(po) if po == ep else f"{po}/{ep}",
                  str(S_["linera_attributed"]["any_channel"]), f"\\textbf{{{S_['linera_sound']['any_channel']}}}"]
        if rk and rk in RA:
            r = RA[rk]["sound"]; meas1, pred1, meas10 = pct(r["erased@0.01"]), pct(r["predicted@0.01"]), pct(r["erased@0.1"])
            m(f"{t}PredZeroZeroOne", pred1); m(f"{t}PredZeroOne", pct(r["predicted@0.1"]))
            if "session_local" in RA[rk]:
                sl = RA[rk]["session_local"]; m(f"{t}LocalFanin", f"{sl['mean_source_ancestors']:.1f}"); m(f"{t}LocalRadZeroZeroOne", pct(sl["erased@0.01"]))
        else:
            dl = C["E5_lineage"]["derived_lost_at_rates"]; meas1, pred1, meas10 = pct(dl["sound@0.01"]), "--", pct(dl["sound@0.1"])
        def f7(k):
            return f"{100*Q['conditions'][k]['f1']:.1f}" if Q and k in Q["conditions"] else "--"
        if Q:
            m(f"{t}QAN", num(Q["meta"]["questions_scored"]))
            for k, r in Q["conditions"].items():
                nm = mname(k.replace("@", "at").replace(".", "p"))
                m(f"{t}QA{nm}Fone", f"{100*r['f1']:.1f}"); m(f"{t}QA{nm}SD", f"{100*r['f1_sd_bootstrap']:.1f}")
                m(f"{t}QA{nm}Calls", num(r["rederive_llm_calls"])); m(f"{t}QA{nm}Alive", num(r["derived_alive"]))
                m(f"{t}QA{nm}Unk", f"{100*r['unknown_rate']:.0f}\\%")
                if "rederive_seconds" in r: m(f"{t}QA{nm}Min", f"{r['rederive_seconds']/60:.0f}")
        rows7.append(f"{lab} & {num(me['derived'])} & {so['mean_source_ancestors']:.1f} & " + " & ".join(lk) + f" & {meas1} & {pred1} & {meas10} & "
                     + " & ".join(f7(k) for k in ("none@0.0", "sound@0.01", "sound_rd@0.01", "sound@0.1", "sound_rd@0.1")) + "\\\\")
    G7 = load(a.res, "results_mem0_qwen7b.json")
    if G7:
        v, l = G7["vanilla"], G7["linera"]
        m("QSevenMemMessages", num(G7["messages"])); m("QSevenMemVanDeleted", num(v["memories_deleted"]))
        m("QSevenMemVanSecretsFiles", num(len(v["secrets_in_files_after"])))
        m("QSevenMemVanFiles", ", ".join(sorted({"\\texttt{" + f.replace("_", "\\_") + "}" for fs in v["secrets_in_files_after"].values() for f in fs})) or "none")
        m("QSevenMemLinBefore", num(l["memories_before"])); m("QSevenMemLinClosure", num(l["objects_in_closures"]))
        m("QSevenMemLinAfter", num(len(l["memories_text_after"]))); m("QSevenMemLinSecretsFiles", num(len(l["secrets_in_files_after"])))
        m("QSevenMemLinSecretsDec", num(len(l["secrets_in_decryptable_records"])))
    if rows7:
        open(os.path.join(a.tex, "tab_qwen7b.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{Extended campaign: Qwen2.5-7B-Instruct writes, re-derives and answers (vLLM, one GPU). "
            "LoCoMo: all ten conversations; LongMemEval-S: 100 histories. Fan-in: mean number of source turns behind a derived memory "
            "(sound rule). Leaks: forgotten canaries recovered by any channel (\\cref{sec:e1}). Erased: share of derived memories erased "
            "by the sound rule when a random 1\\% or 10\\% of turns is forgotten (3 seeds); predicted: fan-in law. F1: token F1 (\\%) from "
            "the facts-only store over every eligible question; rd: after re-derivation.}\\label{tab:qwen7b}\n"
            "\\setlength{\\tabcolsep}{2.6pt}\\begin{tabular}{@{}lrrcccccccrrrrr@{}}\\toprule\n"
            "Writer & Derived & Fan-in & \\multicolumn{4}{c}{Leaks} & \\multicolumn{2}{c}{Erased at 1\\%} & Erased & \\multicolumn{5}{c}{F1}\\\\\n"
            "\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}\\cmidrule(l){11-15} & memories & & logical & obj./ep. & attr. & sound & meas. & pred. & at 10\\% "
            "& none & 1\\% & 1\\%+rd & 10\\% & 10\\%+rd\\\\\\midrule\n" + "\n".join(rows7) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    S = load(a.res, "results_scale.json")
    if S:
        rows = []
        for k in sorted([k for k in S if ":" in k], key=lambda x: (int(x.split(":")[1]), x)):
            r = S[k]; rg, n = k.split(":")
            rows.append(f"{rg.capitalize()} & {num(r['objects'])} & {r['write_ms_per_object']:.2f} & {r['get_ms_cold']:.2f} & {r['search_warm_ms']:.0f} & "
                        f"{r['forget_ms_median']:.0f} & {r['closure_descendants_median']:.0f} & {r['pprf_state_KiB_after_forgets']:.1f} & {r['rotate_s']:.1f} & {r['db_MiB']:.0f}\\\\")
            for f, v in r.items():
                if f == "regime": continue
                if isinstance(v, int): m(f"Sc{rg}{n}{f}", num(v))
                else:
                    x = float(v)
                    m(f"Sc{rg}{n}{f}", num(int(round(x))) if (abs(x) >= 100 or x == int(x)) else (f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"))
        open(os.path.join(a.tex, "tab_scale.tex"), "w").write(
            "\\begin{table*}[t]\\centering\\caption{System costs of \\sys{} (Python, one process on the 2-core Intel Xeon at 2.1~GHz). Write includes "
            "the vector index; search is a warm top-10 scan over all shards; forget includes the closure, the puncture, the atomic state "
            "write and the shard rebuilds (median of 100 random source turns); state is the PPRF key after these forgets.}\\label{tab:scale}\n"
            "\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\n"
            "Regime & Objects & Write & Read & Search & Forget & Descendants & State & Rotate & DB\\\\\n"
            " & & (ms) & (ms) & (ms) & (ms) & (median) & (KiB) & (s) & (MiB)\\\\\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    G = load(a.res, "results_mem0.json")
    if G:
        v, l = G["vanilla"], G["linera"]
        m("MemVanBefore", num(v["memories_before"])); m("MemVanDeleted", num(v["memories_deleted"])); m("MemVanAfter", num(v["memories_after"]))
        m("MemVanSecretsFiles", num(len(v["secrets_in_files_after"])))
        m("MemVanFiles", ", ".join(sorted({"\\texttt{" + f.replace("_", "\\_") + "}" for fs in v["secrets_in_files_after"].values() for f in fs})) or "none")
        m("MemLinBefore", num(l["memories_before"])); m("MemLinAfter", num(len(l["memories_text_after"]))); m("MemLinClosure", num(l["objects_in_closures"]))
        m("MemLinSecretsFiles", num(len(l["secrets_in_files_after"]))); m("MemLinSecretsDec", num(len(l["secrets_in_decryptable_records"])))
        m("MemSecrets", num(len([s for s in ["74219", "Quilbramdor", "55120"]])))
    C = load(a.res, "results_canary_synthetic.json")
    if C:
        for k, r in C["results"].items():
            m(f"Syn{k}Recovered", num(r["canary_recovered_by_decryption"])); m(f"Syn{k}Radius", pct(r["blast_radius_fraction"]))
        m("SynCanaries", num(C["results"]["lineage"]["canaries"])); m("SynObjects", num(C["results"]["lineage"]["objects"]))
    with open(os.path.join(a.tex, "results_macros.tex"), "w") as f:
        f.write("% generated by code/eval/make_tables.py -- do not edit\n")
        for k, v in sorted(M.items()): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(len(M), "macros")

if __name__ == "__main__":
    main()
