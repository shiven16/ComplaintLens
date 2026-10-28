#!/usr/bin/env python3
"""Analyze processed CFPB splits and write aggregate tables and SVG figures."""
from __future__ import annotations

import csv
import html
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

INPUT = Path("data/processed")
REPORTS = Path("reports")
FIGURES = REPORTS / "figures"
SPLITS = ("train", "validation", "test")
SAMPLE_N = 50_000
SEED = 42
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by can cannot could did do does doing down during each
few for from further had has have having he her here hers herself him himself his how
i if in into is it its itself just me more most must my myself no nor not of off on once
only or other ought our ours ourselves out over own same she should so some such than
that the their them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why with would you your
yours yourself yourselves
""".split())


def write_csv(name, header, rows):
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    r = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        r[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    return r


def gamma_q(a, x):
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    eps, tiny = 3e-14, 1e-300
    if x < a + 1:
        ap = a
        term = total = 1.0 / a
        for _ in range(100000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        return max(0.0, min(1.0, 1.0 - total * math.exp(-x + a * math.log(x) - gln)))
    b, c = x + 1 - a, 1 / tiny
    d = 1 / (b if abs(b) > tiny else tiny)
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        if abs(d) < tiny: d = tiny
        c = b + an / c
        if abs(c) < tiny: c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps: break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - gln) * h))


def beta_cf(a, b, x):
    eps, tiny, qab, qap, qam = 3e-14, 1e-300, a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < tiny: d = tiny
    d = 1 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < tiny: d = tiny
        c = 1 + aa / c
        if abs(c) < tiny: c = tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps: break
    return h


def beta_i(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x))
    if x < (a+1)/(a+b+2):
        return bt * beta_cf(a,b,x) / a
    return 1 - bt * beta_cf(b,a,1-x) / b


def kw_test(values, codes, k):
    n = len(values)
    rr = ranks(values)
    sizes = np.bincount(codes, minlength=k).astype(float)
    sums = np.bincount(codes, weights=rr, minlength=k)
    h = 12/(n*(n+1))*np.sum(sums*sums/sizes)-3*(n+1)
    _, ties = np.unique(values, return_counts=True)
    correction = 1 - np.sum(ties.astype(float)**3-ties)/(n**3-n)
    h /= correction
    df = k-1
    return h, df, gamma_q(df/2, h/2), (h-k+1)/(n-k), correction


def chi2_test(tab):
    n = tab.sum()
    expected = tab.sum(1,keepdims=True) @ tab.sum(0,keepdims=True) / n
    stat = float(np.sum((tab-expected)**2/expected))
    df = (tab.shape[0]-1)*(tab.shape[1]-1)
    v = math.sqrt(stat/(n*min(tab.shape[0]-1,tab.shape[1]-1)))
    return stat, df, gamma_q(df/2,stat/2), v


def spearman_p(rho, n):
    df = n-2
    t2 = rho*rho*df/max(1-rho*rho,1e-300)
    return beta_i(df/2, .5, df/(df+t2))


def scores(actual, pred):
    acc = sum(a==p for a,p in zip(actual,pred))/len(actual)
    f1s = []
    for label in (0,1):
        tp=sum(a==label and p==label for a,p in zip(actual,pred))
        fp=sum(a!=label and p==label for a,p in zip(actual,pred))
        fn=sum(a==label and p!=label for a,p in zip(actual,pred))
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return sum(f1s)/2, acc


def report_p(p):
    return "<1e-300" if p == 0 else p


def esc(s): return html.escape(str(s), quote=True)


def tx(x,y,s,size=12,anchor="start",fill="#222",rotate=None,weight="normal"):
    tr=f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}"{tr}>{esc(s)}</text>'


def ln(x1,y1,x2,y2,color="#ddd"):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}"/>'


def save_svg(name,w,h,title,parts):
    FIGURES.mkdir(parents=True,exist_ok=True)
    body=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>',tx(w/2,30,title,20,"middle",weight="bold"),*parts,"</svg>"]
    (FIGURES/name).write_text("\n".join(body)+"\n",encoding="utf-8")


def figures(product_counts, hist, rates):
    colors={"train":"#3569a8","validation":"#e09b2d","test":"#4d9461"}
    products=sorted({p for _,p in product_counts})
    w,h=1420,120+58*len(products)
    x0,cw,y0=450,700,85
    mx=max(product_counts.values())
    p=[]
    for tick in range(6):
        x=x0+cw*tick/5
        p += [ln(x,y0-8,x,y0+len(products)*58),tx(x,y0-16,f"{mx*tick/5:,.0f}",10,"middle")]
    for i,prod in enumerate(products):
        base=y0+i*58
        p.append(tx(x0-12,base+31,prod,11,"end"))
        for j,split in enumerate(SPLITS):
            val=product_counts.get((split,prod),0)
            p.append(f'<rect x="{x0}" y="{base+j*15}" width="{cw*val/mx:.1f}" height="11" fill="{colors[split]}"/>')
    for j,split in enumerate(SPLITS):
        p += [f'<rect x="1190" y="{54+j*22}" width="14" height="14" fill="{colors[split]}"/>',tx(1212,65+j*22,split,12)]
    save_svg("product_counts_per_split.svg",w,h,"Product counts per split",p)

    bins,hcounts=hist
    mx=max(max(hcounts[s]) for s in SPLITS)
    w,h=1240,610
    x0,y0,cw,ch=90,90,1090,390
    p=[]
    maxlog=math.log10(mx+1)
    for tick in (1,10,100,1000,10000,100000):
        if tick<=mx:
            y=y0+ch*(1-math.log10(tick+1)/maxlog)
            p += [ln(x0,y,x0+cw,y),tx(x0-10,y+4,f"{tick:,}",10,"end")]
    gw=cw/len(bins); bw=gw*.23
    for i,b in enumerate(bins):
        for j,split in enumerate(SPLITS):
            val=hcounts[split][i]
            x=x0+i*gw+gw*.12+j*bw
            y=y0+ch*(1-math.log10(val+1)/maxlog)
            p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-2:.1f}" height="{y0+ch-y:.1f}" fill="{colors[split]}"/>')
        p.append(tx(x0+i*gw+gw/2,y0+ch+22,b,10,"middle"))
    for j,split in enumerate(SPLITS):
        x=900+j*100
        p += [f'<rect x="{x}" y="54" width="14" height="14" fill="{colors[split]}"/>',tx(x+20,67,split,11)]
    p.append(tx(x0+cw/2,h-22,"Narrative length (words)",13,"middle"))
    save_svg("narrative_length_histogram.svg",w,h,"Narrative length histogram by split",p)

    products=sorted({prod for _,prod in rates})
    w,h=1350,120+58*len(products)
    x0,cw,y0=430,700,85
    p=[]
    for tick in range(0,101,20):
        x=x0+cw*tick/100
        p += [ln(x,y0-8,x,y0+len(products)*58),tx(x,y0-16,f"{tick}%",10,"middle")]
    for i,prod in enumerate(products):
        base=y0+i*58
        p.append(tx(x0-12,base+31,prod,11,"end"))
        for j,split in enumerate(SPLITS):
            p.append(f'<rect x="{x0}" y="{base+j*15}" width="{cw*rates.get((split,prod),0):.1f}" height="11" fill="{colors[split]}"/>')
    for j,split in enumerate(SPLITS):
        p += [f'<rect x="1160" y="{54+j*22}" width="14" height="14" fill="{colors[split]}"/>',tx(1182,65+j*22,split,12)]
    save_svg("priority_rate_by_product.svg",w,h,"Priority rate by product and split",p)


def main():
    REPORTS.mkdir(parents=True,exist_ok=True)
    count=Counter(); pos=Counter(); total=Counter()
    ranges={s:{"min":None,"max":None} for s in SPLITS}
    lengths={s:[] for s in SPLITS}
    flags={s:Counter() for s in SPLITS}
    yearly=Counter(); yearly_pos=Counter()
    train_products=[]; train_y=[]; train_pr=[]; by_prod_len=defaultdict(list)
    rng=random.Random(SEED); sample=[]; seen_train=0

    for split in SPLITS:
        path=INPUT/f"{split}.parquet"
        if not path.exists(): raise FileNotFoundError(path)
        pf=pq.ParquetFile(path)
        expected={"complaint_id","date_received","product","issue","narrative","company_response","timely"}
        actual=set(pf.schema_arrow.names)
        if actual!=expected: raise ValueError(f"{path}: unexpected columns {pf.schema_arrow.names}")
        for batch in pf.iter_batches(batch_size=5000,columns=["date_received","product","narrative","company_response","timely"]):
            d=batch.to_pydict()
            for dt,product,text,response,timely in zip(d["date_received"],d["product"],d["narrative"],d["company_response"],d["timely"]):
                if dt is None: raise ValueError(f"Null date_received in {split}")
                product=product or "[MISSING]"
                text=text or ""
                day=dt.date() if hasattr(dt,"date") else dt
                ranges[split]["min"]=day if ranges[split]["min"] is None or day<ranges[split]["min"] else ranges[split]["min"]
                ranges[split]["max"]=day if ranges[split]["max"] is None or day>ranges[split]["max"] else ranges[split]["max"]
                length=len(WORD_RE.findall(text))
                no=timely=="No"
                monetary="monetary relief" in (response or "").casefold()
                positive=no or monetary
                count[(split,product)]+=1
                pos[(split,product)]+=int(positive)
                total[split]+=1
                flags[split]["positive"]+=int(positive)
                flags[split]["timely_no_any"]+=int(no)
                flags[split]["monetary_any"]+=int(monetary)
                flags[split]["both"]+=int(no and monetary)
                flags[split]["timely_no_only"]+=int(no and not monetary)
                flags[split]["monetary_only"]+=int(monetary and not no)
                lengths[split].append(length)
                if split=="train":
                    train_products.append(product); train_y.append(length); train_pr.append(int(positive))
                    by_prod_len[product].append(length)
                    yearly[dt.year]+=1; yearly_pos[dt.year]+=int(positive)
                    seen_train+=1
                    if len(sample)<SAMPLE_N: sample.append(text)
                    else:
                        j=rng.randrange(seen_train)
                        if j<SAMPLE_N: sample[j]=text

    # Split and product summaries.
    split_rows=[]
    for s in SPLITS:
        n=total[s]; p=flags[s]["positive"]
        split_rows.append([s,n,p,n-p,p/n,ranges[s]["min"],ranges[s]["max"]])
    write_csv("processed_split_summary.csv",["split","rows","priority_positive","priority_negative","priority_positive_rate","min_date_received","max_date_received"],split_rows)
    product_rows=[]; prodcounts={}; prod_rates={}
    for s,p in sorted(count):
        n=count[(s,p)]; yes=pos[(s,p)]
        product_rows.append([s,p,n,yes,n-yes,yes/n])
        prodcounts[(s,p)]=n; prod_rates[(s,p)]=yes/n
    write_csv("processed_product_class_counts.csv",["split","product","rows","priority_positive","priority_negative","priority_positive_rate"],product_rows)

    # Split narrative length summaries and histograms.
    bins=["<10","10-24","25-49","50-99","100-249","250-499","500-999","1000-1999","2000+"]
    hcounts={s:[0]*len(bins) for s in SPLITS}
    length_rows=[]
    for s in SPLITS:
        a=np.asarray(lengths[s],dtype=np.int64)
        q50,q95,q99=np.percentile(a,[50,95,99])
        length_rows.append([s,len(a),float(a.mean()),float(q50),float(q95),float(q99),int(a.min()),int(a.max())])
        for v in a:
            i=0 if v<10 else 1 if v<25 else 2 if v<50 else 3 if v<100 else 4 if v<250 else 5 if v<500 else 6 if v<1000 else 7 if v<2000 else 8
            hcounts[s][i]+=1
    write_csv("narrative_length_summary_by_split.csv",["split","rows","mean_words","median_words","p95_words","p99_words","min_words","max_words"],length_rows)
    write_csv("narrative_length_histogram_by_split.csv",["split","word_count_bin","rows"],[[s,b,hcounts[s][i]] for s in SPLITS for i,b in enumerate(bins)])

    breakdown=[]
    for s in SPLITS:
        n=total[s]; f=flags[s]; p=f["positive"]
        breakdown.append([s,n,p,p/n,f["timely_no_any"],f["monetary_any"],f["both"],f["timely_no_only"],f["monetary_only"],f["timely_no_only"]/n,f["monetary_only"]/n,f["both"]/n])
    write_csv("priority_flag_breakdown_by_split.csv",["split","rows","positive_union","positive_rate","timely_no_any","monetary_relief_any","both","timely_no_only","monetary_relief_only","timely_no_only_rate","monetary_relief_only_rate","both_rate"],breakdown)

    # Train Kruskal-Wallis with average ranks and tie correction.
    products=sorted(set(train_products)); code={p:i for i,p in enumerate(products)}
    codes=np.fromiter((code[p] for p in train_products),dtype=np.int32,count=len(train_products))
    y=np.asarray(train_y,dtype=np.float64)
    n=len(y); k=len(products); rr=ranks(y)
    sizes=np.bincount(codes,minlength=k).astype(float)
    sums=np.bincount(codes,weights=rr,minlength=k)
    H=12/(n*(n+1))*np.sum(sums*sums/sizes)-3*(n+1)
    _,ties=np.unique(y,return_counts=True)
    tie_correction=1-np.sum(ties.astype(float)**3-ties)/(n**3-n)
    H/=tie_correction; kw_df=k-1; kw_p=gamma_q(kw_df/2,H/2); epsilon=(H-k+1)/(n-k)
    product_len_rows=[]
    for p in products:
        a=np.asarray(by_prod_len[p],dtype=np.int64)
        product_len_rows.append([p,len(a),float(a.mean()),float(np.median(a)),int(a.min()),int(a.max())])
    write_csv("train_narrative_length_by_product.csv",["product","rows","mean_words","median_words","min_words","max_words"],product_len_rows)

    table=np.zeros((k,2),dtype=np.int64)
    for p,flag in zip(train_products,train_pr): table[code[p],flag]+=1
    expected=table.sum(1,keepdims=True)@table.sum(0,keepdims=True)/table.sum()
    chi=float(np.sum((table-expected)**2/expected)); chi_df=(k-1); chi_p=gamma_q(chi_df/2,chi/2)
    cramer=math.sqrt(chi/(table.sum()*min(k-1,1)))

    years=sorted(yearly); rates=[yearly_pos[y]/yearly[y] for y in years]
    rho=float(np.corrcoef(ranks(np.asarray(years,dtype=float)),ranks(np.asarray(rates,dtype=float)))[0,1])
    df=len(years)-2
    t2=rho*rho*df/max(1-rho*rho,1e-300)
    rho_p=beta_i(df/2,.5,df/(df+t2))
    write_csv("train_yearly_priority_rate.csv",["year","rows","priority_positive","priority_rate"],[[y,yearly[y],yearly_pos[y],yearly_pos[y]/yearly[y]] for y in years])
    write_csv("train_statistical_tests.csv",["test","statistic","df_or_n","p_value","effect_size","effect_size_value","details"],[
        ["Kruskal-Wallis narrative length by product",H,kw_df,report_p(kw_p),"epsilon-squared",epsilon,f"n={n}; groups={k}; tie correction={tie_correction:.12g}"],
        ["Chi-square product by priority flag",chi,chi_df,report_p(chi_p),"Cramer's V",cramer,f"n={n}; groups={k}"],
        ["Spearman yearly priority rate vs year",rho,len(years),rho_p,"Spearman rho",rho,f"years {min(years)}-{max(years)}; n={len(years)} years"],
    ])

    # Baselines, learned from train and evaluated on validation.
    overall_majority=int(sum(train_pr)>len(train_pr)-sum(train_pr))
    product_majority={}
    for p in products:
        positive=pos[("train",p)]; nprod=count[("train",p)]
        product_majority[p]=int(positive>nprod-positive)
    actual=[]; pred_major=[]; pred_product=[]
    val=pq.ParquetFile(INPUT/"validation.parquet")
    for batch in val.iter_batches(batch_size=5000,columns=["product","timely","company_response"]):
        d=batch.to_pydict()
        for p,t,r in zip(d["product"],d["timely"],d["company_response"]):
            actual.append(int(t=="No" or "monetary relief" in (r or "").casefold()))
            pred_major.append(overall_majority)
            pred_product.append(product_majority.get(p or "[MISSING]",overall_majority))
    baseline_rows=[]
    for name,predictions,rule in [
        ("majority-class",pred_major,f"always train majority class ({overall_majority})"),
        ("product-only",pred_product,"train majority within product; global fallback for unseen"),
    ]:
        macro_f1,acc=scores(actual,predictions)
        baseline_rows.append([name,len(actual),macro_f1,acc,rule])
    write_csv("validation_baselines.csv",["baseline","validation_rows","macro_f1","accuracy","rule"],baseline_rows)

    vocab=Counter()
    all_vocabulary=set()
    for text in sample:
        tokens=[t.casefold() for t in TOKEN_RE.findall(text)]
        all_vocabulary.update(tokens)
        vocab.update(t for t in tokens if t not in STOPWORDS)
    write_csv("train_vocabulary_summary.csv",["sample_rows","seed","all_vocab_size","non_stopword_vocab_size","non_stopword_token_occurrences"],[[len(sample),SEED,len(all_vocabulary),len(vocab),sum(vocab.values())]])
    write_csv("train_vocabulary_top30_nonstop.csv",["rank","token","count"],[[i,t,c] for i,(t,c) in enumerate(vocab.most_common(30),1)])

    figures(prodcounts,(bins,hcounts),prod_rates)
    print("Split summaries:")
    for r in split_rows: print(f"{r[0]}: {r[1]:,} rows, priority {r[4]:.3%}, {r[5]} to {r[6]}")
    print(f"Kruskal-Wallis H={H:.6g}, p={report_p(kw_p)}, epsilon-squared={epsilon:.6g}")
    print(f"Product/priority chi-square={chi:.6g}, p={report_p(chi_p)}, Cramer's V={cramer:.6g}")
    print(f"Yearly priority Spearman rho={rho:.6g}, p={rho_p:.6g}")
    for r in baseline_rows: print(f"Validation {r[0]}: macro-F1={r[2]:.4f}, accuracy={r[3]:.4f}")
    print(f"Train vocabulary sample={len(sample):,}, vocabulary={len(all_vocabulary):,}, non-stopword={len(vocab):,}")
    print(f"Tables and SVGs saved under {REPORTS}")


if __name__ == "__main__":
    main()
