#!/usr/bin/env python3
"""Generate the muon-site GSE comparison chart (self-contained HTML).

Regenerate after refreshing the plateau energies below from each run's
train_stats.csv (tools/energy_convergence.py). Writes muon_gse_comparison.html
next to this script. Run with the ferminet-piku env python.
"""
import os

# ---- data (block-averaged plateau energies, refreshed from train_stats.csv 2026-07-08) ----
# treatment: seeded | unseeded | classical
# star=True  => NOT strictly converged (tool verdict != CONVERGED)
DIAMOND = [
    # id, config, site, treatment, E, sem, steps, status, star
    ("#4",  "classical bc",        "fixed BC",  "classical", -90.73010, 0.00029, 338000, "converged",       False),
    ("#7",  "classical T_site",    "fixed T",   "classical", -90.69624, 0.00015, 472000, "converged",       False),
    ("#14", "bc_seeded",           "BC",        "seeded",    -90.69597, 0.00019, 580000, "converged",       False),
    ("#13", "classical t_relaxed", "fixed T*",  "classical", -90.69222, 0.00027, 428000, "descending",      True),
    ("#8",  "pp_relax_2",          "off-T",     "unseeded",  -90.66904, 0.00029, 280196, "frozen (final)",  True),
    ("#6",  "unrelaxed pp",        "T",         "unseeded",  -90.66312, 0.00017, 522000, "converged",       False),
    ("#12", "t_relaxed",           "T",         "unseeded",  -90.65979, 0.00023, 478000, "noise-limited",   True),
    ("#5",  "bc (original)",       "off-T",     "unseeded",  -90.59794, 0.00151, 336000, "noise-limited",   True),
    ("#18", "t_seeded",            "T",         "seeded",    -90.54469, 0.00208, 120000, "warm-up",         True),
]
SILICON = [
    ("#11", "classical t_relaxed", "fixed T*",  "classical", -62.92218, 0.00020, 306602, "diverged @306k",  True),
    ("#2",  "classical (T)",       "fixed T",   "classical", -62.91337, 0.00015, 223695, "frozen (final)",  True),
    ("#15", "bc_seeded",           "BC",        "seeded",    -62.89170, 0.00022, 196000, "descending",      True),
    ("#17", "t_seeded",            "T (relaxed)","seeded",   -62.89124, 0.00051, 224000, "noise-limited",   True),
    ("#10", "t_relaxed",           "T (fled)",  "unseeded",  -62.87392, 0.00034, 138000, "frozen (final)",  True),
    ("#16", "classical bc_relaxed","fixed BC",  "classical", -62.86852, 0.00229,  61386, "warm-up",         True),
    ("#1",  "unrelaxed",           "T",         "unseeded",  -62.86151, 0.00047, 180000, "frozen (final)",  True),
    ("#9",  "bc_relaxed",          "T (fled)",  "unseeded",  -62.86146, 0.00076, 138000, "frozen (final)",  True),
]

PANELS = [
    dict(key="diamond", title="Diamond", sub="2×2 supercell · 16 C + μ · doublet · pseudopotential · a = 6.74 bohr",
         dom=(-90.745, -90.535), ticks=[-90.74, -90.70, -90.66, -90.62, -90.58, -90.54], rows=DIAMOND),
    dict(key="silicon", title="Silicon", sub="2×2×2 supercell · 16 Si + μ · doublet · pseudopotential · a = 10.26 bohr",
         dom=(-62.928, -62.855), ticks=[-62.92, -62.90, -62.88, -62.86], rows=SILICON),
]

TREAT = {
    "seeded":   ("--c-seeded",   "Quantum · seeded"),
    "unseeded": ("--c-unseeded", "Quantum · unseeded"),
    "classical":("--c-classical","Classical (fixed μ)"),
}

def steplbl(n):
    return f"{n/1000:.0f}k" if n % 1000 == 0 else f"{n/1000:.1f}k"

def pct(x, lo, hi):
    return max(0.0, min(100.0, (x - lo) / (hi - lo) * 100.0))

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_panel(p):
    lo, hi = p["dom"]; width = hi - lo
    # gridline overlay
    lines = "".join(
        f'<div class="gline" style="left:{pct(t,lo,hi):.3f}%"></div>' for t in p["ticks"]
    )
    axis = "".join(
        f'<span class="tick" style="left:{pct(t,lo,hi):.3f}%">{t:.2f}</span>' for t in p["ticks"]
    )
    rows_html = []
    for (rid, cfg, site, treat, E, sem, steps, status, star) in p["rows"]:
        cvar = TREAT[treat][0]
        left = pct(E, lo, hi)
        ebw = (2 * sem) / width * 100.0
        eleft = pct(E - sem, lo, hi)
        dotcls = "dot" + (" hollow" if star else "")
        starmark = '<span class="star" title="not converged">★</span>' if star else ''
        tip = f"{rid} {cfg} · {site} · {TREAT[treat][1]} · {E:.5f} ± {sem:.5f} E_h · {steplbl(steps)} steps · {status}"
        row = f'''<div class="row">
  <div class="rlabel">
    <span class="rid">{esc(rid)}</span>
    <span class="rcfg">{esc(cfg)}</span>
    <span class="chip">{esc(site)}</span>
  </div>
  <div class="track" title="{esc(tip)}">
    <div class="ebar" style="left:{eleft:.3f}%;width:{ebw:.3f}%"></div>
    <div class="{dotcls}" style="left:{left:.3f}%;--c:var({cvar})"></div>
  </div>
  <div class="rval">
    <span class="e">{E:.5f}</span>{starmark}
    <span class="steps">{steplbl(steps)}</span>
  </div>
</div>'''
        rows_html.append(row)
    return f'''<section class="panel">
  <div class="phead">
    <h2>{p["title"]}</h2>
    <p class="psub">{esc(p["sub"])}</p>
  </div>
  <div class="plot">
    <div class="overlay">{lines}</div>
    <div class="rows">
      {''.join(rows_html)}
    </div>
    <div class="axisrow"><div class="axis">{axis}</div><div class="axislbl">ground-state energy&#8195;E&#8320;&#8202;(hartree)&#8195;&#8592; lower = better</div></div>
  </div>
</section>'''

panels_html = "\n".join(build_panel(p) for p in PANELS)
today = "2026-07-08"

html = f'''<div class="wrap">
<header class="masthead">
  <div class="eyebrow">VMC / FermiNet &middot; muon-site study &middot; refreshed {today}</div>
  <h1>Ground-state energy across all muon-site runs</h1>
  <p class="lede">Block-averaged plateau energies (&plusmn;1&nbsp;SEM) from each run&rsquo;s <code>train_stats.csv</code>.
  Colour marks the muon treatment; a filled dot is a converged run, a hollow dot with a
  <span class="star">&#9733;</span> is one that has <em>not</em> converged. Iteration count sits at the right of every row.</p>
  <div class="finding">
    <strong>Read:</strong> among <em>quantum</em> runs, seeding reaches the lowest energy in both systems &mdash;
    diamond <b>#14 bc_seeded</b> (converged, &minus;90.696&nbsp;E<sub>h</sub>) sits <b>26.9&nbsp;mHa below</b> the unseeded off-T <b>#8</b>,
    and silicon <b>#17 t_seeded</b> / <b>#15 bc_seeded</b> sit ~17&nbsp;mHa below the unseeded runs.
    Classical rows (fixed muon) carry <b>no zero-point energy</b> and sit artificially low &mdash; a reference, not a comparison.
  </div>
</header>

<div class="legend">
  <div class="lg"><span class="sw" style="--c:var(--c-seeded)"></span>Quantum &middot; seeded</div>
  <div class="lg"><span class="sw" style="--c:var(--c-unseeded)"></span>Quantum &middot; unseeded</div>
  <div class="lg"><span class="sw" style="--c:var(--c-classical)"></span>Classical (fixed &mu; &mdash; no ZPE)</div>
  <div class="lg sep"><span class="sw solid"></span>converged</div>
  <div class="lg"><span class="sw ring"></span><span class="star">&#9733;</span>&nbsp;not converged</div>
</div>

{panels_html}

<section class="tablewrap">
  <h2 class="tabh">All runs &mdash; full data</h2>
  <div class="tscroll">
  <table>
    <thead><tr>
      <th>Run</th><th>System</th><th>Config</th><th>Muon</th><th>Site</th>
      <th class="num">E (E<sub>h</sub>)</th><th class="num">&plusmn;SEM</th><th class="num">Steps</th><th>Status</th>
    </tr></thead>
    <tbody>
    {''.join(
      f'<tr class="{"conv" if not star else "unconv"}"><td class="mono">{esc(rid)}</td><td>{sysname}</td>'
      f'<td class="mono">{esc(cfg)}</td><td><span class="tdot" style="--c:var({TREAT[treat][0]})"></span>{TREAT[treat][1].split(" · ")[-1] if " · " in TREAT[treat][1] else TREAT[treat][1]}</td>'
      f'<td>{esc(site)}</td><td class="num">{E:.5f}</td><td class="num">{sem:.5f}</td>'
      f'<td class="num">{steplbl(steps)}</td><td>{"" if not star else "&#9733; "}{esc(status)}</td></tr>'
      for sysname, rows in (("Diamond", DIAMOND), ("Silicon", SILICON))
      for (rid, cfg, site, treat, E, sem, steps, status, star) in rows
    )}
    </tbody>
  </table>
  </div>
</section>

<footer class="notes">
  <h3>Notes &amp; caveats</h3>
  <ul>
    <li><b>Classical &ne; quantum.</b> Classical runs clamp the muon as a point charge, so they omit muon zero-point energy and sit lower by construction &mdash; not a better ground state, a different quantity. They are shown only as fixed-site references.</li>
    <li><b>Diamond and silicon are separate scales</b> (~&minus;90 vs ~&minus;62&nbsp;E<sub>h</sub>) and are never compared across panels.</li>
    <li><b>&#9733; = not converged</b> by the block-drift test (<code>tools/energy_convergence.py</code>, tol&nbsp;1&nbsp;mHa). &ldquo;noise-limited&rdquo; = slope statistically flat but drift just over tol (effectively plateaued); &ldquo;frozen&rdquo; = training stopped; &ldquo;descending / warm-up&rdquo; = still actively falling.</li>
    <li><b>Only 4 runs are strictly converged:</b> diamond #4, #7, #6, #14. All silicon runs are still descending or noise-limited; treat their energies as upper bounds.</li>
    <li><b>Silicon #11</b> classical t_relaxed diverged at step&nbsp;306.6k; the value shown is its last clean block. <b>Diamond #3</b> (all-electron, no pseudopotential, &minus;609&nbsp;E<sub>h</sub>) is a different Hamiltonian entirely and is omitted.</li>
  </ul>
</footer>
</div>'''

STYLE = '''
:root{
  --plane:#eef1f5; --surface:#f7f9fc; --ink:#0e1116; --ink2:#4a515c; --muted:#7b828d;
  --grid:#e2e6ec; --base:#c7cdd6; --border:rgba(14,17,22,.10);
  --c-seeded:#2a78d6; --c-unseeded:#eb6834; --c-classical:#4a3aa7;
  --chipbg:#e9edf3; --chipink:#4a515c; --findbg:#eaf1fb; --findbd:#2a78d6;
}
@media (prefers-color-scheme: dark){
  :root{
    --plane:#0c0e11; --surface:#15181d; --ink:#f4f6f9; --ink2:#c3c8d0; --muted:#8b929c;
    --grid:#262a31; --base:#3a3f48; --border:rgba(255,255,255,.10);
    --c-seeded:#3987e5; --c-unseeded:#d95926; --c-classical:#9085e9;
    --chipbg:#20242b; --chipink:#c3c8d0; --findbg:#16202e; --findbd:#3987e5;
  }
}
:root[data-theme="light"]{
  --plane:#eef1f5; --surface:#f7f9fc; --ink:#0e1116; --ink2:#4a515c; --muted:#7b828d;
  --grid:#e2e6ec; --base:#c7cdd6; --border:rgba(14,17,22,.10);
  --c-seeded:#2a78d6; --c-unseeded:#eb6834; --c-classical:#4a3aa7;
  --chipbg:#e9edf3; --chipink:#4a515c; --findbg:#eaf1fb; --findbd:#2a78d6;
}
:root[data-theme="dark"]{
  --plane:#0c0e11; --surface:#15181d; --ink:#f4f6f9; --ink2:#c3c8d0; --muted:#8b929c;
  --grid:#262a31; --base:#3a3f48; --border:rgba(255,255,255,.10);
  --c-seeded:#3987e5; --c-unseeded:#d95926; --c-classical:#9085e9;
  --chipbg:#20242b; --chipink:#c3c8d0; --findbg:#16202e; --findbd:#3987e5;
}
*{box-sizing:border-box}
body{margin:0}
.wrap{
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  background:var(--plane); color:var(--ink); font-family:var(--sans);
  padding:clamp(20px,4vw,52px); line-height:1.5;
  -webkit-font-smoothing:antialiased;
}
code{font-family:var(--mono); font-size:.9em; background:var(--chipbg); padding:.05em .35em; border-radius:4px}
.masthead{max-width:70ch; margin:0 auto 30px}
.eyebrow{font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:600}
h1{font-size:clamp(24px,3.4vw,36px); line-height:1.12; margin:.35em 0 .3em; letter-spacing:-.02em; text-wrap:balance; font-weight:680}
.lede{color:var(--ink2); font-size:16px; margin:0 0 16px}
.finding{background:var(--findbg); border-left:3px solid var(--findbd); border-radius:0 8px 8px 0;
  padding:12px 16px; font-size:14.5px; color:var(--ink)}
.finding b{font-weight:660}
.star{color:var(--c-unseeded); font-weight:700}
.legend{display:flex; flex-wrap:wrap; gap:8px 20px; max-width:1080px; margin:0 auto 26px;
  padding:12px 16px; background:var(--surface); border:1px solid var(--border); border-radius:10px; font-size:13px; color:var(--ink2)}
.lg{display:flex; align-items:center; gap:8px}
.lg.sep{margin-left:auto}
.sw{width:14px; height:14px; border-radius:50%; background:var(--c); flex:0 0 auto}
.sw.solid{background:var(--ink2)}
.sw.ring{background:var(--surface); border:2px solid var(--ink2)}

.panel{max-width:1080px; margin:0 auto 20px; background:var(--surface);
  border:1px solid var(--border); border-radius:14px; padding:20px clamp(14px,2.4vw,26px) 8px}
.phead{margin-bottom:12px}
.phead h2{margin:0; font-size:19px; letter-spacing:-.01em}
.psub{margin:2px 0 0; color:var(--muted); font-size:12.5px; font-family:var(--mono)}
.plot{position:relative}
.overlay{position:absolute; top:0; bottom:34px; left:calc(var(--lw) + var(--gap)); right:calc(var(--vw) + var(--gap)); pointer-events:none; z-index:0}
.gline{position:absolute; top:0; bottom:0; width:1px; background:var(--grid); transform:translateX(-.5px)}
.rows{position:relative; z-index:1}
:root{--lw:216px; --vw:150px; --gap:14px}
.row{display:grid; grid-template-columns:var(--lw) 1fr var(--vw); column-gap:var(--gap); align-items:center; min-height:38px}
.rlabel{display:flex; align-items:baseline; gap:8px; overflow:hidden}
.rid{font-family:var(--mono); font-weight:700; font-size:13px; color:var(--ink); flex:0 0 auto}
.rcfg{font-size:13px; color:var(--ink2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.chip{margin-left:auto; flex:0 0 auto; font-size:11px; font-family:var(--mono); background:var(--chipbg); color:var(--chipink);
  padding:1px 7px; border-radius:20px; letter-spacing:.01em}
.track{position:relative; height:38px}
.ebar{position:absolute; top:50%; height:2px; transform:translateY(-50%); background:var(--base); border-radius:2px; min-width:1px}
.dot{position:absolute; top:50%; width:13px; height:13px; border-radius:50%; transform:translate(-50%,-50%);
  background:var(--c); box-shadow:0 0 0 2px var(--surface); z-index:2}
.dot.hollow{background:var(--surface); border:2.5px solid var(--c); box-shadow:none}
.rval{display:flex; align-items:baseline; gap:7px; justify-content:flex-end; font-family:var(--mono); font-variant-numeric:tabular-nums}
.rval .e{font-size:13.5px; font-weight:640; color:var(--ink)}
.rval .steps{font-size:11.5px; color:var(--muted); min-width:34px; text-align:right}
.axisrow{margin-top:2px}
.axis{position:relative; height:20px; margin-left:calc(var(--lw) + var(--gap)); margin-right:calc(var(--vw) + var(--gap))}
.tick{position:absolute; transform:translateX(-50%); font-family:var(--mono); font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; top:2px}
.axislbl{text-align:center; margin-left:calc(var(--lw) + var(--gap)); margin-right:calc(var(--vw) + var(--gap));
  font-size:11px; color:var(--muted); letter-spacing:.02em; padding:2px 0 8px}

.tablewrap{max-width:1080px; margin:26px auto 8px}
.tabh{font-size:16px; margin:0 0 10px}
.tscroll{overflow-x:auto; border:1px solid var(--border); border-radius:12px}
table{border-collapse:collapse; width:100%; font-size:13px; min-width:640px; background:var(--surface)}
th,td{text-align:left; padding:8px 12px; border-bottom:1px solid var(--border); white-space:nowrap}
thead th{font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); font-weight:650; background:var(--chipbg)}
td.num,th.num{text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums}
td.mono{font-family:var(--mono)}
tr.conv td:first-child{box-shadow:inset 3px 0 0 var(--c-seeded)}
tbody tr:last-child td{border-bottom:none}
.tdot{display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--c); margin-right:7px; vertical-align:middle}

.notes{max-width:70ch; margin:28px auto 0; font-size:13.5px; color:var(--ink2)}
.notes h3{color:var(--ink); font-size:14px; text-transform:uppercase; letter-spacing:.06em; margin:0 0 8px}
.notes ul{margin:0; padding-left:18px; display:flex; flex-direction:column; gap:7px}
.notes b{color:var(--ink); font-weight:640}
@media (max-width:640px){
  :root{--lw:120px; --vw:120px; --gap:10px}
  .rcfg{display:none}
}
'''

here = os.path.dirname(os.path.abspath(__file__))

out = f"<title>Muon-site ground-state energies</title>\n<style>\n{STYLE}\n</style>\n{html}\n"
hpath = os.path.join(here, "muon_gse_comparison.html")
with open(hpath, "w") as f:
    f.write(out)
print("wrote", hpath, len(out), "bytes")


# ---- markdown twin (GitHub-readable) ----
def md_table(rows):
    out = ["| Run | Config | Muon site | Treatment | E (E_h) | ±SEM | Steps | Converged? |",
           "|-----|--------|-----------|-----------|--------:|-----:|------:|------------|"]
    for (rid, cfg, site, treat, E, sem, steps, status, star) in rows:
        conv = f"★ {status}" if star else "✓ converged"
        treat_lbl = {"seeded": "quantum · seeded", "unseeded": "quantum · unseeded",
                     "classical": "classical (fixed μ)"}[treat]
        out.append(f"| `{rid}` | `{cfg}` | {site} | {treat_lbl} | **{E:.5f}** | {sem:.5f} "
                   f"| {steplbl(steps)} | {conv} |")
    return "\n".join(out)

md = f"""# Muon-site ground-state energy comparison

*Block-averaged plateau energies (±1 SEM) from each run's `train_stats.csv`,
refreshed {today} via `tools/energy_convergence.py`. Interactive chart:
`muon_gse_comparison.html`; regenerate both with `gen_muon_gse_chart.py`.*

**Read:** among *quantum* runs, seeding reaches the lowest energy in both systems.
Diamond **#14 bc_seeded** (converged, −90.696 E_h) sits **26.9 mHa below** the
unseeded off-T **#8**; silicon **#17 t_seeded** / **#15 bc_seeded** sit ~17 mHa below
the unseeded runs. Classical rows fix the muon as a point charge, so they carry **no
zero-point energy** and sit artificially low — a fixed-site reference, not a
comparison. ★ marks a run that has **not** strictly converged.

## Diamond
*2×2 supercell · 16 C + μ · doublet · pseudopotential · a = 6.74 bohr. Lower E = better.*

{md_table(DIAMOND)}

## Silicon
*2×2×2 supercell · 16 Si + μ · doublet · pseudopotential · a = 10.26 bohr. Lower E = better.*

{md_table(SILICON)}

## Notes & caveats
- **Classical ≠ quantum.** Classical runs omit muon zero-point energy and sit lower by
  construction — a different quantity, shown only as fixed-site references.
- **Diamond and silicon are separate scales** (~−90 vs ~−62 E_h) and are never compared
  across the two tables.
- **★ = not converged** by the block-drift test (`tools/energy_convergence.py`, tol 1 mHa).
  *noise-limited* = slope statistically flat but drift just over tol (effectively
  plateaued); *frozen* = training stopped; *descending / warm-up* = still actively falling.
- **Only 4 runs are strictly converged:** diamond #4, #7, #6, #14. All silicon runs are
  still descending or noise-limited — treat their energies as upper bounds.
- **Silicon #11** diverged at step 306.6k; the value shown is its last clean block.
  **Diamond #3** (all-electron, no pseudopotential, −609 E_h) is a different Hamiltonian
  and is omitted.
"""
mpath = os.path.join(here, "muon_gse_comparison.md")
with open(mpath, "w") as f:
    f.write(md)
print("wrote", mpath, len(md), "bytes")
