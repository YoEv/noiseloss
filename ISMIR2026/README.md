# ISMIR 2026 Submission

Source files for the ISMIR 2026 paper on model-internal / intrinsic signal
analysis for music quality evaluation (loss, entropy, and SAE features).

## Directory Layout

```
ISMIR2026/
├── README.md                       # this file
├── latex/                          # LaTeX source
│   ├── ISMIR2026_template.tex      # main manuscript
│   ├── ISMIRtemplate.bib           # bibliography
│   ├── ismir.sty                   # ISMIR 2026 style
│   ├── IEEEtran.bst                # bibliography style used by .tex
│   ├── cite.sty                    # cite package
│   ├── cc_by.{eps,pdf,png}         # CC-BY license logo
│   ├── example.png                 # example figure asset
│   └── ISMIR2026_template.pdf      # compiled output (regenerated on build)
└── word/                           # Word-format assets (if any)
```

## Build

The LaTeX source is compiled with [Tectonic](https://tectonic-typesetting.github.io/),
which is already installed in the `torch21` conda env on this machine
(together with `IEEEtran.bst` used by BibTeX).

Activate the env and build:

```bash
conda activate torch21
cd ISMIR2026/latex
tectonic ISMIR2026_template.tex
```

Tectonic handles the BibTeX pass, re-runs, and intermediate cleanup automatically.
Output is `ISMIR2026_template.pdf` in the same directory.

### Keeping intermediates (for debugging `.bbl` / `.aux`)

```bash
tectonic --keep-intermediates ISMIR2026_template.tex
```

### Alternative: vanilla `pdflatex` + `bibtex` (if Tectonic is unavailable)

```bash
cd ISMIR2026/latex
pdflatex ISMIR2026_template
bibtex   ISMIR2026_template
pdflatex ISMIR2026_template
pdflatex ISMIR2026_template
```

Note: the conda-forge `texlive-core` package on this machine is missing
`tlpkg/` (causing `fmtutil` / `mktexfmt` to fail on first run),
so Tectonic is the recommended path.

## Editing Notes

- The `.tex` source uses **semantic linefeeds** (one sentence / clause per line)
  for diff-friendliness. Line breaks are treated as spaces by LaTeX,
  so PDF output is unaffected.
- Citations use author-year keys (e.g. `\cite{guo2017calibration}`)
  resolved from `ISMIRtemplate.bib` via `\bibliographystyle{IEEEtran}`.
- Placeholder figures (e.g. `\ref{fig:calibration}`) will emit `??`
  until the corresponding `\label{...}` is added.
