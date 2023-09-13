#!/bin/env perl

# ======================================================================================
# Perl `latexmk` configuration file
# ======================================================================================

# PDF-generating modes are:
# 1: pdflatex, as specified by $pdflatex variable (still largely in use)
# 2: postscript conversion, as specified by the $ps2pdf variable (useless)
# 3: dvi conversion, as specified by the $dvipdf variable (useless)
# 4: lualatex, as specified by the $lualatex variable (best)
# 5: xelatex, as specified by the $xelatex variable (second best)
$pdf_mode = 5;

$warnings_as_errors = 0;

# Show used CPU time. Looks like: https://tex.stackexchange.com/a/312224/120853
$show_time = 0;

$max_repeat=7;

# `set_tex_cmds` applies to all *latex commands (latex, xelatex, lualatex, ...), so
# no need to specify these each. This allows to simply change `$pdf_mode` to get a
# different engine. Check if this works with `latexmk --commands`.
set_tex_cmds("--shell-escape --synctex=1 -8bit %O %S");

$bibtex_use = 2;  # default: 1

# Change default `biber` call, help catch errors faster/clearer. See
# https://web.archive.org/web/20200526101657/https://www.semipol.de/2018/06/12/latex-best-practices.html#database-entries
$biber = "biber --validate-datamodel %O %S";


$deps_out = ".cache/latex/dependencies.list";

$tmpdir = ".cache/latex/tmp";
