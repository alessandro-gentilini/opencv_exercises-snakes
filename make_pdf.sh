#!/bin/sh
rm snakes.aux
rm snakes.bbl
rm snakes.blg
rm snakes.out
rm snakes.log

pdflatex snakes.tex
bibtex snakes.aux
pdflatex snakes.tex
pdflatex snakes.tex

rm snakes.aux
rm snakes.bbl
rm snakes.blg
rm snakes.out
rm snakes.log
