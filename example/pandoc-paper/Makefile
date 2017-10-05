BIB=~/Dropbox/Papers/MultiscaleSAM.bib
OPTS= -V biblio-style=authoryear --filter pandoc-crossref
 
outline.tex : outline.md
	pandoc -s $(OPTS) -V bibliography=$(BIB) --biblatex -o $@ $^

%.pdf: %.tex
	pdflatex $^
	biber $*
	pdflatex $^
	pdflatex $^

clean:
	rm -f outline.tex outline.pdf
