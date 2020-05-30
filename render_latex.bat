@RD /S /Q "./pythontex-files-pdi"
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf pdi.tex
bibtex -include-directory=C:/Users/kgb/Documents/python_projects/liar_liar/latex pdi
pythontex pdi.tex
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf pdi.tex