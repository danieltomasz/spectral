.ONESHELL:

HOME_DIR = /Users/daniel/PhD/Projects/ephys-poirot-analysis
ROOT_DIR = ${HOME_DIR}/poirot/studies/ASSR


TABLE = ${ROOT_DIR}/data/table_exp_S.txt
RESULTS = ${ROOT_DIR}/results/results_exp_S


rba:
	tcsh -c "nohup \
	RBA \
	-ridgePlot 10 8 \
	-prefix ${RESULTS} \
	-chains 4 \
	-iterations 1000 \
	-mean 'y~1+(1|ROI)+(1|Subj)' \
	-sigma '1+(1|ROI)+(1|Subj)' \
	-ROI 'ROI' \
	-EOI 'Intercept' \
	-dataTable ${TABLE}\
	|& tee diary.txt &"

clean:
	rm -f ${RESULTS}*
	rm -f diary.txt