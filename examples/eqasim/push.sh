mkdir -p data

scp hcm:/home/sebastian/work/scenarios/output/pc_1pm_egt/*.xml data
scp hcm:/home/sebastian/work/scenarios/output/pc_1pm_egt/*.xml.gz data

cp /home/shoerl/code/eqasim-java/ile_de_france/target/ile_de_france-1.3.1.jar data

cp /home/shoerl/explo22/egt_pc_trips.csv data

scp -R data bullx:/scratch/sebastian.horl/boptx/examples/eqasim
