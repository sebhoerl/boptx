git clone git@github.com:eqasim-org/eqasim-java.git --branch convergence --single-branch --depth 1
cd eqasim-java
mvn -Pstandalone --projects ile_de_france --also-make package -DskipTests=true
mkdir ../data
mv ile_de_france/target/ile_de_france-1.3.1.jar ../data/eqasim.jar
