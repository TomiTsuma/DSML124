#!/bin/bash
sudo apt-get install libgdal-dev libgeos-dev libproj-dev

wget https://cran.r-project.org/src/contrib/clhs_0.9.0.tar.gz
wget https://cran.r-project.org/src/contrib/Archive/terra/terra_1.7-28.tar.gz
wget https://cran.r-project.org/src/contrib/Archive/sf/sf_1.0-0.tar.gz
wget https://cran.r-project.org/src/contrib/lattice_0.22-6.tar.gz

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
sudo apt update --assume-yes
sudo apt install -y build-essential libcurl4-openssl-dev libxml2-dev libssl-dev libfontconfig1-dev libreadline-dev libbz2-dev liblzma-dev libpcre2-dev
sudo apt-get install -y libx11-dev libxt-dev

sudo apt-get install default-jdk --asume-yes
java -version
update-alternatives --config java
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
export PATH=$JAVA_HOME/bin:$PATH

sudo apt-get install gfortran
curl -O https://cran.r-project.org/src/base/R-4/R-4.4.0.tar.gz
tar -zxvf R-4.4.0.tar.gz
cd R-4.4.0
./configure --enable-R-shlib --with-blas --with-lapack
make
sudo make install
cd ..
# wget https://download.osgeo.org/gdal/3.3.0/gdal-3.3.0.tar.gz
# tar -xzf gdal-3.3.0.tar.gz
# cd gdal-3.3.0
# ./configure
# make -j$(nproc)
# sudo make install
# sudo ldconfig

R --version
# sudo add-apt-repository ppa:ubuntugis/ubuntugis-unstable
# sudo apt-get update
# sudo apt-get install gdal-bin libgdal-dev
# gdalinfo --version
wait 

sudo apt install libudunits2-dev --assume-yes

wait
# sudo apt install libgdal-dev --assume-yes
# sudo apt install r-base-core --assume-yes



Rscript -e 'install.packages("prospectr",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("sp",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("raster",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("reshape2",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("plyr",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("ggplot2",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("stringr",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("classInt",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("DBI",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("s2",repos = "http://cran.us.r-project.org")' --assume-yes
Rscript -e 'install.packages("units",repos = "http://cran.us.r-project.org")' --assume-yes



Rscript -e 'install.packages("data.table", dependencies=TRUE,repos = "http://cran.us.r-project.org")' --assume-yes
sudo Rscript -e 'install.packages("/home/tom/DSML125/sf_1.0-0.tar.gz", repos = NULL, type = "source")' --assume-yes
Rscript -e 'install.packages("/home/tom/DSML125/clhs_0.9.0.tar.gz", repos = NULL, type = "source")' --assume-yes
# Rscript -e 'install.packages("/home/tom/DSML125/terra_1.7-28.tar.gz", repos = NULL, type = "source")' --assume-yes

Rscript -e 'install.packages("terra",repos = "http://cran.us.r-project.org")' --assume-yes
