#!/bin/bash

chemicals=( "phosphorus" "ph" "exchangeable_acidity" "calcium" "magnesium" "sulphur" "sodium" "iron" "manganese" "boron" "copper" "zinc" "total_nitrogen" "potassium" "ec_salts" "organic_carbon" "cec" "sand" "silt" "clay")


for chemical in "${chemicals[@]}"
do
    echo "Splitting data for $chemical..."
    Rscript /home/tom/DSML124/splits.r "$chemical"
done