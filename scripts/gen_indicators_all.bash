cd ../prediction
for i in ../data_in/*.CSV; do python gen_indicators.py -o ../da
ta_out/$(basename $i) $i; done
