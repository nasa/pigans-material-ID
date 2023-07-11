#!/bin/bash
# Runs Monte Carlo simulation to calculate KI using material fields.
# Specify directory containing material sample csvs as first command line arg.

module load ABAQUS_2021

curr_dir=`pwd`
temp_dir=tmp_`date +"%m-%d-%y-%s"`
sample_dir=`realpath $1`
output_file=output_`date +"%m-%d-%y-%s.txt"`
script=$curr_dir"/run_sample.py"

mkdir $temp_dir
cd $temp_dir

for filename in `ls $sample_dir/*.csv`; do
abaqus cae noGUI=$script -- $filename >> $output_file
done

mv $output_file $sample_dir
cd $curr_dir
rm -rf $temp_dir
