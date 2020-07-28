#!/bin/bash

# FILEs paths
PROJECTDIR="../../data/schneider50k"
DATADIR=${PROJECTDIR}
model_folder="${PROJECTDIR}/training"
test_file="${DATADIR}/precursors-train.txt"
target_file="${DATADIR}/product-train.txt"
output_folder="${PROJECTDIR}/translation/train"

# SIZE of the translation batcha
BATCH_SIZE=64
# FORWARD BEAM to generate
FORWARD_BEAM=10
# Number of likelihood sorted predictions to output
TOPN=1

echo "Launching the following jobs:"
echo "Model folder: ${model_folder}"
echo "Test file: ${test_file}"
echo "Output folder: ${output_folder}"
mkdir -p $output_folder

for filename in ${model_folder}/*.pt ; do # translation of the last step as well!
echo "FILENAME: $filename"
OUTFILE=$(basename "$filename" .pt)_on_$(basename "$test_file")
MODEL=$filename
echo $OUTFILE

# Perform inference on the given data
onmt_translate -model ${filename} -src ${test_file} -tgt ${target_file} -output ${output_folder}/${OUTFILE}.out.txt -log_probs -n_best $TOPN -beam_size $FORWARD_BEAM -max_length 300 -batch_size $BATCH_SIZE -gpu 0

# Canonicalize the output
python ../../onmt_utils/computation_utils/smiles_utils.py ${output_folder}/${OUTFILE}.out.txt

done
