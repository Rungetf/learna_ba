#!/bin/bash
echo "Create target directory rfam_raw"
mkdir -p data/rfam_raw
cd data/rfam_raw
echo "Remove old database files"
rm -rf *
echo "Start downloading database"
wget -r ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files -P .
# echo "Move files to upper level"
# mv ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/* .
# echo "Remove subfolders"
# rm -r ftp.ebi.ac.uk/
echo "Start unpacking database files"
gunzip -rfk .
echo "Remove gzipped files"
rm -rf *.gz
