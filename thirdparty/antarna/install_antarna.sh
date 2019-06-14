wget --directory-prefix=src http://www.bioinf.uni-freiburg.de//Software/antaRNA/antaRNA_v114.py
cd src
awk 'NR==1320 {$0="# LINES DELETED FOR RUNNING ON NEMO CLUSTER TIMEOUT SCRIPT"} 1' antaRNA_v114.py >> intermediate.py
awk 'NR==1321 {$0="# "} 1' intermediate.py >> final.py
# awk 'NR==1321 {$0="# "} 1' intermediate2.py >> final.py
#
rm -f antaRNA_v114.py
rm -f intermediate.py
# rm -f intermediate2.py
mv final.py antaRNA_v114.py
cd ..
source activate antarna
pip install -r requirements.txt
