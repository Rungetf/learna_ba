wget -O source.zip https://doi.org/10.1371/journal.pcbi.1006176.s002
unzip source.zip
mv SI src
rm source.zip
mv src/best_model .
cd src
awk 'NR==17 {$0="end_time = start_time + 24*60*60*2147483647"} 1' solve_one_puzzle.py >> intermediate.py
rm -f solve_one_puzzle.py
mv intermediate.py solve_one_puzzle.py
