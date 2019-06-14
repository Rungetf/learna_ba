git clone https://github.com/tsudalab/MCTS-RNA.git src
cd src
awk 'NR==177 {$0="    out_time=running_time+2147483647"} 1' MCTS-RNA.py >> mcts_intermediate.py
rm MCTS-RNA.py
mv mcts_intermediate.py MCTS-RNA.py
