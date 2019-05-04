# python3 -m scripts.train --algo ppo --model CarRacingDQNStacked4.1 --save-interval 5 --procs 12 --tb 1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacingGRU --save-interval 10 --procs 8 --tb --recurrence 32 1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacingGRU256 --save-interval 10 --procs 8 --tb --recurrence 64
#python3 -m scripts.train --algo ppo --model CarRacingGRU_RAND_CCLRN --save-interval 10 --procs 8 --tb --recurrence 2 --batch-size 512 --epochs 2 --lr 1e-4 1>out 2>err

python3 -m scripts.train --algo ppo --model CarRacingGRU_RAND_CCLRN_Rec4 --save-interval 10 --procs 16 --tb --recurrence 4 --batch-size 1024 --epochs 2 --lr 1e-4 --clip-eps 0.1 --frames-per-proc 128 1>out 2>err
