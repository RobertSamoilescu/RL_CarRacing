# python3 -m scripts.train --algo ppo --model CarRacingDQNStacked4.1 --save-interval 5 --procs 12 --tb 1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacingGRU --save-interval 10 --procs 8 --tb --recurrence 32 1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacingGRU256 --save-interval 10 --procs 8 --tb --recurrence 64
# python3 -m scripts.train --algo ppo --model CarRacingGRU_RAND_CCLRN --save-interval 10 --procs 8 --tb --recurrence 2 --batch-size 512 --epochs 2 --lr 1e-4 1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacingGRU_RAND_CCLRN_Rec4 --save-interval 10 --procs 16 --tb --recurrence 4 --batch-size 1024 --epochs 2 --lr 1e-4 --clip-eps 0.1 --frames-per-proc 128 1>out 2>err


# baseline model, domain randomization architecture https://arxiv.org/pdf/1710.06537.pdf, easiest environment 
# python3 -m scripts.train --algo ppo --model CarRacing_Baseline_1 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4  --seed 1 1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacing_Baseline_9071 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4 --seed 9072 1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacing_Baseline_7309 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4 --seed 7309 1>out 2>err

# randomized envirnomnet from the start
# python3 -m scripts.train --algo ppo --model CarRacing_RandDomain_1 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4 --seed 1  1>out 2>err
# python3 -m scripts.train --algo ppo --model CarRacing_Baseline_9072 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4 --seed 9072 1>out_9071 2>err_9071
# python3 -m scripts.train --algo ppo --model CarRacing_Baseline_7309 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4 --seed 7309 1>out_7309 2>err_7309

# symmetric actor critic
python3 -m scripts.train --algo ppo --model CarRacing_Symmetric_CL_1 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4  --seed 1 1>out 2>err
python3 -m scripts.train --algo ppo --model CarRacing_Symmetric_CL_9072 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4 --seed 9072 1>out_9072 2>err_9072
python3 -m scripts.train --algo ppo --model CarRacing_Symmetric_CL_7309 --save-interval 5 --procs 16 --tb --recurrence 8 --batch-size 1024 --epochs 2 --lr 7e-4 --seed 7309 1>out_7309 2>err_7309
