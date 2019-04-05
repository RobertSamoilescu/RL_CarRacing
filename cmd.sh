# python3 -m scripts.train --algo ppo --model CarRacingDQNStacked4 --save-interval 50 --procs 12 --tb 1>out 2>err
python3 -m scripts.train --algo ppo --model CarRacing --save-interval 10 --procs 1 --tb --recurrence 64
