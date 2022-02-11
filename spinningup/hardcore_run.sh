#!/bin/csh
set env_name=hardcore
set envir=BipedalWalkerHardcore-v3
set BETA_REF=100 # reward amplification
set k=50 # explo/exploit balance
set PREC=$k/100
set do_reward=1
set gamma=0.99
set bandwidth=0.1
set start_steps=10000
set steps_per_epoch=4000
set update_after=10000
set update_every=50
set epochs=500
set replay_size=1000000 #int(1e6)
set batch_size=100
set max_ep_len=1600
set hid=256
#nice python3 -m spinup.run macao --seed 0 1 2 3 4 --hid "[$hid,$hid]" --env $envir --exp_name walker_macao_test2 --beta $BETA_REF --prec $PREC --gamma $gamma --bandwidth $bandwidth --epochs $epochs --steps_per_epoch $steps_per_epoch --start_steps $start_steps  --update_after $update_after --update_every $update_every --replay_size $replay_size --batch_size $batch_size --max_ep_len $max_ep_len&
nice python3 -m spinup.run macao --algo "sac" --seed 0 1 2 3 4 --hid "[$hid,$hid]" --env $envir --exp_name walker_sac_test --beta $BETA_REF --prec $PREC --gamma $gamma --bandwidth $bandwidth --epochs $epochs --steps_per_epoch $steps_per_epoch --start_steps $start_steps  --update_after $update_after --update_every $update_every --replay_size $replay_size --batch_size $batch_size --max_ep_len $max_ep_len&
nice python3 -m spinup.run ppo --seed 0 1 2 3 4 --hid "[$hid,$hid]" --env $envir --exp_name walker_ppo_test --gamma $gamma --epochs $epochs --steps_per_epoch $steps_per_epoch --max_ep_len $max_ep_len&
nice python3 -m spinup.run td3 --seed 0 1 2 3 4 --hid "[$hid,$hid]" --env $envir --exp_name walker_td3_test --gamma $gamma --epochs $epochs --steps_per_epoch $steps_per_epoch --update_after $update_after --start_steps $start_steps  $steps_per_epoch --update_every $update_every --replay_size $replay_size --batch_size $batch_size --max_ep_len $max_ep_len&
