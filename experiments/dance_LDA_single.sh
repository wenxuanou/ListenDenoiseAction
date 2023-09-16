#!/bin/bash

checkpoint=pretrained_models/dance_LDA.ckpt
dest_dir=results/generated/dance_LDA

if [ ! -d "${dest_dir}" ]; then
    mkdir -p "${dest_dir}"
fi

data_dir=data/my_music      # data/motorica_dance
wav_dir=data/my_music       # data/motorica_dance
basenames=$(cat "${data_dir}/gen_files.txt")

start_s=30
seed=50                    # 150
fps=30
trim_s=0
length_s=70                 # length in seconds
start=$((start_s*fps))
trim=$((trim_s*fps))
length=$((length_s*fps))    # number of samples
fixed_seed=false
gpu="cuda:0"
render_video=true

for wavfile in $basenames; 
do
	style=$(echo $wavfile | awk -F "_" '{print $2}') #Coherent style parsed from file-name
	postfix="single"

  input_file=${wavfile}.audio29_${fps}fps.pkl

  output_file=${wavfile::-3}_${postfix}_${style}

  echo "start=${start}, len=${length}, postfix=${postfix}, seed=${seed}"
  python synthesize.py --checkpoints="${checkpoint}" --data_dirs="${data_dir}" --input_files="${input_file}" --styles="${style}" --start=${start} --end=${length} --seed=${seed} --postfix=${postfix} --trim=${trim} --dest_dir=${dest_dir} --gpu=${gpu} --video=${render_video} --outfile=${output_file}
  if [ "$fixed_seed" != "true" ]; then
    seed=$((seed+1))
  fi
  echo seed=$seed
  python utils/cut_wav.py ${wav_dir}/${wavfile::-3}.wav $(((start+trim)/fps)) $(((start+length-trim)/fps)) ${postfix} ${dest_dir}
  if [ "$render_video" == "true" ]; then
    ffmpeg -y -i ${dest_dir}/${output_file}.mp4 -i ${dest_dir}/${wavfile::-3}_${postfix}.wav ${dest_dir}/${output_file}_audio.mp4
    rm ${dest_dir}/${output_file}.mp4
  fi

done
