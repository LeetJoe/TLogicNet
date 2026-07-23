#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# conda env
#conda_path=$(conda info --base)
#source "${conda_path}"/etc/profile.d/conda.sh
#conda activate test

threads='60'
limit_threads='10'

walk_num='200'
rule_length='1 2 3'

min_rule_conf="0.1"
min_rule_body_support="2"

iterations='1000'
learn_rate='0.0001'
start_time_gap='1'
event_threshold='-1'
infer_time_gap='7'
score_win='90'
min_path_len='5'
infer_time_range='-1'

rule_batch='500'
jvm_min_heap='64G'
jvm_max_heap='128G'

code_base='.'
dataset='icews14'
# in case data folder was in different name of the dataset
dataset_folder="${dataset}"

data_init="${code_base}"/data/"${dataset}"_init
data_base="${code_base}"/data/"${dataset_folder}"

## eceformer specific
infer_config='kge/eceformer/'"${dataset}"'-infer.yaml'
#infer_config="${data_base}"'/origin/'"${dataset}"'-infer.yaml'
echo 'Using dataset '"${dataset}"' in path '"${data_base}"' and eceformer config '"${infer_config}"'...'







if [ "${dataset}" = "icews14" ]; then
  is_interval='0'

  rule_batch='4000'

  walk_num='200'
  rule_length='1 2 3'

  min_rule_conf="0.1"
  min_rule_body_support="2"
  random_ratio="1.0"
  start_time_gap='1'
  min_path_len='5'

  iterations='1000'
  learn_rate='0.0001'

  data_size='81767'
  found_thresh='0.9'
  valid_thresh='0.85'
  infer_topk=$(( data_size * 2 ))
elif [ "${dataset}" = "icews14_to" ]; then
  is_interval='0'

  rule_batch='2000'

  walk_num='200'
  rule_length='1 2 3'

  min_rule_conf="0.1"
  min_rule_body_support="2"
  random_ratio="1.0"
  start_time_gap='1'
  min_path_len='5'
  infer_time_range='261'

  iterations='1000'
  learn_rate='0.0001'

  data_size='77508'
  found_thresh='0.9'
  valid_thresh='0.85'
  infer_topk=$(( data_size * 2 ))
elif [ "${dataset}" = "icews0515" ]; then
  is_interval='0'
  limit_threads='8'

  rule_batch='2000'
  event_threshold='0'

  jvm_min_heap='128G'
  jvm_max_heap='420G'

  walk_num='200'
  rule_length='1 2 3'

  min_rule_conf="0.3"
  min_rule_body_support="3"
  random_ratio="1.0"
  start_time_gap='30'
  infer_time_gap='30'
  min_path_len='30'

  iterations='1000'
  learn_rate='0.0001'

  data_size='415237'
  found_thresh='0.9'
  valid_thresh='0.8'
  infer_topk=$(( data_size * 2 ))
elif [ "${dataset}" = "icews0515_to" ]; then
  is_interval='0'
  limit_threads='8'

  rule_batch='2000'
  event_threshold='0'

  jvm_min_heap='128G'
  jvm_max_heap='420G'

  walk_num='200'
  rule_length='1 2 3'

  min_rule_conf="0.3"
  min_rule_body_support="3"
  random_ratio="1.0"
  start_time_gap='30'
  infer_time_gap='30'
  min_path_len='30'
  infer_time_range='2774'

  iterations='1000'
  learn_rate='0.0001'

  data_size='415237'
  found_thresh='0.9'
  valid_thresh='0.8'
  infer_topk=$(( data_size * 2 ))
elif [ "${dataset}" = "icews18" ]; then
  is_interval='0'
  limit_threads='2'

  rule_batch='500'
  event_threshold='0'

  jvm_min_heap='128G'
  jvm_max_heap='420G'

  walk_num='200'
  rule_length='1 2 3'

  min_rule_conf="0.2"
  min_rule_body_support="3"
  random_ratio="1.0"
  start_time_gap='30'
  infer_time_gap='30'
  min_path_len='30'
  infer_time_range='240'

  iterations='1000'
  learn_rate='0.0001'

  data_size='415237'
  found_thresh='0.9'
  valid_thresh='0.8'
  infer_topk=$(( data_size * 2 ))
elif [ "${dataset}" = "wikidata12k" ]; then
  is_interval='1'
  walk_num='10000'
  rule_length='1 2 3'

  min_rule_conf="0.01"
  min_rule_body_support="2"
  random_ratio="1.0"

  iterations='1000'
  learn_rate='0.0001'

  data_size='53782'
  found_thresh='0.9'
  valid_thresh='0.85'
  infer_topk=$(( data_size * 2 ))
else
  echo 'Unknown dataset '"${dataset}"'!'
  exit 1
fi





if [[ ! -e "${data_base}" ]]; then
  cp -r "${data_init}" "${data_base}"
fi




expfolder='tece_main'
exp_base="${data_base}"/"${expfolder}"

if [[ ! -e "${exp_base}" ]]; then
  mkdir -p "${exp_base}"
  python data_init.py "${data_base}" "${is_interval}"
  cp "${0}" "${exp_base}"
  sed -i "2s/.*/dataset.name: ${dataset_folder}/" "${infer_config}"
  cp "${infer_config}" "${exp_base}"
fi

#cp "${data_base}"/xxxxxxx/set_seed.sh "${exp_base}"/set_seed.sh
kge_config="${exp_base}"'/'"${dataset}"'-infer.yaml'


max_iter=6
i=0

if [[ i -eq 0 ]]; then
  python gen_seed.py "${data_base}"
  cp "${data_base}"/set_seed.sh "${exp_base}"/set_seed.sh
  source "${exp_base}"/set_seed.sh
  python mln/gen_tlogic_rules.py -d "${data_base}" -p "${threads}" -n "${walk_num}" -mc "${min_rule_conf}" -l ${rule_length} -s "${MLN_SEED_PYTHON}"
  cp "${data_base}"/tlogic_rules_"${walk_num}".txt "${exp_base}"
else
  source "${exp_base}"/set_seed.sh
fi

while [ $i -le $max_iter ]
do

  echo 'Start iteration '"${i}"

  mln_base="${exp_base}"/"${i}"/mln
  mkdir -p "${mln_base}"
  kge_base="${exp_base}"/"${i}"/kge

  if [[ i -eq 0 ]]; then
    java -Xms"${jvm_min_heap}" -Xmx"${jvm_max_heap}" -jar mln/mln_test.jar -rr "${random_ratio}" -rb "${rule_batch}" -et "${event_threshold}" \
    -t ${threads} -tg "${start_time_gap}" -d "${data_base}" -r "${exp_base}"/tlogic_rules_"${walk_num}".txt -s "${MLN_SEED_JAVA}" \
    -tr "${min_rule_conf}" -ms "${min_rule_body_support}" -i "${iterations}" -lr "${learn_rate}" -sr "${mln_base}"/rules.txt \
    -sp "${mln_base}"/pred_hidden.txt -lt "${mln_base}"/test_linked_batch.txt -an "${mln_base}"/annotation.txt \
    -sh "${mln_base}"/hiddenids.txt
  else
    i_pre=$(( i - 1 ))
    python mln/filter_annotation.py -o "${exp_base}"/"${i_pre}"/kge/annotation.txt  -s "${mln_base}"/annotation.txt -t "${valid_thresh}"

    java -Xms"${jvm_min_heap}" -Xmx"${jvm_max_heap}" -jar mln/mln_test.jar -rr "${random_ratio}" -rb "${rule_batch}" -et "${event_threshold}" \
    -t ${threads} -tg "${start_time_gap}" -d "${data_base}" -r "${exp_base}"/tlogic_rules_"${walk_num}".txt -s "${MLN_SEED_JAVA}" \
    -tr "${min_rule_conf}" -ms "${min_rule_body_support}" -i "${iterations}" -lr "${learn_rate}" -sr "${mln_base}"/rules.txt \
    -sp "${mln_base}"/pred_hidden.txt -lt "${mln_base}"/test_linked_batch.txt -an "${mln_base}"/annotation.txt

    cp "${exp_base}"/0/mln/hiddenids.txt "${mln_base}"/hiddenids.txt
  fi

  python mln/merge_test_link.py --stat_file "${data_base}"/stat.txt --file_path "${mln_base}"/test_linked_batch.txt --save_path "${mln_base}"/test_linked.txt
  python mln/tlogic_infer.py --stat "${data_base}"/stat.txt --test "${data_base}"/testids.txt --ltest "${mln_base}"/test_linked.txt \
  --rule "${mln_base}"/rules.txt --save "${mln_base}"/pred_test.txt --threads ${threads}

  python mln/logic_eval.py --data_path "${data_base}" --prediction "${mln_base}"/pred_test.txt --save "${mln_base}"/result_mln.txt --threads ${threads}

  python mln/infer_found_main.py -d "${data_base}" -m "${mln_base}" -ml "${min_path_len}" -tg "${infer_time_gap}" -t "${infer_topk}" -th "${found_thresh}" -td "${limit_threads}" -sw "${score_win}" -tl "${infer_time_range}"

  if [[ i -eq 0 ]]; then
    echo -n > "${data_base}"/foundids.txt
  else
    cp "${mln_base}"/foundids.txt "${data_base}"/foundids.txt
  fi

  ## eceformer specific
  python kge/eceformer/preprocess.py --dataset "${dataset}" --datafolder "${dataset_folder}"
  cp "${data_base}"/dataset.yaml "${exp_base}"/"${i}"

  python -m kge start "${kge_config}" --folder "${kge_base}"
#   python -m kge resume "${kge_base}" --checkpoint "${kge_base}"/checkpoint_00344.pt

  python -m kge test "${kge_base}"
  python em_eval.py --data_path "${data_base}" --pred_kge "${kge_base}"/pred_kge.txt --pred_mln "${mln_base}"/pred_test.txt \
  --save_ranks "${exp_base}"/"${i}"/ranks.txt --save_result "${exp_base}"/"${i}"/result_em.txt

  cp "${mln_base}"/inferids.txt "${data_base}"
  python -m kge infer "${kge_base}"

  i=$(( i + 1 ))
done

# get result
sh get_result.sh "${dataset}" "${expfolder}" "${max_iter}" > "${exp_base}"/result_final.txt

