# OsCAR
OSCAR: Orthogonal Subspace Correction and Rectification of Biases inWord Embeddings


## Prerequisites
In addition to the packages in ``requirements.txt``, pleast also install nvidia-apex from [here](https://github.com/NVIDIA/apex).

Download snli_1.0 data and make the txt files located at ``./data/snli_1.0/``.

The evaluation datasets are located at ``./data/oscar.zip``. Make those files located at ``./data/oscar/``.

## Preprocess
You can process SNLI dataset by the following:
```
python3 -u -m preprocess.preprocess
```

And process our evaluation datasets by:
```
python3 -u -m preprocess.preprocess_bias --data entail_templates.txt --gold_label entailment --output entail
python3 -u -m preprocess.preprocess_bias --data contradict_templates.txt --gold_label contradiction --output contradict
python3 -u -m preprocess.preprocess_bias --data gender_occupation_templates.txt --gold_label neutral --output gender_occupation
```

By default, preprocessing assumes you are using ``roberta-base``.
To use other type of transformers, use the ``--transformer_type`` option to customize it.

Lastly, reserve a directory for models and logs at ``./models/``.

## Training vanilla NLI models

You can train vanilla NLI models without using any bias-mitigation techniques.
```
GPUID=[GPUID]
for SEED in 1 2 3; do
	python3 -u train.py --gpuid $GPUID --seed $SEED \
		--save_file ./models/robertabase_snli_seed${SEED}_default | tee ./models/robertabase_snli_seed${SEED}_default.txt
done
```

To evaluate on the official SNLI test set, run:
```
for SEED in 1 2 3; do
    python3 -u predict.py --gpuid $GPUID \
    	--load_file ./models/robertabase_snli_seed${SEED}_default | tee ./models/robertabase_snli_seed${SEED}_default.test.txt
done
```

To evaluate on our NLI bias datasets, run:
```
for SEED in 1 2 3; do
for DATA in entail contradict gender_occupation; do
    python3 -u predict.py --gpuid $GPUID --dir ./data/oscar/ --data ${DATA}.hdf5 \
    	--res ${DATA}.x_pair.txt,${DATA}.sent1.txt,${DATA}.sent2.txt --ref_log unlabeled \
        --load_file ./models/robertabase_snli_seed${SEED}_default --pred_output ./models/${DATA}.robertabase_snli_seed${SEED}_default
done
done
```

## Training NLI models with projective debiasing

You can train NLI models with projective debiasing via:
```
GPUID=[GPUID]
for SEED in 1 2 3; do
	python3 -u train.py --gpuid $GPUID --enc proj_transformer --seed $SEED \
    	--save_file ./models/robertabase_snli_proj_seed${SEED} | tee ./models/robertabase_snli_proj_seed${SEED}.txt
done
```

To evaluate on the official SNLI test set, run:
```
for SEED in 1 2 3; do
    python3 -u predict.py --gpuid $GPUID \
    	--load_file ./models/robertabase_snli_proj_seed${SEED} | tee ./models/robertabase_snli_proj_seed${SEED}.test.txt
done
```

To evaluate on our NLI bias datasets, run:
```
for SEED in 1 2 3; do
for DATA in entail contradict gender_occupation; do
    python3 -u predict.py --gpuid $GPUID --dir ./data/oscar/ --data ${DATA}.hdf5 \
    	--res ${DATA}.x_pair.txt,${DATA}.sent1.txt,${DATA}.sent2.txt --ref_log unlabeled \
        --load_file ./models/robertabase_snli_proj_seed${SEED} --pred_output ./models/${DATA}.robertabase_snli_proj_seed${SEED}
done
done
```

## Training NLI models with OsCAR

You can train NLI models with rotational debiasing (OsCAR) via:
```
GPUID=[GPUID]
for SEED in 1 2 3; do
	python3 -u train.py --gpuid $GPUID --enc oscar_transformer --seed $SEED \
    	--save_file ./models/robertabase_snli_oscar_seed${SEED} | tee ./models/robertabase_snli_oscar_seed${SEED}.txt
done
```

To evaluate on the official SNLI test set, run:
```
for SEED in 1 2 3; do
    python3 -u predict.py --gpuid $GPUID \
    	--load_file ./models/robertabase_snli_oscar_seed${SEED} | tee ./models/robertabase_snli_oscar_seed${SEED}.test.txt
done
```

To evaluate on our NLI bias datasets, run:
```
for SEED in 1 2 3; do
for DATA in entail contradict gender_occupation; do
    python3 -u predict.py --gpuid $GPUID --dir ./data/oscar/ --data ${DATA}.hdf5 \
    	--res ${DATA}.x_pair.txt,${DATA}.sent1.txt,${DATA}.sent2.txt --ref_log unlabeled \
        --load_file ./models/robertabase_snli_oscar_seed${SEED} --pred_output ./models/${DATA}.robertabase_snli_oscar_seed${SEED}
done
done
```

## Training with debiased embeddings, e.g. Hard Debiasing (HD), INLP\*

We have uploaded preprocessed embedding files for HD and INLP\*.
Simply unzip ``./data/roberta_emb_hd.zip`` and ``./data/roberta_emb_iter_filtered.zip``, and make the ``.txt`` files located at ``./data/``.



Suppose we are on a NLI model with HD. The training follows:
```
for SEED in 1 2 3; do
python3 -u train.py --gpuid 0 --seed $SEED \
    --emb_overwrite ./data/roberta_emb_hd.txt \
    --save_file ./models/robertabase_snli_hd_seed${SEED} | tee ./models/robertabase_snli_hd_seed${SEED}.txt
done
```

And evaluation (on SNLI test set) follows:
```
for SEED in 1 2 3; do
    python3 -u predict.py --gpuid $GPUID \
    	--load_file ./models/robertabase_snli_hd_seed${SEED} | tee ./models/robertabase_snli_hd_seed${SEED}.test.txt
done
```

Similarly, the evaluation on bias probing datasets follows:
```
for SEED in 1 2 3; do
for DATA in entail contradict gender_occupation; do
    python3 -u predict.py --gpuid $GPUID --dir ./data/oscar/ --data ${DATA}.hdf5 \
    	--res ${DATA}.x_pair.txt,${DATA}.sent1.txt,${DATA}.sent2.txt --ref_log unlabeled \
        --load_file ./models/robertabase_snli_hd_seed${SEED} --pred_output ./models/${DATA}.robertabase_snli_hd_seed${SEED}
done
done
```
