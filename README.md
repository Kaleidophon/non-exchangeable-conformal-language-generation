# Non-Exchangeable Conformal Language Generation with Nearest Neighbors

This is the Github repository for the paper EACL 2024 Findings paper of the same name by Dennis Ulmer, Chrysoula Zerva and 
André F.T. Martins [Link](https://arxiv.org/pdf/2402.00707.pdf).

## Installation

Install the necessary requirements the following way:

    pip3 install -r requirements.txt

This repository also requires the [FAISS](https://github.com/facebookresearch/faiss) library. 
Depending on the hardware available, install either

    pip3 install faiss-cpu

or 

    pip3 install faiss-gpu

## Usage

Replicating the experiments in the paper requires running the following steps.
First of all, in the case of machine translation experiments, download the corresponding data files [here](https://www.statmt.org/wmt22/translation-task.html).
To prepare for the experiments, create datastores using the following command:

    python3 create_datastore.py --save-dir datastores/deen_m2m100_418M_l2 --dataset deen --device cuda --num-probes 32 --num-centroids 2048 --use-quantization --model facebook/m2m100_418M --batch-size 4 --distance-type l2

For the Japanese-English dataset, specify `--dataset jaen` instead, and use `--model facebook/m2m100_1.2B --sharding 1 2 3` instead (1, 2, 3 here indicating the indices of GPUs to use).
Similarly for text generation experiments, run

    python3 create_datastore.py --save-dir datastores/openwebtext_opt_350M_l2 --dataset openwebtext --device cuda --num-probes 32 --num-centroids 2048 --use-quantization --model facebook/opt-350m --distance-type l2

and replace the model identifier by `facebook/opt-1.3B` for the larger OPT model.

From there, run the following scripts to replicate the main results of the paper
(we will only show the results for the smaller models and the de->en task from here, to reproduce the other results use the same
argument substitutions as used above).
For the coverage results in section 4.1, run

    python3 run_coverage_experiment.py --datastore-dir results/deen_m2m100_418M_l2 --result-dir results/deen_m2m100_418M_l2 --dataset deen --device cuda --num-probes 1024 --num-neighbors 100 --num-centroids 2048 --temperature 512.1416 --use-quantization --distance-type l2
    python3 run_coverage_experiment.py --datastore-dir results/openwebtext_opt_350m_l2 --result-dir results/results/openwebtext_opt_350m_l2 --dataset openwebtext --device cuda --num-probes 32 --num-neighbors 100 --num-centroids 2048 --temperature 15538.91 --use-quantization --model facebook/opt-350m --distance-type l2

For the distributional shift results in section 4.2, run

    python3 run_shift_coverage_experiment.py \
        --method non_exchangeable_conformal_nucleus_sampling --alpha 0.1 \
        --datastore-dir results/deen_m2m100_418M_l2 \
        --result-dir results/shift_coverage \
        --dataset deen --device cuda\
        --num-probes 32 --num-neighbors 100 --num-centroids 2048 \
        --temperature 512.14  --use-quantization --distance-type l2

    python3 run_shift_coverage_experiment.py \
        --method non_exchangeable_conformal_nucleus_sampling --alpha 0.1\
        --datastore-dir results/openwebtext_opt_350m_l2 \
        --result-dir results/shift_coverage\
        --dataset openwebtext --device cuda\ 
        --model-identifier facebook/opt-350m\
        --num-probes 32 --num-neighbors 100 --num-centroids 2048 \
         --temperature 15538.91 --use-quantization --distance-type l2

For the generation results in section 4.3, run

    python3 evaluate_generation.py \
        --generation-method non_exchangeable_nucleus_sampling --alpha 0.1 \
        --datastore-dir results/deen_m2m100_418M_l2 \
        --result-dir results/deen_m2m100_418M_l2 \
        --dataset deen \
        --device cuda --num-samples 5 --softmax-temperature 0.1 \
        --num-probes 32 --num-neighbors 100 --num-centroids 2048 \
        --temperature 512.14  --use-quantization --distance-type l2

    python3 evaluate_generation.py \
        --generation-method non_exchangeable_nucleus_sampling --alpha 0.1 --num-samples 5 \
        --datastore-dir results/openwebtext_opt_350m_l2 \
        --result-dir results/openwebtext_opt_350m_l2 \
        --dataset openwebtext\
        --device cuda --model-identifier facebook/opt-350m\
        --num-probes 32 --num-neighbors 100 --num-centroids 2048 \
         --temperature 15538.91 --use-quantization --distance-type l2\
        --evaluation-metrics bert_score mauve bleurt

For the ablation studies in appendix A.4, run
    
    python3 run_alpha_ablations.py --datastore-dir datastores/deen_m2m100_418M_l2\
        --result-dir results/alpha_ablations/deen_m2m100_418M_l2 --dataset deen\
        --device cuda --num-probes 1024 --num-neighbors 100 --num-centroids 2048\
        --temperature 512.1416 --use-quantization --distance-type l2

    python3 run_alpha_ablations.py --datastore-dir datastores/openwebtext_opt_350M_l2\
        --result-dir results/alpha_ablations/openwebtext_opt_350M_l2 --dataset openwebtext\
        --device cuda --num-probes 32 --num-neighbors 100 --num-centroids 2048 --temperature 15538.91\
        --model-identifier facebook/opt-350m  --use-quantization --distance-type l2 

    python3 run_neighbor_ablations.py --datastore-dir datastores/deen_m2m100_418M_l2\
        --result-dir results/neighbor_ablations/deen_m2m100_418M_l2 --dataset deen\
        --device cuda --num-probes 1024 --num-neighbors 100 --num-centroids 2048\
        --temperature 512.1416 --use-quantization --distance-type l2

    python3 run_neighbor_ablations.py --datastore-dir datastores/openwebtext_opt_350M_l2\
        --result-dir results/neighbor_ablations/openwebtext_opt_350M_l2\
        --dataset openwebtext --device cuda --num-probes 1024 --num-neighbors 100\
        --num-centroids 2048 --temperature 15538.91 --model-identifier facebook/opt-350m\
        --use-quantization --distance-type l2 --data-dir ./data\

## Citation

Please cite the paper and code as following:

@article{ulmer2024non,
  title={Non-Exchangeable Conformal Language Generation with Nearest Neighbors},
  author={Ulmer, Dennis and Zerva, Chrysoula and Martins, Andr{\'e} FT},
  journal={arXiv preprint arXiv:2402.00707},
  year={2024}
}
