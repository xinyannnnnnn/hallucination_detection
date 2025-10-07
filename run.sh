#!/bin/bash

set -euo pipefail

#alldone
# python src/haloscope/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --most_likely 1 --num_gene 1
# python src/haloscope/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1 --most_likely 1
# python src/haloscope/main.py --model_name llama2_7B --dataset_name tigrinya --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --most_likely 1 --num_gene 1
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1 --most_likely 1
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name tigrinya --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
# python src/haloscope/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --most_likely 1 --num_gene 1
# python src/haloscope/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1 --most_likely 1
# python src/haloscope/main.py --model_name llama2_7B --dataset_name armenian --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --most_likely 1 --num_gene 1
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1 --most_likely 1
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name armenian --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
# python src/haloscope/main.py --model_name llama2_7B --dataset_name basque --gene 1 --most_likely 1 --num_gene 1
# python src/haloscope/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1 --most_likely 1
# python src/haloscope/main.py --model_name llama2_7B --dataset_name basque --most_likely 1 --weighted_svd 1 --feat_loc_svd 3
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --most_likely 1 --num_gene 1
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1 --most_likely 1
# python src/haloscope/main.py --model_name opt_6_7b --dataset_name basque --most_likely 1 --weighted_svd 1 --feat_loc_svd 3

#alldone
# python src/self_evaluation/main.py --model_name opt_6_7b --dataset_name tigrinya
# python src/self_evaluation/main.py --model_name llama2_7B --dataset_name tigrinya
# python src/self_evaluation/main.py --model_name opt_6_7b --dataset_name armenian
# python src/self_evaluation/main.py --model_name llama2_7B --dataset_name armenian
# python src/self_evaluation/main.py --model_name opt_6_7b --dataset_name basque
# python src/self_evaluation/main.py --model_name llama2_7B --dataset_name basque

#alldone
# python src/verbalize/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --most_likely 1 --gpu_id 3
# python src/verbalize/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1
# python src/verbalize/main.py --model_name llama2_7B --dataset_name tigrinya
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --most_likely 1 --gpu_id 3
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name tigrinya
# python src/verbalize/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --most_likely 1 
# python src/verbalize/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
# python src/verbalize/main.py --model_name llama2_7B --dataset_name armenian 
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --most_likely 1 
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name armenian
# python src/verbalize/main.py --model_name llama2_7B --dataset_name basque --gene 1 --most_likely 1 --gpu_id 1
# python src/verbalize/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1
# python src/verbalize/main.py --model_name llama2_7B --dataset_name basque
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --most_likely 1 --gpu_id 2
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1
# python src/verbalize/main.py --model_name opt_6_7b --dataset_name basque

#alldone
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --num_gene 3
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name tigrinya
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --num_gene 3
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name tigrinya
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --num_gene 3 --gpu_id 1
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name armenian
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --num_gene 3
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name armenian
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name basque --gene 1 --num_gene 3
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1
# python src/selfcheckgpt/main.py --model_name llama2_7B --dataset_name basque
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --num_gene 3
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1
# python src/selfcheckgpt/main.py --model_name opt_6_7b --dataset_name basque

# CCS* (Contrast-Consistent Search) experiments
#done
# python src/ccs/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --num_gene 3
# python src/ccs/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1
# python src/ccs/main.py --model_name llama2_7B --dataset_name tigrinya --layer -1
#done
# python src/ccs/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --num_gene 3
# python src/ccs/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/ccs/main.py --model_name opt_6_7b --dataset_name tigrinya --layer -1
#done
# python src/ccs/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --num_gene 3
# python src/ccs/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
# python src/ccs/main.py --model_name llama2_7B --dataset_name armenian --layer -1
# done
# python src/ccs/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --num_gene 3
# python src/ccs/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/ccs/main.py --model_name opt_6_7b --dataset_name armenian --layer -1
#done
# python src/ccs/main.py --model_name llama2_7B --dataset_name basque --gene 1 --num_gene 3
# python src/ccs/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1
# python src/ccs/main.py --model_name llama2_7B --dataset_name basque --layer -1
#done
#python src/ccs/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --num_gene 3
# python src/ccs/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1
# python src/ccs/main.py --model_name opt_6_7b --dataset_name basque --layer -1

#alldone
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --num_gene 10
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1 --auto_threshold
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name armenian
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --num_gene 10
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name armenian
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name basque --gene 1 --num_gene 10 
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1 --auto_threshold
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name basque
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --num_gene 10
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1 --auto_threshold
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name basque
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --num_gene 10
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1 --auto_threshold
# python src/lexical_similarity/main.py --model_name llama2_7B --dataset_name tigrinya 
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --num_gene 10
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/lexical_similarity/main.py --model_name opt_6_7b --dataset_name tigrinya

#alldone
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --num_gene 9 --gpu_id 1
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name armenian --use_percentile_threshold 1 --positive_percentage 0.3 --gpu_id 1 #not done
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --num_gene 9 --gpu_id 0 # done
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name armenian
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name basque --gene 1 --num_gene 9 --gpu_id 1 # done
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name basque
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --num_gene 9 --gpu_id 2 # done
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name basque
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --num_gene 9 
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1
# python src/eigenscore/main.py --model_name llama2_7B --dataset_name tigrinya 
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --num_gene 9 
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/eigenscore/main.py --model_name opt_6_7b --dataset_name tigrinya 

#alldone
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --num_gene 10
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name armenian
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --num_gene 3 --gpu_id 1
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name armenian
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name basque --gene 1 --num_gene 10 --gpu_id 2
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name basque
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --num_gene 10 --gpu_id 2
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name basque
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --num_gene 10 
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1
# python src/ln_entropy/main.py --model_name llama2_7B --dataset_name tigrinya 
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --num_gene 10
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/ln_entropy/main.py --model_name opt_6_7b --dataset_name tigrinya

# alldone
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name armenian --gene 1 --num_gene 3
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name armenian
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name armenian --gene 1 --num_gene 3 --gpu_id 3
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name armenian
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name basque --gene 1 --num_gene 3
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name basque
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name basque --gene 1 --num_gene 3
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name basque
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1 --num_gene 3
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1
# python src/semantic_entropy/main.py --model_name llama2_7B --dataset_name tigrinya
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1 --num_gene 3
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/semantic_entropy/main.py --model_name opt_6_7b --dataset_name tigrinya

# done python src/perplexity/main.py --model_name llama2_7B --dataset_name armenian
# donepython src/perplexity/main.py --model_name opt_6_7b --dataset_name armenian 
# donepython src/perplexity/main.py --model_name llama2_7B --dataset_name basque
# donepython src/perplexity/main.py --model_name opt_6_7b --dataset_name basque 
# done python src/perplexity/main.py --model_name llama2_7B --dataset_name tigrinya 
# done python src/perplexity/main.py --model_name opt_6_7b --dataset_name tigrinya 

# alldone
# python src/hallushift/main.py --model_name llama2_7B --dataset_name tigrinya --gene 1
# python src/hallushift/main.py --model_name llama2_7B --dataset_name tigrinya --generate_gt 1
# python src/hallushift/main.py --model_name llama2_7B --dataset_name tigrinya
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name tigrinya --gene 1
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name tigrinya --generate_gt 1
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name tigrinya
# python src/hallushift/main.py --model_name llama2_7B --dataset_name armenian --gene 1
# python src/hallushift/main.py --model_name llama2_7B --dataset_name armenian --generate_gt 1
# python src/hallushift/main.py --model_name llama2_7B --dataset_name armenian
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name armenian --gene 1
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name armenian --generate_gt 1
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name armenian
#python src/hallushift/main.py --model_name llama2_7B --dataset_name basque --gene 1
# python src/hallushift/main.py --model_name llama2_7B --dataset_name basque --generate_gt 1
# python src/hallushift/main.py --model_name llama2_7B --dataset_name basque
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name basque --gene 1
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name basque --generate_gt 1
# python src/hallushift/main.py --model_name opt_6_7b --dataset_name basque