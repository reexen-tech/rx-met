
# code generation
python download_dataset_from_huggingface.py --dataset google-research-datasets/mbpp --output-dir ../datasets/model_evaluation/code_generation
python download_dataset_from_huggingface.py --dataset openai/openai_humaneval --output-dir ../datasets/model_evaluation/code_generation

# commonsense reasoning
python download_dataset_from_huggingface.py --dataset Rowan/hellaswag --output-dir ../datasets/model_evaluation/commonsense_reasoning
python download_dataset_from_huggingface.py --dataset gimmaru/piqa --output-dir ../datasets/model_evaluation/commonsense_reasoning
python download_dataset_from_huggingface.py --dataset allenai/winogrande --output-dir ../datasets/model_evaluation/commonsense_reasoning

# general evaluation
python download_dataset_from_huggingface.py --dataset nyu-mll/glue --output-dir ../datasets/model_evaluation/general_evaluation
python download_dataset_from_huggingface.py --dataset SetFit/mnli --output-dir ../datasets/model_evaluation/general_evaluation
python download_dataset_from_huggingface.py --dataset SetFit/qnli --output-dir ../datasets/model_evaluation/general_evaluation

# knowledge evaluation
python download_dataset_from_huggingface.py --dataset cais/mmlu --output-dir ../datasets/model_evaluation/knowledge_evaluation
python download_dataset_from_huggingface.py --dataset mandarjoshi/trivia_qa --output-dir ../datasets/model_evaluation/knowledge_evaluation
python download_dataset_from_huggingface.py --dataset truthfulqa/truthful_qa --output-dir ../datasets/model_evaluation/knowledge_evaluation

# language modeling
# python download_dataset_from_huggingface.py --dataset c4 --output-dir ../datasets/model_evaluation/language_modeling
python download_dataset_from_huggingface.py --dataset cimec/lambada --output-dir ../datasets/model_evaluation/language_modeling
python download_dataset_from_huggingface.py --dataset perplexity-ai/draco --output-dir ../datasets/model_evaluation/language_modeling
python download_dataset_from_huggingface.py --dataset Salesforce/wikitext --output-dir ../datasets/model_evaluation/language_modeling
# python download_dataset_from_huggingface.py --dataset wikitext-2 --output-dir ../datasets/model_evaluation/language_modeling

# math reasoning
python download_dataset_from_huggingface.py --dataset Idavidrein/gpqa --output-dir ../datasets/model_evaluation/math_reasoning
python download_dataset_from_huggingface.py --dataset openai/gsm8k --output-dir ../datasets/model_evaluation/math_reasoning

# reading comprehension
python download_dataset_from_huggingface.py --dataset allenai/ai2_arc --output-dir ../datasets/model_evaluation/reading_comprehension
python download_dataset_from_huggingface.py --dataset allenai/openbookqa --output-dir ../datasets/model_evaluation/reading_comprehension
python download_dataset_from_huggingface.py --dataset SetFit/rte --output-dir ../datasets/model_evaluation/reading_comprehension
python download_dataset_from_huggingface.py --dataset rajpurkar/squad --output-dir ../datasets/model_evaluation/reading_comprehension

