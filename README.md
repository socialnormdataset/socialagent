# Measuring Social Norms of Large Language Models
*Ye Yuan, Kexin Tang, Jianhao Shen, Ming Zhang\*, Chenguang Wang\**

The code for NAACL 2024 paper: [Measuring Social Norms of Large Language Models](https://arxiv.org/abs/2404.02491).
<p align="center">
  📃 <a href="https://arxiv.org/abs/2404.02491" target="_blank">[Paper]</a> • 💻 <a href="https://github.com/socialnormdataset/socialagent" target="_blank">[Github]</a> • 🤗 <a href="https://huggingface.co/datasets/socialnormdataset/social" target="_blank">[Dataset]</a> • 📽 <a href="https://github.com/socialnormdataset/socialagent/blob/main/assets/slides-final.pdf" target="_blank">[Slides]</a> • 📋 <a href="https://github.com/socialnormdataset/socialagent/blob/main/assets/poster-final.pdf" target="_blank">[Poster]</a>
</p>

We present a new challenge to examine whether large language models understand social norms. In contrast to existing datasets, our dataset requires a fundamental understanding of social norms to solve. Our dataset features the largest set of social norm skills, consisting of 402 skills and 12,383 questions covering a wide set of social norms ranging from opinions and arguments to culture and laws. We design our dataset according to the K-12 curriculum. This enables the direct comparison of the social understanding of large language models to humans, more specifically, elementary students. While prior work generates nearly random accuracy on our benchmark, recent large language models such as GPT3.5-Turbo and LLaMA2-Chat are able to improve the performance significantly, only slightly below human performance. We then propose a multi-agent framework based on large language models to improve the models' ability to understand social norms. This method further improves large language models to be on par with humans. Given the increasing adoption of large language models in real-world applications, our finding is particularly important and presents a unique direction for future improvements.

## Setup Environment
We recommend using Anaconda to create a new environment and install the required packages. You can create a new environment and install the required packages using the following commands:
```bash
conda create -n social python=3.10
conda activate social
pip install -r requirements.txt
```

## Run the Code
First, please add your OpenAI API key into the `keys.py` file in the following format:
```python
openai_key = "<YOUR_OPENAI_API_KEY>"
```

Then you can run the inference code using the following command:
```bash
git lfs install
git clone https://github.com/socialnormdataset/socialagent
cd socialagent
python main.py --model gpt-3.5-turbo --output_dir $OUTPUT_DIR
python main.py --model gpt-3.5-turbo --output_dir $OUTPUT_DIR --social_agent
python main.py \
  --model llama2 \
  --model_path $LLAMA2_CHAT_MODEL_PATH \
  --output_dir $OUTPUT_DIR
python main.py \
  --model llama2 \
  --model_path $LLAMA2_CHAT_MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --social_agent
```

## Citation
```bibtex
@inproceedings{yuan2024measuring,
    title={Measuring Social Norms of Large Language Models}, 
    author={Ye Yuan and Kexin Tang and Jianhao Shen and Ming Zhang and Chenguang Wang},
    year={2024},
    booktitle={NAACL},
}
```
