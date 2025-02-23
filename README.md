# 2024 U.S. Election Multimodal AIGC Dataset

The **2024 U.S. Election Multimodal AIGC Dataset** comprises two components: text and image. The text section includes an AI text classifier and its training set. This dataset consists of tweets produced by four leading AI models, along with human-written tweets drawn from a pre-ChatGPT era collection. The image section features a dataset categorized by GPT-4o.

## AI-Generated Text: Dataset and Classifer

The dataset is constructed based on the methodology introduced in Dmonte et al. (2024) and further refined using prompts as suggested by Cinus et al. (2025). It includes:
- **Human-Written Tweets:** Randomly sampled from a dataset related to the 2020 U.S. Election (Chen, Deb, and Ferrara 2022), ensuring the texts predate widespread large language model (LLM) usage.
- **AI-Generated Tweets:** For each human-written tweet, an AI-generated counterpart is created using the following prompt:
  > *"This is a tweet related to the 2020 U.S. election: {tweet}. Based on the topic of the given tweet, write a new tweet, mimicking the language styles used by Twitter users."*

The AI-generated tweets are produced using four different models:
- GPT-4o
- Claude 3 Sonnet
- Gemini 1.5
- Llama 3 8B

This dataset is split into training and testing sets and is suitable for binary classification tasks to detect whether a tweet is human-written or AI-generated.
**The dataset is located in the `text_dataset` folder. The train and test datasets are in `train.csv` and `test.csv`, respectively. Human and AI-generated tweet pairs for each LLM (GPT-4o, Claude 3 Sonnet, Gemini 1.5, and Llama 3 8B) can be found in the `human_ai_tweet_pairs` folder under the `text_dataset` folder.**


### Model Training

A RoBERTa model is fine-tuned to classify tweets as human-written or AI-generated. Key training details include:
- **Optimizer:** AdamW
- **Learning Rate:** 1e-5
- **Epochs:** 3
- **Performance:** The model achieved an F1-score of 0.96 on the validation set.

To train the model, simply run the following command:

```bash
python train.py
```


## AI-Generated Image: Dataset

### Dataset Structure

#### **1. AI Image Classification**

- The dataset is **filtered and labeled using GPT-4o**, with the following prompt:
   *"Is this an AI-generated image? Answer in one word: 'yes' or 'no'."*
- The model classifies images into two categories:
  - `"yes"` → AI-generated image
  - `"no"` → non-AI-generated image
- Only valid responses (`"yes"` or `"no"`) are included in the dataset.

#### **2. Data Organization**

- **One folder per month** (July, August, September)
- **Each folder contains multiple CSV files**, with **each CSV storing up to 100,000 records**
- **Each row represents a single image**

#### **3. CSV File Format**

Each CSV file consists of **three columns**:

| Column      | Description                                                  | Example                                    |
| ----------- | ------------------------------------------------------------ | ------------------------------------------ |
| `content`   | Classification result (`yes` = AI-generated, `no` = non-AI-generated) | `yes`                                      |
| `image_url` | Direct link to the image                                     | `https://pbs.twimg.com/media/xxxxx.jpg`    |
| `tweet_url` | Link to the original tweet containing the image              | `https://twitter.com/user/status/xxxxxxxx` |

------

### Data Statistics 📊

The dataset contains a total of **2,228,462** classified images, distributed as follows:

| Month         | Number of Images |
| ------------- | ---------------- |
| **July**      | 1,130,560        |
| **August**    | 519,687          |
| **September** | 578,215          |
| **Total**     | **2,228,462**    |

------

### File Structure 📂

The dataset is organized into monthly folders, each containing chunked CSV files:

```
/AIGC_Image_Dataset/
│── July/
│   ├── july_chunk_0.csv
│   ├── july_chunk_1.csv
│   ├── ...
│
│── August/
│   ├── august_chunk_0.csv
│   ├── august_chunk_1.csv
│   ├── ...
│
│── September/
│   ├── september_chunk_0.csv
│   ├── september_chunk_1.csv
│   ├── ...
```

- Each folder contains **multiple CSV files**, starting from `*_chunk_0.csv`, with each file storing up to **100,000** records.
- **Files are numbered sequentially** (`chunk_0`, `chunk_1`, etc.).

------

### How to Use

- Researchers can use the `content` column to **filter AI-generated vs. non-AI-generated images**.
- The `image_url` and `tweet_url` allow users to **trace back to the original content** for further analysis.


## Citation

If you find this work helpful in your research, please consider citing our paper:

```bibtex
@misc{chen2025prevalencesharingpatternsspreaders,
      title={Prevalence, Sharing Patterns, and Spreaders of Multimodal AI-Generated Content on X during the 2024 U.S. Presidential Election}, 
      author={Zhiyi Chen and Jinyi Ye and Emilio Ferrara and Luca Luceri},
      year={2025},
      eprint={2502.11248},
      archivePrefix={arXiv},
      primaryClass={cs.SI},
      url={https://arxiv.org/abs/2502.11248}, 
}
