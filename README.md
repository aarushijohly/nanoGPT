# LLM-from-scratch
Neural networks that are optimized durung training to predict the next word in a sequence. Next word prediction is sensible as it harnesses the inherent sequential nature of language to train models on understanding context, structure and relationships within text. LLMs utilize an architecture called the transformer, which allows them to pay selective attention to different parts of the input when making predictiions.

Applications of LLMs: Machine translation, generation of novel texts, sentiment analysis, text summarization.
LLMs can power sophisticated chatbots and virtual assistants such as ChatGPT or BARD.

Custom build LLMs offers several advantages particularly ragarding data privacy. Companies may prefer not to share their data with third party LLM providers. Also, developing smaller custom LLMs enables deployment directly on customer devices. This local implementation can significantly decrease latency and reduce server related cost. Furthermore, custom LLMs grant developers complete autonomy, allowing them to control updates and modifications to the model as needed.

Foundational/base model: LLM is trained on large corpus of raw data to predict the next word. pre-training
Fine-tuning: Furthur LLM is trained on labeled data. Two most popular categories of fine-tuning LLMs are instruction fine-tuning and classification fine-tuning.

Transformer architecture: Consists of two submodules an encoder and a decoder.
Encoder: Processes the input text and encodes it into a series of numerical representations or vector that captures the contextual information of the input.
Decoder: Takes the encoded vectors and generates the output text.

Bidirectional Encoder Representaion from Transformers(BERT): Encoder only. Specialize in masked word prediction, where the model predict masked or hidden words in a given sentence. This training strategy enables BERT in text classification tasks, including sentiment prediction and document categorization.


Generative Pretrained Transformers(GPT): Decoder only. Designed for generative tasks. Example: Machine translation, text summarization, fiction writing, writing computer code etc. Type of autoregressive models.
