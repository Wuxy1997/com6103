from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import gradio as gr
import torch

# 加载预训练的问答模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
# 加载语义搜索模型
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 假设这里是关于HSUHK的信息文档列表
hsuhk_docs = [
    "hsuhk is the short name for the hang seng university of hong kong",
    "商业管理学院提供会计学、金融学等专业课程。",
    "HSUHK注重实践教学，与多家企业有合作。"
]
# 对文档进行编码
doc_embeddings = embedder.encode(hsuhk_docs, convert_to_tensor=True)


def answer_question(question):
    # 语义搜索，找到最相关的文档
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(question_embedding, doc_embeddings)[0]
    top_result_index = torch.argmax(cos_scores).item()
    context = hsuhk_docs[top_result_index]

    inputs = tokenizer(question, context, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start_index:answer_end_index]))
    return answer


iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="HSUHK问答机器人",
    description="输入关于HSUHK的问题，获取答案。"
)
iface.launch(share=True)