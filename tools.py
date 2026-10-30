import os
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
import pdfplumber
import pymupdf
from dashscope import MultiModalConversation

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def search_documents(query, chunks, embeddings, top_k=5):
    """输入查询，返回最相关的几个文本块"""
    resp = TextEmbedding.call(
        model="text-embedding-v1",
        input=query
    )
    if resp.status_code != 200:
        print("向量化失败")
        return []

    query_emb = resp.output["embeddings"][0]["embedding"]

    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_emb, emb)
        scores.append((score, i))

    scores.sort(reverse=True)

    # 1. 先取出常规的 Top-K
    retrieved_chunks = [chunks[idx] for _, idx in scores[:top_k]]

    # 2. 混合检索兜底：如果问作者/邮箱，强行把第一块插进去
    if "作者" in query or "author" in query.lower() or "邮箱" in query:
        if chunks[0] not in retrieved_chunks:
            retrieved_chunks.insert(0, chunks[0])

    # 3. 确保绝对返回一个列表
    return retrieved_chunks

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def extract_tables(pdf_path):
    """提取PDF中的所有表格，转成文字描述"""
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                table_text = "【表格】\n"
                for row in table:
                    row_text = " | ".join([str(c) if c else "" for c in row])
                    table_text += row_text + "\n"
                all_tables.append(table_text)
    return "\n\n".join(all_tables)


def extract_images(pdf_path, output_dir="images"):
    """从PDF里提取所有图片，保存到文件夹"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    doc = pymupdf.open(pdf_path)
    image_paths = []
    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = f"{output_dir}/page{page_num + 1}_img{img_index + 1}.{image_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_path)
    return image_paths


def read_image(image_path):
    """读取图片内容，返回文字描述或识别出的文字"""
    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"file://{image_path}"},
                {
                    "text": "请精确提取图片中的所有文字（OCR）。如果是英文，请严格保留英文原文，绝对不要翻译成中文。如果是表格，请用Markdown格式输出。"}
            ]
        }
    ]
    resp = MultiModalConversation.call(
        model="qwen3-vl-plus",
        messages=messages
    )
    if resp.status_code == 200:
        return resp.output["choices"][0]["message"]["content"][0]["text"]
    else:
        return f"识别失败: {resp.message}"