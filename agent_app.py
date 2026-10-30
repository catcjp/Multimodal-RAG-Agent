import streamlit as st
import os
import json
import pickle
import glob  # 新增：用于查找文件
from agent import load_pdf, split_text, get_embedding, simple_agent
from tools import extract_tables, extract_images, read_image

# 1. 页面配置：改为宽屏模式
st.set_page_config(page_title="多模态 Agent", layout="wide")
st.title("🖼️ 多模态 Agent 测试")

# 2. 侧边栏：上传、清理与数据展示
with st.sidebar:
    st.header("📄 文档管理")

    # ===== 新增：一键清理按钮 =====
    if st.button("🧹 一键清理无用缓存与临时文件", use_container_width=True):
        # 查找匹配的文件
        temp_files = glob.glob("temp_*.pdf")
        cache_jsons = glob.glob("cache_*.json")
        cache_pkls = glob.glob("cache_*.pkl")

        all_files = temp_files + cache_jsons + cache_pkls
        count = 0
        for file in all_files:
            try:
                os.remove(file)
                count += 1
            except Exception:
                pass

        # 清空当前的网页状态，防止读出已删除的缓存
        if "chunks" in st.session_state:
            del st.session_state["chunks"]
            del st.session_state["embeddings"]

        st.success(f"✅ 清理完成！共删除了 {count} 个文件。")
    # ==============================

    st.divider()  # 分割线

    uploaded_file = st.file_uploader("上传 PDF", type=["pdf"])

    # 永远显示当前状态
    if uploaded_file is not None:
        st.success(f"当前文档：{uploaded_file.name}")

    # 清空对话按钮
    if st.button("🗑️ 清空当前对话", use_container_width=True):
        st.session_state["messages"] = []

    # 展示提取的多模态数据
    if uploaded_file is not None and "chunks" in st.session_state:
        st.divider()
        with st.expander("🔍 查看底层多模态提取结果"):
            st.info("这是系统从 PDF 抠出来的文本、表格和图片 OCR 结果：")
            st.write(st.session_state["chunks"][0][:1000] + "...\n\n(文本已成功融入大模型上下文)")

# 3. 核心处理逻辑
if uploaded_file is not None:
    pdf_path = f"./temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_prefix = uploaded_file.name.replace(".pdf", "")
    cache_text_file = f"cache_{file_prefix}_text.json"
    cache_data_file = f"cache_{file_prefix}_data.pkl"

    # 如果网页刷新且没有加载过数据，则去查缓存或重新解析
    if "chunks" not in st.session_state:
        if os.path.exists(cache_text_file) and os.path.exists(cache_data_file):
            st.sidebar.success("🚀 检测到缓存，秒开加载！")
            with open(cache_text_file, "r", encoding="utf-8") as f:
                text = json.load(f)
            with open(cache_data_file, "rb") as f:
                cached_data = pickle.load(f)
                st.session_state["chunks"] = cached_data["chunks"]
                st.session_state["embeddings"] = cached_data["embeddings"]
        else:
            st.sidebar.warning("⚠️ 未找到缓存，正在全量解析...")
            with st.spinner("解析中（包含抠图、调模型），请稍候..."):
                try:
                    text = load_pdf(pdf_path)
                    tables = extract_tables(pdf_path)
                    text = text + "\n\n【表格内容】\n" + tables

                    images = extract_images(pdf_path)
                    image_texts = []
                    for img_path in images[:2]:
                        img_desc = read_image(img_path)
                        image_texts.append(img_desc)
                    text = text + "\n\n【图片内容】\n" + "\n".join(image_texts)

                    chunks = split_text(text)
                    embeddings = []
                    for chunk in chunks:
                        emb = get_embedding(chunk)
                        if emb is None:
                            st.error("❌ 向量化失败，请检查终端。")
                            st.stop()
                        embeddings.append(emb)

                    st.session_state["chunks"] = chunks
                    st.session_state["embeddings"] = embeddings

                    with open(cache_text_file, "w", encoding="utf-8") as f:
                        json.dump(text, f, ensure_ascii=False)
                    with open(cache_data_file, "wb") as f:
                        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

                    st.sidebar.success("✅ 解析完成并保存缓存！")
                except Exception as e:
                    st.error(f"解析出错: {e}")
                    st.stop()

# 4. 聊天交互逻辑
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 渲染历史聊天气泡
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 接收用户输入
# 如果没传文件，给个提示
if uploaded_file is None:
    st.info("👈 请先在左侧上传 PDF 文档，然后就可以开始聊天啦！")

# ===== 新增：如果没有聊天记录，给个占位提示 =====
# 判断展示什么提示语
if len(st.session_state["messages"]) == 0:
    if uploaded_file is None:
        st.info("👋 你好！请在左侧上传文档，然后在下方输入你想问的问题。")
    elif "chunks" in st.session_state and st.session_state.get("chunks") is not None:
        st.success("✅ 文档已就绪！请在下方输入你想问的问题。")
    else:
        st.warning("⚠️ 文档内容为空或无法解析，请尝试换一个 PDF。")

question = st.chat_input("请输入关于文档的问题...")

if question:
    if uploaded_file is None:
        st.warning("请先在左侧上传 PDF 文件！")
    elif "chunks" not in st.session_state:
        st.warning("文档还在处理中，请稍候...")
    else:
        # 显示用户提问
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # 调用 Agent 思考
        with st.chat_message("assistant"):
            with st.spinner("Agent 正在思考并检索..."):
                # 提取历史记录给后端
                history = []
                for i in range(0, len(st.session_state["messages"]) - 1, 2):
                    history.append({
                        "user": st.session_state["messages"][i]["content"],
                        "assistant": st.session_state["messages"][i + 1]["content"]
                    })

                # 调后端函数
                answer = simple_agent(
                    question,
                    st.session_state["chunks"],
                    st.session_state["embeddings"],
                    history
                )

                st.write(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})