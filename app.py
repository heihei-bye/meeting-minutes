import json
import re

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="会议纪要助手", page_icon="📝", layout="wide")

# --- 模型供应商配置 ---
PROVIDERS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "url": "platform.deepseek.com",
    },
    "通义千问 (Qwen)": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max"],
        "default_model": "qwen-plus",
        "url": "dashscope.console.aliyun.com",
    },
    "智谱 (GLM)": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4-flash", "glm-4", "glm-4-plus"],
        "default_model": "glm-4-flash",
        "url": "open.bigmodel.cn",
    },
    "月之暗面 (Kimi)": {
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "default_model": "moonshot-v1-8k",
        "url": "platform.moonshot.cn",
    },
    "硅基流动 (SiliconFlow)": {
        "base_url": "https://api.siliconflow.cn/v1",
        "models": ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct", "THUDM/glm-4-9b-chat"],
        "default_model": "deepseek-ai/DeepSeek-V3",
        "url": "siliconflow.cn",
    },
    "小米 MiLM": {
        "base_url": "https://api.xiaomi.com/v1",
        "models": ["milm-chat"],
        "default_model": "milm-chat",
        "url": "dev.mi.com",
    },
    "自定义 (OpenAI兼容)": {
        "base_url": "",
        "models": [],
        "default_model": "",
        "url": "",
    },
}


def get_client():
    provider = st.session_state.get("provider", "DeepSeek")
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        return None, None

    cfg = PROVIDERS[provider]

    if provider == "自定义 (OpenAI兼容)":
        base_url = st.session_state.get("custom_base_url", "")
        model = st.session_state.get("custom_model", "")
        if not base_url or not model:
            return None, None
    else:
        base_url = cfg["base_url"]
        model = st.session_state.get("model", cfg["default_model"])

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


SYSTEM_PROMPT = """你是一个专业的会议纪要分析师。分析会议转录文本，提取结构化信息。

严格返回以下JSON格式，不要添加任何其他文本：

{
  "title": "会议标题（10字以内）",
  "meeting_date": "YYYY-MM-DD或null",
  "duration_minutes": 数字或null,
  "participants": ["参会人1", "参会人2"],
  "summary": "2-3句话概括核心内容",
  "key_topics": [
    {"topic": "议题名称", "description": "讨论要点"}
  ],
  "decisions": [
    {"content": "决策内容", "people": ["相关人员"]}
  ],
  "action_items": [
    {"assignee": "负责人", "task": "具体任务", "deadline": "截止时间或null"}
  ]
}

规则：
1. 从发言标签识别参会人
2. 行动项明确分配负责人
3. 保持原始语言
4. 缺失信息用null或空数组"""


def parse_json(text):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def analyze(transcript):
    client, model = get_client()
    if not client:
        return None
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"分析以下会议转录文本：\n\n---\n{transcript}\n---"},
        ],
    )
    return parse_json(resp.choices[0].message.content)


# --- 页面 ---
st.title("📝 会议纪要助手")

# 侧边栏
with st.sidebar:
    st.header("设置")

    provider = st.selectbox("AI 服务商", list(PROVIDERS.keys()), key="provider")
    cfg = PROVIDERS[provider]

    if provider == "自定义 (OpenAI兼容)":
        st.text_input("API 地址", placeholder="https://your-api.com/v1", key="custom_base_url")
        st.text_input("模型名称", placeholder="gpt-4o", key="custom_model")
    else:
        if cfg["models"]:
            st.selectbox("模型", cfg["models"], key="model")
        st.caption(f"获取 Key：[{cfg['url']}]({cfg['url']})")

    st.text_input("API Key", type="password", key="api_key", placeholder="粘贴你的 API Key")

    if st.session_state.get("api_key"):
        st.success("已配置")

# 主区域
tab1, tab2 = st.tabs(["新建纪要", "使用说明"])

with tab1:
    transcript = st.text_area(
        "粘贴会议转录文本",
        height=250,
        placeholder="把飞书转写的文字稿粘贴到这里...\n\n例如：\n张三：今天我们讨论一下Q3的营销计划...\n李四：我觉得我们应该重点做线上推广...",
    )

    if st.button("✨ 生成会议纪要", type="primary", use_container_width=True):
        if not transcript.strip():
            st.warning("请先输入转录文本")
        elif not st.session_state.get("api_key"):
            st.error("请先在左侧输入 API Key")
        else:
            with st.spinner("AI 正在分析会议内容..."):
                try:
                    result = analyze(transcript)
                except Exception as e:
                    st.error(f"分析失败：{e}")
                    result = None

            if result:
                st.divider()
                st.header(result.get("title", "会议纪要"))

                cols = st.columns(4)
                with cols[0]:
                    st.metric("日期", result.get("meeting_date") or "未知")
                with cols[1]:
                    dur = result.get("duration_minutes")
                    st.metric("时长", f"{dur}分钟" if dur else "未知")
                with cols[2]:
                    st.metric("参会人", f"{len(result.get('participants', []))}人")
                with cols[3]:
                    st.metric("议题数", f"{len(result.get('key_topics', []))}个")

                participants = result.get("participants", [])
                if participants:
                    st.subheader("参会人员")
                    st.write("  ".join([f"`{p}`" for p in participants]))

                st.subheader("会议摘要")
                st.info(result.get("summary", "无"))

                topics = result.get("key_topics", [])
                if topics:
                    st.subheader("关键议题")
                    for i, t in enumerate(topics, 1):
                        with st.expander(f"📌 {i}. {t['topic']}", expanded=True):
                            st.write(t["description"])

                decisions = result.get("decisions", [])
                if decisions:
                    st.subheader("决策事项")
                    for i, d in enumerate(decisions, 1):
                        people = ", ".join(d.get("people", []))
                        st.markdown(f"**{i}.** {d['content']}" + (f"  *({people})*" if people else ""))

                action_items = result.get("action_items", [])
                if action_items:
                    st.subheader("行动项")
                    for item in action_items:
                        cols = st.columns([1, 3, 1])
                        with cols[0]:
                            st.badge(item["assignee"])
                        with cols[1]:
                            st.write(item["task"])
                        with cols[2]:
                            if item.get("deadline"):
                                st.caption(f"截止：{item['deadline']}")

with tab2:
    st.markdown("""
### 快速开始

1. 左侧选择 **AI 服务商**，输入对应的 **API Key**
2. 粘贴飞书转写的会议文字稿
3. 点击「生成会议纪要」
4. 等待几秒即可

### 推荐方案

| 服务商 | 价格 | 特点 |
|---|---|---|
| **DeepSeek** | 极便宜 | 中文能力强，推荐首选 |
| **通义千问** | 有免费额度 | 阿里云生态，国内访问快 |
| **智谱 GLM** | 有免费额度 | GLM-4-Flash 免费 |
| **月之暗面** | 便宜 | Kimi，支持长文本 |
| **硅基流动** | 便宜 | 聚合多家模型 |

### 手机访问

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
手机浏览器打开 `http://电脑IP:8501`（需同一WiFi）
""")
