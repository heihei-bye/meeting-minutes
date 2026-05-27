import json
import re
from collections import Counter

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="智能纪要", page_icon="📝", layout="wide")

# --- 自定义样式：报告风格 ---
st.markdown("""
<style>
    .report-title { font-size: 2.2rem; font-weight: 700; text-align: center; margin-bottom: 0; color: #1a1a2e; }
    .report-date { text-align: center; color: #888; font-size: 1rem; margin-bottom: 2rem; }
    .section-header { font-size: 1.3rem; font-weight: 700; color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 8px; margin-top: 2rem; }
    .info-card { background: #f8f9fa; border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem; border-left: 4px solid #0f3460; }
    .topic-card { background: #fff; border: 1px solid #e9ecef; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
    .topic-title { font-weight: 600; color: #0f3460; font-size: 1.05rem; }
    .topic-desc { color: #555; margin-top: 4px; }
    .decision-item { background: #f0f7ff; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; border-left: 4px solid #4A90D9; }
    .action-item { background: #fffbf0; border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; border-left: 4px solid #fa8c16; }
    .assignee-badge { display: inline-block; background: #0f3460; color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 0.85rem; margin-right: 8px; }
    .deadline-text { color: #f5222d; font-size: 0.85rem; }
    .summary-box { background: linear-gradient(135deg, #f0f7ff, #e8f4f8); border-radius: 12px; padding: 1.5rem; border: 1px solid #d6eaf8; }
    .metric-label { font-size: 0.85rem; color: #888; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #16213e; }
</style>
""", unsafe_allow_html=True)

# --- 模型供应商 ---
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
        "models": [
            "XiaomiMiMo/MiMo-7B-RL",
            "deepseek-ai/DeepSeek-V3",
            "Qwen/Qwen2.5-72B-Instruct",
        ],
        "default_model": "XiaomiMiMo/MiMo-7B-RL",
        "url": "siliconflow.cn",
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
    return OpenAI(api_key=api_key, base_url=base_url), model


SYSTEM_PROMPT = """你是一个专业的会议纪要分析师。分析会议转录文本，提取结构化信息。

严格返回以下JSON格式，不要添加任何其他文本：

{
  "title": "会议标题（精炼概括，15字以内）",
  "meeting_date": "YYYY-MM-DD或null",
  "duration_minutes": 数字或null,
  "participants": ["参会人1", "参会人2"],
  "summary": "会议摘要（3-5句话，概括背景、讨论过程和结论）",
  "key_topics": [
    {
      "topic": "议题名称",
      "description": "该议题的详细讨论要点（2-3句话）",
      "category": "议题分类（如：技术方案/需求讨论/进度汇报/资源协调/其他）"
    }
  ],
  "decisions": [
    {
      "content": "决策内容",
      "reason": "决策原因或依据",
      "people": ["相关人员"]
    }
  ],
  "action_items": [
    {
      "assignee": "负责人",
      "task": "具体任务描述",
      "deadline": "截止时间或null",
      "priority": "高/中/低"
    }
  ],
  "timeline": [
    {
      "time": "时间段（如 10:00-10:15）",
      "content": "该时间段讨论内容摘要"
    }
  ],
  "risks_and_issues": [
    {
      "issue": "风险或问题描述",
      "suggestion": "建议的解决方案"
    }
  ]
}

分析规则：
1. 从发言标签识别参会人，没有标签则根据上下文推断
2. 行动项必须明确分配给具体负责人
3. 保持原始语言（中文会议用中文输出）
4. 议题分类要准确，便于后续统计
5. 如果信息无法提取，使用null或空数组"""


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
            {"role": "user", "content": f"分析以下会议转录文本，生成专业会议纪要：\n\n---\n{transcript}\n---"},
        ],
    )
    return parse_json(resp.choices[0].message.content)


def render_report(result):
    """渲染报告风格的会议纪要"""
    # === 标题 ===
    st.markdown(f'<div class="report-title">{result.get("title", "会议纪要")}</div>', unsafe_allow_html=True)

    date_str = result.get("meeting_date") or "未注明日期"
    dur = result.get("duration_minutes")
    dur_str = f" | 时长 {dur} 分钟" if dur else ""
    st.markdown(f'<div class="report-date">{date_str}{dur_str}</div>', unsafe_allow_html=True)
    st.divider()

    # === 基础信息 ===
    st.markdown('<div class="section-header">基础信息</div>', unsafe_allow_html=True)
    participants = result.get("participants", [])
    topics = result.get("key_topics", [])
    decisions = result.get("decisions", [])
    actions = result.get("action_items", [])

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="metric-label">参会人数</div><div class="metric-value">{len(participants)}</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f'<div class="metric-label">议题数量</div><div class="metric-value">{len(topics)}</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="metric-label">决策事项</div><div class="metric-value">{len(decisions)}</div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="metric-label">行动项</div><div class="metric-value">{len(actions)}</div>', unsafe_allow_html=True)

    if participants:
        st.markdown(f'**参会人员：** {" / ".join(participants)}')

    # === 会议摘要 ===
    st.markdown('<div class="section-header">会议摘要</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">{result.get("summary", "暂无摘要")}</div>', unsafe_allow_html=True)

    # === 议题分类统计图 ===
    if topics:
        st.markdown('<div class="section-header">议题分类</div>', unsafe_allow_html=True)
        categories = [t.get("category", "其他") for t in topics]
        cat_counts = Counter(categories)

        cols = st.columns([1, 1])
        with cols[0]:
            try:
                import plotly.express as px
                fig = px.pie(
                    names=list(cat_counts.keys()),
                    values=list(cat_counts.values()),
                    color_discrete_sequence=["#0f3460", "#4A90D9", "#52c41a", "#fa8c16", "#f5222d", "#722ed1"],
                )
                fig.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=300,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.bar_chart(cat_counts)

        with cols[1]:
            for cat, count in cat_counts.items():
                st.markdown(f"**{cat}** — {count} 个议题")

    # === 关键议题 ===
    if topics:
        st.markdown('<div class="section-header">关键议题</div>', unsafe_allow_html=True)
        for i, t in enumerate(topics, 1):
            st.markdown(f"""
            <div class="topic-card">
                <div class="topic-title">{i}. {t['topic']}</div>
                <div style="margin-top:6px"><span style="background:#e8f0fe;color:#0f3460;padding:2px 8px;border-radius:8px;font-size:0.8rem;">{t.get('category', '其他')}</span></div>
                <div class="topic-desc" style="margin-top:8px">{t['description']}</div>
            </div>
            """, unsafe_allow_html=True)

    # === 决策事项 ===
    if decisions:
        st.markdown('<div class="section-header">关键决策</div>', unsafe_allow_html=True)
        for i, d in enumerate(decisions, 1):
            people = " / ".join(d.get("people", []))
            reason = d.get("reason", "")
            st.markdown(f"""
            <div class="decision-item">
                <div><strong>决策 {i}：</strong>{d['content']}</div>
                {'<div style="color:#888;font-size:0.9rem;margin-top:4px">依据：' + reason + '</div>' if reason else ''}
                {'<div style="margin-top:6px"><span class="assignee-badge">' + people + '</span></div>' if people else ''}
            </div>
            """, unsafe_allow_html=True)

    # === 行动事项 ===
    if actions:
        st.markdown('<div class="section-header">行动事项</div>', unsafe_allow_html=True)
        for i, item in enumerate(actions, 1):
            priority = item.get("priority", "")
            priority_colors = {"高": "#f5222d", "中": "#fa8c16", "低": "#52c41a"}
            p_color = priority_colors.get(priority, "#999")
            deadline = item.get("deadline", "")
            st.markdown(f"""
            <div class="action-item">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                    <span class="assignee-badge">{item['assignee']}</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#333">{item['task']}</span>
                </div>
                <div style="margin-top:6px;display:flex;gap:12px;align-items:center">
                    {f'<span class="deadline-text">截止：{deadline}</span>' if deadline else ''}
                    {f'<span style="background:{p_color};color:#fff;padding:1px 8px;border-radius:8px;font-size:0.75rem">{priority}优先级</span>' if priority else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # === 时间线 ===
    timeline = result.get("timeline", [])
    if timeline:
        st.markdown('<div class="section-header">会议时间线</div>', unsafe_allow_html=True)
        for i, t in enumerate(timeline):
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start">
                <div style="min-width:8px;min-height:8px;width:8px;height:8px;border-radius:50%;background:#0f3460;margin-top:8px"></div>
                <div>
                    <div style="font-weight:600;color:#0f3460">{t.get('time', '')}</div>
                    <div style="color:#555;font-size:0.95rem;margin-top:2px">{t.get('content', '')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # === 风险与问题 ===
    risks = result.get("risks_and_issues", [])
    if risks:
        st.markdown('<div class="section-header">风险与问题</div>', unsafe_allow_html=True)
        for r in risks:
            st.markdown(f"""
            <div style="background:#fff5f5;border-radius:10px;padding:1rem;margin-bottom:0.8rem;border-left:4px solid #f5222d">
                <div style="font-weight:600;color:#f5222d">⚠ {r.get('issue', '')}</div>
                <div style="color:#555;margin-top:4px">建议：{r.get('suggestion', '')}</div>
            </div>
            """, unsafe_allow_html=True)

    # === 底部 ===
    st.divider()
    st.caption("本纪要由 AI 自动生成，仅供参考，请以实际会议内容为准。")


# ========== 页面 ==========
st.markdown("# 📝 智能纪要")

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
tab1, tab2 = st.tabs(["📝 新建纪要", "📖 使用说明"])

with tab1:
    transcript = st.text_area(
        "粘贴会议转录文本",
        height=280,
        placeholder="把飞书转写的文字稿粘贴到这里...\n\n张三：今天我们讨论一下Q3的营销计划...\n李四：我觉得我们应该重点做线上推广...",
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
                render_report(result)

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
| **硅基流动** | 便宜 | 聚合多家模型，含MiMo |
""")
