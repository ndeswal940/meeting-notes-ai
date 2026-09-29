import json
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="AI Meeting Summarizer & Action Item Tracker",
    layout="wide",
    page_icon="📋",
)

st.title("📋 AI Meeting Notes & Action Item Summarizer")
st.write(
    "Transform unstructured meeting notes or transcripts into structured key decisions, high-level summaries, and action-item matrices."
)

# Retrieve API key securely from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

# Active production models on Groq (Updated to exclude decommissioned models)
PRODUCTION_TEXT_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

# Fetch active models dynamically while filtering out guardrail & audio models
active_models = []
if groq_api_key:
    try:
        client = Groq(api_key=groq_api_key)
        models_data = client.models.list()

        fetched = [
            m.id
            for m in models_data.data
            if not any(
                excluded in m.id.lower()
                for excluded in [
                    "guard",
                    "whisper",
                    "orpheus",
                    "safeguard",
                    "vision",
                    "3.1-8b-instant",
                ]
            )
        ]
        if fetched:
            active_models = sorted(
                fetched, key=lambda x: x not in PRODUCTION_TEXT_MODELS
            )
    except Exception:
        pass

if not active_models:
    active_models = PRODUCTION_TEXT_MODELS

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

model_option = st.sidebar.selectbox(
    "Select AI Model:",
    active_models,
    index=0,
)

# Load Sample Data Button
if st.sidebar.button("📄 Load Sample Transcript"):
    st.session_state["transcript_input"] = (
        "Project Sync - Sept 28\n"
        "Attendees: Nitin, Rahul, Priya\n\n"
        "Nitin shared that the frontend deployment on Streamlit Cloud is almost done, "
        "but we need the final API keys integrated. Rahul mentioned that he will complete "
        "the Groq API integration by tomorrow evening. Priya agreed to draft the "
        "end-term project report covering Sections A through E by Wednesday 5 PM.\n\n"
        "Decisions made:\n"
        "1. We will use Groq API with Llama 3.3 / GPT-OSS models for fast processing.\n"
        "2. Streamlit Community Cloud will be used for hosting."
    )

# Text Area Input
raw_text = st.text_area(
    "Paste Raw Meeting Notes / Transcript:",
    value=st.session_state.get("transcript_input", ""),
    height=220,
    placeholder="Paste meeting transcript or notes here...",
)

# Run Button
if st.button("🚀 Summarize & Extract Action Items"):
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY is missing in App Secrets.")
    elif len(raw_text.strip()) < 30:
        st.warning(
            "⚠️ Input is too short. Please enter a valid transcript (at least 30 characters)."
        )
    else:
        with st.spinner(f"Processing transcript using {model_option}..."):
            try:
                client = Groq(api_key=groq_api_key)

                system_prompt = """
                You are an executive assistant AI. Analyze the provided meeting transcript and extract structured details.
                Return strictly valid JSON only (no markdown, no extra commentary) matching this schema:
                {
                    "summary": "Concise executive summary of the meeting",
                    "key_decisions": ["Decision 1", "Decision 2"],
                    "action_items": [
                        {"task": "Task description", "owner": "Person responsible or Unassigned", "due_date": "YYYY-MM-DD or Not specified"}
                    ]
                }
                """

                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Transcript:\n{raw_text}"},
                    ],
                    model=model_option,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )

                response_content = chat_completion.choices[0].message.content
                data = json.loads(response_content)

                st.success("Analysis Complete!")

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("💡 High-Level Summary")
                    st.info(data.get("summary", "No summary generated."))

                    st.subheader("📌 Key Decisions Made")
                    decisions = data.get("key_decisions", [])
                    if decisions:
                        for d in decisions:
                            st.markdown(f"- {d}")
                    else:
                        st.write("No specific decisions detected.")

                with col2:
                    st.subheader("✅ Action-Item Matrix")
                    action_items = data.get("action_items", [])
                    if action_items:
                        df = pd.DataFrame(action_items)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.write("No action items detected.")

            except Exception as e:
                st.error(f"An error occurred during API execution: {str(e)}")
