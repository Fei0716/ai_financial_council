import streamlit as st
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from financial_council_agents import app  # Import your compiled graph

# 1. Page Config
st.set_page_config(page_title="AI Financial Council", layout="centered")

# --- CUSTOM CSS FOR USER ALIGNMENT & AGENT BORDERS ---
st.markdown("""
<style>
    /* Force User Message to Right */
    [data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] > p:contains("user")) {
        flex-direction: row-reverse;
        text-align: right;
    }
    div[data-testid="stChatMessage"] {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Financial Council 🤖💰")

# 2. Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Input Handling
user_input = st.chat_input("Ask about stocks or your finances...")

if user_input:
    # A. Display User Message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # B. Run the Graph
    config = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": "user_default"}}

    with st.chat_message("assistant"):
        # --- UI LAYOUT ---

        # 1. Status Indicator (New Feature)
        # This sits at the top and updates live as the loop runs
        status_placeholder = st.empty()

        # 2. The Expander: Hides the complex intermediate steps
        details = st.expander("🕵️ View Agent Reasoning & Steps", expanded=True)

        # 3. The Placeholder: Shows ONLY the clean final answer
        final_answer_container = st.empty()

        current_final_text = ""
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # STREAMING LOOP
        for output in app.stream(inputs, config=config):
            for node_name, value in output.items():

                # --- UPDATE STATUS INDICATOR ---
                # Updates the text to show which node is currently active
                status_placeholder.info(f"⏳ **Active Agent:** {node_name} is working...")

                new_messages = value.get("messages", [])

                for msg in new_messages:
                    # --- TYPE 1: INTERNAL STEPS ---
                    is_tool_call = isinstance(msg, AIMessage) and msg.tool_calls
                    is_tool_output = isinstance(msg, ToolMessage)
                    is_instruction = isinstance(msg, HumanMessage) and msg.name == "Supervisor"

                    if is_tool_call or is_tool_output or is_instruction:
                        with details:
                            # Used st.container(border=True) to separate steps visually
                            if is_instruction:
                                with st.container(border=True):
                                    st.markdown(f"**👮 Supervisor Instructions:**")
                                    st.caption(msg.content)

                            elif is_tool_call:
                                with st.container(border=True):
                                    st.markdown(f"**🛠️ {node_name}** calling tools...")
                                    for tool in msg.tool_calls:
                                        st.caption(f"Tool: `{tool['name']}`")
                                        st.code(json.dumps(tool['args'], indent=2), language="json")

                            elif is_tool_output:
                                with st.container(border=True):
                                    st.markdown(f"**✅ Tool Result:** `{msg.name}`")
                                    content = str(msg.content)
                                    if len(content) > 500:
                                        st.text(content[:500] + " ... (truncated)")
                                    else:
                                        st.code(content, language="text")

                    # --- TYPE 2: FINAL ANSWER ---
                    elif isinstance(msg, AIMessage) and msg.content:
                        with details:
                            with st.container(border=True):
                                st.markdown(f"**🗣️ {node_name} (Final Output):**")
                                st.write(msg.content)

                        # Update main display
                        current_final_text = msg.content
                        final_answer_container.markdown(current_final_text)

        # CLEAR STATUS WHEN DONE
        status_placeholder.empty()

    # C. Save Final Response to History
    if current_final_text:
        st.session_state.messages.append({"role": "assistant", "content": current_final_text})