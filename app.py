

import streamlit as st
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, List, Annotated
import operator
from langchain_community.llms import Ollama

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    user_input: str
    response: str

@st.cache_resource
def load_llm():
    return Ollama(model="llama3:latest", temperature=0.7)

def chat_node(state: AgentState):
    llm = load_llm()
    
    conversation = []
    for msg in state["messages"]:
        conversation.append(f"{msg['role']}: {msg['content']}")
    
    context = "\n".join(conversation[-10:])
    
    prompt = f"""You are a helpful assistant with perfect memory.
    
Conversation history:
{context}

User: {state['user_input']}
Assistant:"""
    
    response = llm.invoke(prompt)
    return {"response": response}

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("chat", chat_node)
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)

st.set_page_config(page_title="Multi-Turn Chatbot", page_icon="🤖")

st.title("🤖 Multi-Turn Conversational Agent")
st.caption("With FULL conversation memory using LangGraph")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
    st.session_state.thread_id = "session-1"
    st.session_state.messages = []

with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = f"session-{len(st.session_state.messages)}"
        st.rerun()
    st.write(f"Messages: {len(st.session_state.messages)}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking with memory..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = st.session_state.graph.invoke(
                {
                    "messages": st.session_state.messages,
                    "user_input": prompt,
                    "response": ""
                },
                config=config
            )
            st.write(result["response"])
            st.session_state.messages.append({"role": "assistant", "content": result["response"]})
    
    st.rerun()