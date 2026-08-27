from langgraph.graph import StateGraph, START, END
from app.graph.state import VoiceCallState
from app.graph.nodes.tts import text_to_speech
from app.graph.nodes.stt import speech_to_text
from app.graph.nodes.fast_reply_llm import fast_reply_llm
from app.graph.nodes.language import detect_language
from app.graph.nodes.lead_extraction import lead_extraction
from app.graph.nodes.hot_action import hot_action
from app.graph.nodes.warm_action import warm_action
from app.graph.routing import conversation_should_continue, decide_hot_warm
import os


builder= StateGraph(VoiceCallState)

builder.add_node('speech_to_text', speech_to_text)
builder.add_node("fast_reply_llm", fast_reply_llm)
builder.add_node("detect_language", detect_language)
builder.add_node("lead_extraction", lead_extraction)
builder.add_node("hot_action", hot_action)
builder.add_node("warm_action", warm_action)
builder.add_node("text_to_speech", text_to_speech)



builder.add_edge(START, "speech_to_text")
builder.add_edge("speech_to_text", "fast_reply_llm")
builder.add_conditional_edges("fast_reply_llm", conversation_should_continue, {"connected": "text_to_speech", "disconnected":END})
builder.add_edge("text_to_speech",END)

builder.add_edge("speech_to_text", "detect_language")
builder.add_edge("detect_language", "lead_extraction")
builder.add_conditional_edges("lead_extraction", decide_hot_warm)
builder.add_edge("hot_action",END)
builder.add_edge("warm_action", END)


graph= builder.compile()
png= graph.get_graph().draw_mermaid_png()

path= os.path.join(os.getcwd(), "app\graph\graph.png")

with open(path, "wb") as f:
    f.write(png)
