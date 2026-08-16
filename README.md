# Agentic AI Research & Content Generation Agent

An agentic AI system that autonomously researches user-provided topics using web search, synthesizes information, generates structured blog content and relevant visuals, and exports the final result as a Markdown document.

## 🚀 Key Features

- 🔎 Autonomous web-based topic research
- 🤖 LangGraph-based agentic workflow orchestration
- 🧠 LLM-powered research analysis and content generation
- 📊 Automatic research timeline and key-concept extraction
- 👤 Human-in-the-Loop validation and approval
- ✍️ Structured blog generation in Markdown
- 🖼️ AI-powered image generation
- 📈 LangSmith tracing and workflow observability
- 💻 Interactive Streamlit interface

## 🏗️ System Architecture

![Architecture](docs/architecture.png)

## 🔄 Workflow

```text
User Topic
    ↓
Streamlit Interface
    ↓
LangGraph Workflow
    ↓
Web Research
    ↓
Research Analysis
    ↓
Key Concepts + Timeline
    ↓
Human Review / Approval
    ↓
Blog Content Generation
    ↓
AI Image Generation
    ↓
Markdown Output
