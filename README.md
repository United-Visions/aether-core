# AetherMind - Open Core

**The Backbone of the Digital Organism Architecture.**

AetherMind Core is the open-source foundation of the **Developmental Continual Learning Architecture (DCLA)**. This repository contains the "Brain," "Mind," and "Orchestrator" logic that powers the Aether AGI system.

---

## 🧬 Architecture Overview

AetherMind is designed as a split-brain cognitive architecture. The core logic is distributed across four primary modules:

- **Brain (`/brain`)**: Fixed reasoning logic, logic engine abstractions (LiteLLM), and the Safety Inhibitor.
- **Mind (`/mind`)**: Memory management systems, including episodic memory and vector store (Pinecone) integrations.
- **Body (`/body`)**: Hardware and interface adapters (GPIO, Serial, Chat UI) that allow the brain to interact with the world.
- **Orchestrator (`/orchestrator`)**: The "Nervous System" that manages the Active Inference Loop and session state.

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Pinecone Account** (for Vector DB)
- **Redis Server** (for state management and queues)
- **LiteLLM Compatible API Keys** (OpenAI, Anthropic, or Gemini)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/United-Visions/aether-core.git
   cd aether-core
   ```

2. **Setup Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Configure Variables:**
   Create a `.env` file in the root directory:
   ```env
   PINECONE_API_KEY=your_key_here
   REDIS_URL=redis://localhost:6379
   OPENAI_API_KEY=your_key_here  # Or your preferred provider
   ```

## 🔄 Core Concepts

### The Active Inference Loop
The core intelligence logic resides in `orchestrator/active_inference.py`. It follows a continuous cycle:
`Sense → Feel → Reason → Embellish → Parse → Execute → Learn`.

### Action Tag System
AetherMind uses an XML-based action system to bridge reasoning and execution. Developers can extend capabilities by adding new tags in `orchestrator/action_parser.py` (Proprietary Layer) and implementing handlers in `body/adapters/`.

## 🤝 Contributing

We welcome contributions to the AetherMind Core! Whether it's fixing bugs, improving the memory promoter, or adding new hardware adapters.

### Contribution Workflow
1. **Fork** the repository.
2. **Create a Feature Branch** (`git checkout -b feature/amazing-logic`).
3. **Commit Your Changes** (`git commit -m 'Add some amazing logic'`).
4. **Push to the Branch** (`git push origin feature/amazing-logic`).
5. **Open a Pull Request**.

### Coding Standards
- **Async First:** Most core logic is asynchronous. Ensure you use `async/await` and non-blocking libraries.
- **Type Hinting:** All new functions must include Python type hints.
- **Safety First:** Any changes to reasoning or hardware interaction MUST pass through the `SafetyInhibitor` located in `brain/safety_inhibitor.py`.

## 🛡️ License & Open Core Strategy
AetherMind follows an **Open Core (70/30)** model.
- **70% (This Repo):** The fundamental reasoning engine, memory management, and interface adapters are open-source.
- **30% (Proprietary):** The "Heart" (Emotional Valence), "ToolForge" (Autonomous Tool Generation), and specialized AGI Meta-Controllers remain proprietary to United Visions.

For more details on our philosophy, see the [Open Core Strategy](https://github.com/United-Visions/AetherAGI/blob/main/docs/strategy/02_open_core_strategy.md).

---
Built with ❤️ by [United Visions](https://github.com/United-Visions).
