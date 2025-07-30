# AILib

一个简单、直观的 Python SDK，用于构建基于 LLM 的应用程序，支持链式调用、智能代理和工具集成。

**设计理念**：Vercel AI SDK 的简洁性 + LangChain 的强大功能 = AILib 🚀

## 特性

-   🚀 **简洁 API**：受 Vercel AI SDK 启发 - 最少的样板代码，最高的生产力
-   🔗 **链式调用**：通过流畅的 API 实现顺序提示执行
-   🤖 **智能代理**：支持工具使用的 ReAct 风格自主代理
-   🛠️ **工具系统**：使用装饰器轻松创建工具，支持类型安全
-   📝 **提示模板**：强大的提示模板系统
-   💾 **会话管理**：对话状态和内存管理
-   🔒 **类型安全**：完整的类型提示和可选的 Pydantic 验证
-   🛡️ **安全保护**：内置内容审核和安全钩子
-   📊 **追踪系统**：全面的可观测性和调试支持
-   ⚡ **异步支持**：同时支持同步和异步 API

## 安装

```bash
# 基础安装
pip install ailib

# 包含所有可选依赖
pip install ailib[all]

# 特定功能安装
pip install ailib[dev,test]  # 用于开发
pip install ailib[tracing]   # 用于高级追踪
```

### 开发环境设置

用于开发时，克隆仓库并安装开发依赖：

```bash
# 克隆仓库
git clone https://github.com/kapuic/ailib.git
cd ailib

# 使用 uv 创建虚拟环境（推荐）
uv venv
source .venv/bin/activate  # Windows 系统使用：.venv\Scripts\activate

# 以开发模式安装所有依赖
uv pip install -e ".[dev,test]"

# 安装 pre-commit 钩子
pre-commit install

# 运行格式化和检查工具
make format  # 使用 black 和 isort 格式化代码
make lint    # 检查代码风格
```

## 快速开始

### 简单的文本补全

```python
from ailib import OpenAIClient, Prompt

# 初始化客户端
client = OpenAIClient(model="gpt-3.5-turbo")

# 创建提示
prompt = Prompt()
prompt.add_system("你是一个有帮助的助手。")
prompt.add_user("法国的首都是什么？")

# 获取补全
response = client.complete(prompt.build())
print(response.content)
```

### 使用链式调用 - 简化方式

```python
from ailib import create_chain

# 使用简化 API 创建链 - 无需客户端！
chain = create_chain(
    "你是一个有帮助的助手。",
    "{country}的首都是什么？",
    "那里的人口是多少？"
)

result = chain.run(country="法国")
print(result)
```

### 创建工具

```python
from ailib import tool

@tool
def weather(city: str) -> str:
    """获取城市的天气。"""
    return f"{city}的天气是晴天，温度 25°C"

@tool
def calculator(expression: str) -> float:
    """计算数学表达式。"""
    return eval(expression)
```

### 使用代理 - 简化方式

```python
from ailib import create_agent

# 使用简化 API 创建代理
agent = create_agent(
    "助手",
    tools=[weather, calculator],
    model="gpt-4"
)

# 运行代理
result = agent.run("巴黎的天气如何？另外，85 的 15% 是多少？")
print(result)
```

### 会话管理 - 简化方式

```python
from ailib import create_session, OpenAIClient

# 创建带验证的会话
session = create_session(
    session_id="tutorial-001",
    metadata={"user": "学生"}
)

client = OpenAIClient()

# 添加消息
session.add_system_message("你是一个有帮助的导师。")
session.add_user_message("解释量子计算")

# 获取带上下文的响应
response = client.complete(session.get_messages())
session.add_assistant_message(response.content)

# 存储记忆
session.set_memory("topic", "量子计算")
session.set_memory("level", "初学者")
```

## 为什么选择 AILib？

AILib 遵循 **Vercel AI SDK** 的理念而非 LangChain：

-   **默认简单**：一行代码即可开始，而不是多页配置
-   **渐进式披露**：需要时提供复杂性，不需要时隐藏
-   **类型安全**：完整的 TypeScript 风格类型提示和可选的运行时验证
-   **生产就绪**：内置安全、追踪和错误处理

```python
# LangChain 风格（冗长）
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="为生产{product}的公司起个好名字？"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("彩色袜子")

# AILib 风格（简洁）
from ailib import create_chain

chain = create_chain("为生产{product}的公司起个好名字？")
result = chain.run(product="彩色袜子")
```

## 高级功能

### 安全和内容审核

AILib 包含内置的安全功能，确保负责任的 AI 使用：

```python
from ailib.safety import enable_safety, with_moderation

# 启用全局安全检查
enable_safety(
    block_harmful=True,
    max_length=4000,
    sensitive_topics=["暴力", "仇恨"]
)

# 使用 OpenAI 审核
pre_hook, post_hook = with_moderation()

# 直接检查内容
from ailib.safety import check_content
is_safe, violations = check_content("要检查的文本")
```

### 追踪和可观测性

全面的追踪支持，用于调试和监控：

```python
from ailib.tracing import get_trace_manager

# 代理和链的自动追踪
agent = create_agent("助手", verbose=True)
result = agent.run("复杂任务")  # 自动追踪

# 访问追踪数据
manager = get_trace_manager()
trace = manager.get_trace(trace_id)
print(trace.to_dict())  # 完整的执行历史
```

### 速率限制

内置速率限制防止滥用：

```python
from ailib.safety import set_rate_limit, check_rate_limit

# 设置速率限制：每分钟每用户 10 个请求
set_rate_limit(max_requests=10, window_seconds=60)

# 在发出请求前检查
if check_rate_limit("user-123"):
    result = agent.run("查询")
else:
    print("超过速率限制")
```

## 工厂函数 vs 直接实例化

AILib 提供两种创建对象的方式：

1. **工厂函数**（推荐）：简单、经过验证且安全

    ```python
    agent = create_agent("助手", temperature=0.7)
    chain = create_chain("提示模板")
    session = create_session(max_messages=100)
    ```

2. **直接实例化**：更多控制，无验证
    ```python
    agent = Agent(llm=client, temperature=5.0)  # 无验证！
    ```

使用工厂函数以确保安全，使用直接实例化以获得灵活性。

## 教程

在 `examples/tutorials/` 目录中提供了全面的教程：

1. **[设置和安装](examples/tutorials/01_setup_and_installation.ipynb)** - AILib 入门
2. **[基本 LLM 补全](examples/tutorials/02_basic_llm_completions.ipynb)** - 进行第一次 API 调用
3. **[提示模板](examples/tutorials/03_prompt_templates.ipynb)** - 构建动态提示
4. **[提示构建器](examples/tutorials/04_prompt_builder.ipynb)** - 以编程方式构建对话
5. **[会话管理](examples/tutorials/05_session_management.ipynb)** - 管理对话状态
6. **[链式调用](examples/tutorials/06_chains.ipynb)** - 构建顺序工作流
7. **[工具和装饰器](examples/tutorials/07_tools_and_decorators.ipynb)** - 创建可重用工具
8. **[代理](examples/tutorials/08_agents.ipynb)** - 构建自主 AI 代理
9. **[高级功能](examples/tutorials/09_advanced_features.ipynb)** - 异步、流式传输和优化
10. **[实际示例](examples/tutorials/10_real_world_examples.ipynb)** - 完整的应用程序

🆕 **新增**：查看我们的[简化 API 示例](examples/simplified_api_example.py)，展示新的工厂函数！

从 **[教程索引](examples/tutorials/00_index.ipynb)** 开始，获得指导性学习路径。

## 最佳实践

1. **使用环境变量**存储 API 密钥：

    ```bash
    export OPENAI_API_KEY="your-key"
    ```

2. **启用详细模式**进行调试：

    ```python
    # 使用工厂函数
    agent = create_agent("助手", verbose=True)
    chain = create_chain("模板", verbose=True)

    # 或使用流畅 API
    chain.verbose(True)
    ```

3. **为代理设置适当的 max_steps** 以防止无限循环

4. **使用会话**维护对话上下文

5. **为工具函数添加类型注释**以获得更好的验证和文档

6. **在生产环境中使用安全功能**

7. **启用追踪**以调试复杂的工作流

## 项目状态

AILib 正在积极开发中。当前版本包括：

-   ✅ 核心 LLM 客户端抽象
-   ✅ 链和代理实现
-   ✅ 带装饰器的工具系统
-   ✅ 会话管理
-   ✅ 安全和审核钩子
-   ✅ 全面的追踪
-   ✅ 完整的异步支持
-   🔄 更多 LLM 提供商（即将推出）
-   🔄 向量存储集成（即将推出）
-   🔄 流式支持（即将推出）

## 相关项目

-   [LangChain](https://github.com/langchain-ai/langchain) - 全面但复杂
-   [Vercel AI SDK](https://github.com/vercel/ai) - 我们简洁性的灵感来源
-   [AutoGen](https://github.com/microsoft/autogen) - 多代理对话
-   [CrewAI](https://github.com/joaomdmoura/crewAI) - 代理协作

## 技术评估背景

本项目最初作为一个技术评估提交给 Moonshot AI。评估的主题是"代理架构"，评估要求包括：

-   流行的 Agent Workflow 有哪些？
-   Workflow 的 Structure 指什么？
-   如何让 Agent 最终输出符合预期的内容
-   在 Agent 执行过程中，应如何制定内容安全策略
-   如何进行 Tracing

AILib 展示了所有这些概念的生产就绪实现，同时保持了 Vercel AI SDK 风格的简洁 API。

## 系统要求

-   Python >= 3.10
-   OpenAI API 密钥（用于 OpenAI 模型）

## 许可证

MIT 许可证 - 详见 LICENSE 文件

## 贡献

欢迎贡献！请参阅 CONTRIBUTING.md 了解指南。

## 致谢

由 Kapui Cheung 创建，展示现代 Python SDK 设计，结合了 Vercel AI SDK 的简洁性和 LangChain 的强大功能。
