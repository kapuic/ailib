# Comparison

## Basic Prompt/Completion

**LangChain (Python)**

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Initialize an OpenAI chat model (e.g., GPT-3.5 Turbo)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is your name?")
]
response = llm(messages)  # invoke the LLM with the message list
print(response.content)  # e.g., "My name is LangChain Bot."
```

**LangChain (TypeScript)**

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { SystemMessage, HumanMessage } from "@langchain/core/messages";

const llm = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0 });
const messages = [
    new SystemMessage("You are a helpful assistant."),
    new HumanMessage("What is your name?"),
];
const response = await llm.invoke(messages);
console.log(response.content); // e.g., "My name is LangChain Bot."
```

**Vercel AI SDK**

```typescript
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

const result = await generateText({
    model: openai("gpt-3.5-turbo"), // specify the model
    system: "You are a helpful assistant.",
    prompt: "What is your name?",
    maxTokens: 50,
    temperature: 0,
});
console.log(result.text); // e.g., "My name is Vercel AI Assistant."
```

**Analysis:** In a basic LLM call, the Vercel AI SDK offers a **simplified functional API**. For example, one can call a single `generateText()` function with a model and prompts, and get the text result directly. In contrast, LangChain requires **instantiating an LLM client class and constructing message objects** (like `SystemMessage` and `HumanMessage`) before invocation. This reflects the design philosophies: **Vercel’s SDK prioritizes quick integration and minimal boilerplate**, which is great for simple chat functionality (it even provides React hooks like `useChat` for front-end streaming updates). **LangChain** is more verbose for a simple prompt, but its object-oriented approach (using message classes and LLM instances) provides structure and flexibility for more complex interactions. From a DX perspective, Vercel’s one-call interface feels **simpler and faster to get started**, while LangChain’s approach can seem **heavier** for basic use cases but lays groundwork for advanced workflows.

## Adding a Tool

**LangChain (Python)**

```python
from langchain.agents import tool

@tool
def get_temperature(city: str) -> str:
    """Get the current temperature in the given city."""
    # (Example implementation)
    temperature = "25°C"  # Here you'd call a weather API
    return f"{temperature}"
```

**LangChain (TypeScript)**

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

export const temperatureTool = tool(
    async ({ city }) => {
        // Example implementation (simulate a temperature lookup)
        const temperature = "25°C";
        return `${temperature}`;
    },
    {
        name: "temperature",
        description: "Gets current temperature in the given city",
        schema: z.object({
            city: z
                .string()
                .describe("The city to get the current temperature for"),
        }),
        responseFormat: "content",
    }
);
```

**Vercel AI SDK**

```typescript
import { tool } from "ai";
import { z } from "zod";

export const temperatureTool = tool({
    description: "Gets current temperature in the given city",
    parameters: z.object({
        city: z
            .string()
            .describe("The city to get the current temperature for"),
    }),
    execute: async ({ city }) => {
        // Example implementation (simulate a temperature lookup)
        const temperature = "25°C";
        return `${temperature}`;
    },
});
```

**Analysis:** Both libraries allow developers to **extend LLM capabilities with custom tools** (functions the model can call). LangChain (Python) provides a convenient `@tool` decorator that transforms a Python function into a Tool object with name and description drawn from the function name and docstring. In LangChain’s TypeScript API, a `tool()` function is used similarly, where you pass an async executor and a config specifying name, description, and an input schema (often using Zod for validation). The Vercel AI SDK likewise offers a `tool()` helper that takes a description, Zod-based `parameters` schema, and an `execute` function. One notable difference is that **LangChain’s tool definition can include extra metadata** like an explicit name and even output format (as seen in the TS example with `responseFormat`), whereas **Vercel’s `tool` API doesn’t require or allow specifying an output schema/format** – it assumes the tool returns a result or an error object. From a DX standpoint, defining a tool in both is fairly straightforward: Vercel’s approach is very declarative (all in one object), and LangChain’s Python decorator makes for clean syntax. LangChain’s TS version is a bit more verbose (two arguments to `tool()`), but it provides slightly more structure (e.g., enforcing output type if needed). Overall, **both libraries were designed to simplify tool creation**, using intuitive schema definitions (leveraging type systems, e.g., Pydantic in Python, Zod in TS). Vercel’s minimalism here aligns with its goal of hiding complexity from the developer, while LangChain’s tools integrate deeply into its ecosystem (which can pay off when combining tools with agents).

## Agent Orchestration (Using Tools)

**LangChain (Python)**

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
tools = [get_temperature]  # our tool from above (already decorated as a Tool)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Ask a question that may require the tool:
result = agent.run("What's the temperature in New York right now?")
print(result)
# The agent will decide to call get_temperature(city="New York") internally, then return the answer.
```

**LangChain (TypeScript)**

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { temperatureTool } from "./tools/temperatureTool"; // the tool we defined

const llm = new ChatOpenAI({ modelName: "gpt-4", temperature: 0 });
const agent = createReactAgent({
    llm,
    tools: [temperatureTool],
    prompt: "You are a helpful assistant. You can use tools to answer questions.",
});

const question = new HumanMessage("What's the temperature in New York?");
const result = await agent.invoke({ messages: [question] });
const finalAnswer = result.returned; // or process result messages list for final answer
console.log(finalAnswer);
```

**Vercel AI SDK**

```typescript
import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";
import { temperatureTool } from "./tools/temperatureTool"; // the tool we defined

const response = await generateText({
    model: openai("gpt-4"),
    system: "You are a helpful assistant. You can use tools to answer questions.",
    prompt: "What's the temperature in New York?",
    tools: { temperature: temperatureTool }, // make the tool available by name
    maxSteps: 2, // allow the model to take up to 2 turns (tool use + final answer)
    temperature: 0,
});
console.log(response.text);
```

**Analysis:** This section highlights how each library enables an **agentic interaction**, where the LLM can decide to use a tool to reach an answer. In LangChain (Python), this is done by initializing an agent with a list of tools and an LLM, typically using a standard agent type like `ZERO_SHOT_REACT_DESCRIPTION` which follows the ReAct framework. Once set up, calling `agent.run()` will let the model reason and use tools as needed. LangChain’s TypeScript support for agents is still evolving – in the example, we use `createReactAgent` from the LangChain’s LangGraph toolkit to simplify creating an agent with tools. Without LangGraph, orchestrating tool usage in LangChain.js can be verbose, and indeed the example above requires processing the agent’s message outputs to extract the final answer. In contrast, the Vercel AI SDK makes basic agent-like behavior quite accessible: you pass the `tools` in the `generateText` call and simply increase `maxSteps` to allow the model multiple turns. If the prompt or model’s reasoning decides a tool is needed, the SDK handles calling `temperatureTool` and then continues the model to produce a final answer, returning the result in `response.text`. This design shows Vercel’s **opinionated simplicity** – it implicitly runs a mini agent loop under the hood with minimal developer effort, whereas LangChain gives developers more **explicit control** over the agent’s logic and internals (at the cost of more code). From a DX perspective, if a developer wants a quick way to enable tool usage for an LLM, Vercel’s approach feels **lightweight** (just plug in the tool and go). LangChain’s approach is **more powerful** (supporting complex multi-step planning, custom agent types, callback hooks, etc.), but can be harder to grok and requires reading more docs or utility functions (especially in JS) to do the same. Notably, LangChain Python abstracts away most of the loop details once configured, whereas LangChain JS needed an auxiliary package (LangGraph) for convenience, indicating that **LangChain’s DX in Python is currently smoother than in TS** for agent workflows.

## Retrieval-Augmented Generation (RAG)

**LangChain (Python)**

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. Index documents into a vector store (FAISS in-memory example)
texts = ["Alice went to Wonderland.", "Dorothy visited Oz."]  # corpus documents
vector_store = FAISS.from_texts(texts, OpenAIEmbeddings())
retriever = vector_store.as_retriever(search_type="similarity", search_k=2)

# 2. Create a RetrievalQA chain that uses the retriever and an LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 3. Ask a question; the chain will auto-retrieve relevant text and include it in the LLM prompt
query = "Who went to Wonderland?"
answer = qa_chain.run(query)
print(answer)  # e.g., "It was Alice who went to Wonderland."
```

**LangChain (TypeScript)**

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "langchain/vectorstores"; // HNSWLib in-memory vector store
import { RetrievalQAChain } from "langchain/chains";
import { ChatOpenAI } from "@langchain/openai";

// 1. Create embeddings and vector store from documents
const texts = ["Alice went to Wonderland.", "Dorothy visited Oz."];
const vectorStore = await HNSWLib.fromTexts(texts, [], new OpenAIEmbeddings());
const retriever = vectorStore.asRetriever();

// 2. Set up a Retrieval QA chain with an OpenAI chat model
const llm = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0 });
const qaChain = RetrievalQAChain.fromLLM(llm, retriever);

// 3. Ask a question; the chain handles retrieval and prompting
const response = await qaChain.call({ query: "Who went to Wonderland?" });
console.log(response.text); // e.g., "Alice went to Wonderland."
```

**Vercel AI SDK**

```typescript
import { openai } from "@ai-sdk/openai";
import { embed, embedMany, tool, generateText } from "ai";
import { z } from "zod";

// Assume we have a function to search our knowledge base (vector DB) by question:
async function searchKnowledgeBase(question: string): Promise<string> {
    // Example: embed the question and query a vector store (not shown: actual DB operations)
    const { embedding } = await embed({
        model: openai.embedding("text-embedding-ada-002"),
        value: question,
    });
    // ... perform similarity search in your DB, get most relevant text or answer
    return "Alice went to Wonderland."; // (for illustration, we return a relevant fact)
}

// Define a tool that the model can use to get info from the knowledge base
const getInfoTool = tool({
    description: "Get information from the knowledge base to answer questions",
    parameters: z.object({
        question: z.string().describe("the user's question"),
    }),
    execute: async ({ question }) => {
        return await searchKnowledgeBase(question);
    },
});

// Use the tool in a generation call
const result = await generateText({
    model: openai("gpt-4"),
    system: `You are a knowledgeable assistant.
    You have access to a knowledge base. Always use the tool to fetch info before answering.`,
    prompt: "Who went to Wonderland?",
    tools: { getInfo: getInfoTool },
    maxSteps: 2,
});
console.log(result.text); // e.g., "It was Alice who went to Wonderland."
```

**Analysis:** _Retrieval-Augmented Generation_ is a pattern where the LLM is provided with external context (from a vector database or other knowledge store) relevant to the query. **LangChain** excels in this area by offering high-level abstractions: you can vectorize documents and wrap the vector store with a `retriever` in just a few lines, then use a `RetrievalQA` chain that automatically handles the process of finding relevant text and inserting it into the prompt. This means a developer can implement RAG without writing the low-level glue – LangChain handles chunking, embedding, similarity search (via integrations like FAISS, Pinecone, etc.), and combines it with the LLM call. On the other hand, the **Vercel AI SDK provides lower-level tools but no out-of-the-box RAG pipeline**. Developers using Vercel’s SDK must perform steps like text splitting, embedding (using functions like `embed` or `embedMany`), storing vectors in a database, and querying that store manually. In the example above, we sketched how one might use the SDK’s `embed` function to get an embedding and then create a custom search function. Vercel’s SDK can then incorporate that via a tool (as shown with `getInfoTool`), allowing the model to call it for information. This approach is **explicit and flexible** – the developer has full control over how data is retrieved – but it requires more work and careful design. In contrast, LangChain’s approach is **more turnkey** for RAG: you declare what you want (documents -> vectors -> retriever -> QA chain) and it manages the details. From a DX perspective, if your goal is to quickly stand up a QA system over your data, **LangChain’s higher-level utilities can significantly speed up development**. Vercel’s SDK, while lacking a one-liner solution for RAG, can still achieve the same result with manual coding, and some developers might appreciate the clarity of doing it themselves. Indeed, it’s common to **combine the two** – for example, using LangChain (or its LangChain.js variant) on the backend for document retrieval and processing, while using Vercel AI SDK on the frontend to handle the chat UI and streaming responses. This hybrid approach leverages each tool’s strengths: LangChain for complex data/agent logic and Vercel’s SDK for smooth developer experience in deployment and interaction.
