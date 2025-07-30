"""Basic tracing example - zero configuration required!"""

import json

from ailib import create_agent
from ailib.agents import tool
from ailib.tracing import export_trace, get_trace, start_trace, trace_step


# Define a simple tool
@tool
def search_weather(city: str) -> str:
    """Get weather information for a city."""
    # Simulate weather API call
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 60°F",
        "Tokyo": "Rainy, 68°F",
    }
    return weather_data.get(city, "Unknown city")


def example_automatic_tracing():
    """Demonstrate automatic tracing - no configuration needed!"""
    print("=== Automatic Tracing Example ===\n")

    # Create an agent - tracing happens automatically
    agent = create_agent(
        "weather_assistant",
        instructions="You are a helpful weather assistant.",
        model="gpt-3.5-turbo",
        tools=[search_weather],
    )

    # Run the agent - all LLM calls and tool usage are traced
    result = agent.run("What's the weather in New York and Tokyo?")
    print(f"Agent result: {result}\n")

    # Get the trace data
    trace = get_trace()
    if trace:
        print(f"Trace captured {len(trace.steps)} steps:")
        for i, step in enumerate(trace.steps):
            print(f"  {i+1}. {step.name} ({step.step_type}) - {step.duration_ms:.1f}ms")

        # Export trace to file
        export_trace("weather_trace.json")
        print("\nTrace exported to weather_trace.json")


def example_custom_tracing():
    """Demonstrate custom trace steps."""
    print("\n=== Custom Tracing Example ===\n")

    # Start a new trace
    start_trace("data_processing")

    # Use trace_step context manager
    with trace_step("load_data", step_type="io"):
        # Simulate loading data
        data = ["item1", "item2", "item3"]
        print("Data loaded")

    # Nested trace steps
    with trace_step("process_items", step_type="computation") as parent_step:
        processed = []
        for item in data:
            with trace_step(
                f"process_{item}",
                step_type="computation",
                parent_id=parent_step.step_id,
            ):
                # Simulate processing
                result = item.upper()
                processed.append(result)

        print(f"Processed {len(processed)} items")

    # Use traced decorator
    @trace_step("save_results", step_type="io")
    def save_data(data):
        # Simulate saving
        return f"Saved {len(data)} items"

    save_result = save_data(processed)
    print(save_result)

    # Get trace as JSON
    trace_json = export_trace()
    trace_data = json.loads(trace_json)
    print(f"\nCustom trace has {len(trace_data['steps'])} steps")


def example_trace_analysis():
    """Demonstrate trace analysis capabilities."""
    print("\n=== Trace Analysis Example ===\n")

    # Start trace
    start_trace("analysis_demo")

    # Simulate multiple operations
    operations = ["query_db", "call_api", "process_data", "generate_report"]

    for op in operations:
        with trace_step(op, step_type="operation"):
            # Simulate work with varying durations
            import time

            time.sleep(0.1)  # 100ms

    # Get trace and analyze
    trace = get_trace()
    if trace:
        # Calculate statistics
        total_time = sum(step.duration_ms for step in trace.steps if step.duration_ms)
        avg_time = total_time / len(trace.steps) if trace.steps else 0

        print("Trace Analysis:")
        print(f"  Total steps: {len(trace.steps)}")
        print(f"  Total time: {total_time:.1f}ms")
        print(f"  Average step time: {avg_time:.1f}ms")

        # Find slowest step
        slowest = max(trace.steps, key=lambda s: s.duration_ms or 0)
        print(f"  Slowest step: {slowest.name} ({slowest.duration_ms:.1f}ms)")


if __name__ == "__main__":
    # Note: Set OPENAI_API_KEY environment variable or pass api_key to client

    print("AILib Tracing Examples")
    print("=" * 50)

    # Run examples
    try:
        example_automatic_tracing()
    except Exception as e:
        print(f"Skipping automatic tracing example: {e}")

    example_custom_tracing()
    example_trace_analysis()

    print("\n✅ Tracing examples completed!")
    print("\nKey Takeaways:")
    print("- Tracing is enabled by default - zero configuration")
    print("- LLM calls and tool usage are traced automatically")
    print("- Custom trace steps can be added with context managers")
    print("- Traces can be exported as JSON for analysis")
    print("- Minimal performance overhead with smart defaults")
