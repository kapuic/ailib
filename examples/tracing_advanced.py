"""Advanced tracing example with OpenTelemetry integration."""

import os
import time

from ailib.agents import tool
from ailib.tracing import (
    enable_otel,
    export_trace,
    get_trace,
    start_trace,
    trace_step,
    traced,
)


# Example tools for demonstration
@tool
def fetch_user_data(user_id: str) -> dict:
    """Fetch user data from database."""
    # Simulate database query
    time.sleep(0.05)  # 50ms
    return {
        "id": user_id,
        "name": "John Doe",
        "email": "john@example.com",
        "preferences": {"language": "en", "timezone": "UTC"},
    }


@tool
def send_notification(user_id: str, message: str) -> bool:
    """Send notification to user."""
    # Simulate API call
    time.sleep(0.03)  # 30ms
    return True


def example_otel_integration():
    """Demonstrate OpenTelemetry integration."""
    print("=== OpenTelemetry Integration Example ===\n")

    # Enable OpenTelemetry export
    # This would normally send to a real collector like Jaeger or Tempo
    try:
        enable_otel(
            endpoint=os.getenv("OTEL_ENDPOINT", "http://localhost:4317"),
            service_name="ailib_demo",
        )
        print("✓ OpenTelemetry enabled")
    except ImportError:
        print(
            "⚠️  OpenTelemetry not available - install with: pip install ailib[tracing]"
        )
        print("   Continuing with default tracing...\n")

    # Create a complex workflow
    start_trace("user_notification_workflow")

    # Step 1: Fetch user data
    with trace_step("fetch_users", step_type="database") as fetch_step:
        users = []
        for user_id in ["user1", "user2", "user3"]:
            user_data = fetch_user_data(user_id)
            users.append(user_data)

        fetch_step.metadata["user_count"] = len(users)

    # Step 2: Process notifications
    with trace_step("process_notifications", step_type="business_logic"):
        # Process each user
        for user in users:
            with trace_step(
                f"notify_user_{user['id']}",
                step_type="notification",
                user_id=user["id"],
            ):
                # Generate personalized message
                message = f"Welcome {user['name']}!"  # Simulate agent response

                # Send notification
                send_notification(user["id"], message)

    # Get trace summary
    trace = get_trace()
    if trace:
        print(f"\nWorkflow completed with {len(trace.steps)} steps")
        print(f"Total duration: {trace.duration_ms:.1f}ms")


def example_distributed_tracing():
    """Demonstrate distributed tracing across services."""
    print("\n=== Distributed Tracing Example ===\n")

    # Start parent trace
    parent_trace = start_trace("distributed_operation")

    # Simulate service A
    with trace_step("service_a_processing", step_type="service"):
        # Simulate processing
        time.sleep(0.02)

        # Pass trace context to service B (in real app, via headers)
        trace_context = {
            "trace_id": parent_trace.trace_id,
            "parent_step_id": parent_trace._current_step_id,
        }

    # Simulate service B (would normally be separate process)
    with trace_step(
        "service_b_processing",
        step_type="service",
        parent_id=trace_context.get("parent_step_id"),
    ):
        # Service B does its work
        time.sleep(0.03)

    print("✓ Distributed trace captured across services")


def example_performance_monitoring():
    """Demonstrate performance monitoring with traces."""
    print("\n=== Performance Monitoring Example ===\n")

    # Create a chain with tracing
    start_trace("chain_performance_test")

    # Add steps with different complexities
    @traced(name="data_preprocessing", include_args=True)
    def preprocess(text: str) -> str:
        time.sleep(0.01)  # Simulate work
        return text.lower().strip()

    @traced(name="enrichment")
    def enrich(text: str) -> dict:
        time.sleep(0.02)  # Simulate API call
        return {"text": text, "metadata": {"source": "api", "confidence": 0.95}}

    @traced(name="validation")
    def validate(data: dict) -> dict:
        time.sleep(0.005)  # Quick validation
        data["validated"] = True
        return data

    # Run chain multiple times
    results = []
    for i in range(5):
        with trace_step(f"chain_run_{i}", step_type="iteration"):
            text = f"Sample text {i}"
            processed = preprocess(text)
            enriched = enrich(processed)
            validated = validate(enriched)
            results.append(validated)

    # Analyze performance
    trace = get_trace()
    if trace:
        # Group steps by name
        step_times = {}
        for step in trace.steps:
            if step.name not in step_times:
                step_times[step.name] = []
            if step.duration_ms:
                step_times[step.name].append(step.duration_ms)

        print("Performance Analysis:")
        for name, times in step_times.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"  {name}: avg={avg_time:.1f}ms, count={len(times)}")


def example_error_tracing():
    """Demonstrate error capture in traces."""
    print("\n=== Error Tracing Example ===\n")

    start_trace("error_handling_demo")

    # Simulate operations with potential errors
    operations = [
        ("operation_1", True),  # Success
        ("operation_2", False),  # Will fail
        ("operation_3", True),  # Success
    ]

    for op_name, should_succeed in operations:
        try:
            with trace_step(op_name, step_type="operation"):
                if not should_succeed:
                    raise ValueError(f"{op_name} failed!")
                time.sleep(0.01)
                print(f"✓ {op_name} succeeded")
        except ValueError as e:
            print(f"✗ {op_name} failed: {e}")

    # Export trace with errors
    export_trace()
    trace = get_trace()

    if trace:
        error_count = sum(1 for step in trace.steps if step.error)
        print(f"\nTrace captured {error_count} errors out of {len(trace.steps)} steps")


if __name__ == "__main__":
    print("AILib Advanced Tracing Examples")
    print("=" * 50)

    # Run examples
    example_otel_integration()
    example_distributed_tracing()
    example_performance_monitoring()
    example_error_tracing()

    # Export final trace
    export_trace("advanced_trace.json")

    print("\n✅ Advanced tracing examples completed!")
    print("\nKey Features Demonstrated:")
    print("- OpenTelemetry integration for production observability")
    print("- Distributed tracing across services")
    print("- Performance monitoring and analysis")
    print("- Error capture and debugging")
    print("- Nested trace steps for complex workflows")
