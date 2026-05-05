// ═══════════════════════════════════════════════════════════════════════════
//  OpenAI API Client – Minimal Implementation Demo
//
//  This file demonstrates all the features of the custom framework:
//    1. Basic streaming chat
//    2. Tool/function calling with reflection-based invocation
//    3. Agentic loop (automatic multi-turn tool call resolution)
//    4. Image attachment
//    5. Chat history
//    6. Request/Response interceptors (provider-specific customisation)
//    7. Reasoning content display (DeepSeek R1, QwQ, etc.)
// ═══════════════════════════════════════════════════════════════════════════

using SmartBlocks.OpenAIFramework;

// ── Provider configuration ───────────────────────────────────────────────
// Edit these to switch between providers. The keys are real (live) keys.

// OpenRouter (default – general purpose, supports many models)
var endpoint = "https://openrouter.ai/api/v1/";
var modelName = "qwen/qwen3.6-27b";
var apiKey = "";

// DeepSeek – reasoning model (supports "reasoning" field in delta)
// var endpoint = "https://api.deepseek.com";
// var modelName = "deepseek-v4-pro";
// var apiKey = "";

// Ollama (local) – uncomment to test with local models
// var endpoint = "http://localhost:11434/v1/";
// var modelName = "gemma4:e4b"; // "qwen3.6:27b";
// var apiKey = ""; // Ollama doesn't require a key, but the Bearer header is harmless


Console.WriteLine($"Endpoint : {endpoint}");
Console.WriteLine($"Model     : {modelName}");
if (!string.IsNullOrEmpty(apiKey)) Console.WriteLine($"API Key   : {apiKey[..8]}...{apiKey[^4..]}");
Console.WriteLine();

// ── Create the client with increased timeout ─────────────────────────────
using var client = new OpenAIClient(endpoint, modelName, apiKey);

// client.Options.ReasoningEffort = "high";
// client.Options.Thinking = new ThinkingConfig { Type = ThinkingType.Enabled };

// ── Register request interceptor ─────────────────────────────────────────
// Add provider-specific headers or modify the request before it is sent.
// NOTE: OpenRouter HTTP-Referer / X-Title headers are now auto-added by
//       the core OpenAIClient (ApplyProviderDefaultHeaders).
client.RequestInterceptor = request =>
{
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    Console.WriteLine($"\n[RequestInterceptor] {request.Method} {request.RequestUri}");
    Console.ResetColor();
};

// ── Register response interceptor ────────────────────────────────────────
// Inspect raw response data (e.g. for debugging provider-specific formats).
client.ResponseInterceptor = rawBody =>
{
    // Only log first 500 chars to avoid flooding
    var preview = rawBody.Length > 500 ? rawBody[..500] + "..." : rawBody;
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    Console.WriteLine($"\n[ResponseInterceptor] Received {rawBody.Length} chars of SSE data");
    Console.ResetColor();
};

// ── Register reasoning callback ──────────────────────────────────────────
// Reasoning content is displayed in DarkGray so it is visually distinct
// from the main response content.

// ── Register tools ───────────────────────────────────────────────────────
var weatherSchema = """
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string", "description": "City name, e.g. 'New York'" }
      },
      "required": ["location"]
    }
  }
}
""";

client.RegisterTool("get_weather", weatherSchema, (string location) =>
{
    if (location.Contains("NYC", StringComparison.OrdinalIgnoreCase) ||
        location.Contains("New York", StringComparison.OrdinalIgnoreCase))
        return "25°C, sunny";
    if (location.Contains("London", StringComparison.OrdinalIgnoreCase))
        return "15°C, cloudy";
    if (location.Contains("Paris", StringComparison.OrdinalIgnoreCase))
        return "22°C, partly cloudy";
    return "20°C, unknown";
});

var calcSchema = """
{
  "type": "function",
  "function": {
    "name": "calculate",
    "description": "Perform basic arithmetic",
    "parameters": {
      "type": "object",
      "properties": {
        "a": { "type": "integer", "description": "First operand" },
        "b": { "type": "integer", "description": "Second operand" },
        "op": { "type": "string", "description": "Operator: +, -, *, /" }
      },
      "required": ["a", "b", "op"]
    }
  }
}
""";

client.RegisterTool("calculate", calcSchema, (int a, int b, string op) =>
{
    return op switch
    {
        "+" => (a + b).ToString(),
        "-" => (a - b).ToString(),
        "*" => (a * b).ToString(),
        "/" => b != 0 ? (a / b).ToString() : "Error: division by zero",
        _ => $"Error: unknown operator '{op}'"
    };
});

// var message = await client.ChatAsync("What is the weather in London?");
// Console.ForegroundColor = ConsoleColor.DarkYellow;
// Console.WriteLine(message.ReasoningContent);
// Console.ResetColor();
// Console.WriteLine(message.Content);

// ── Set system prompt ──────────────────────────────────────────────────
client.History.Add(ChatMessage.System(
    "You are a helpful assistant with access to tools. " +
    "Use get_weather to check weather and calculate to perform arithmetic. " +
    "When you need to do both, make the tool calls and combine the results."));

// ── Demo 1: Simple chat with streaming ─────────────────────────────────
Console.WriteLine("═══ Demo 1: Simple streaming chat ═══");
Console.Write("User: What is 2 + 2?\nAssistant: ");
var response1 = await StreamingChat(client, "What is 2 + 2?");
Console.WriteLine($"\nAssistant message: {response1?.Content ?? "(no text)"}");
Console.WriteLine();

// ── Demo 2: Tool call (weather lookup) ─────────────────────────────────
Console.WriteLine("═══ Demo 2: Tool call (weather) ═══");
Console.Write("User: What's the weather in NYC?\nAssistant: ");
var response2 = await StreamingChat(client, "What's the weather in NYC?");
Console.WriteLine($"\nAssistant message: {response2?.Content ?? "(no text)"}");
Console.WriteLine();

// ── Demo 3: Agentic loop – multi-step tool calling ────────────────────
Console.WriteLine("═══ Demo 3: Agentic loop (multi-step) ═══");
Console.Write("User: Calculate 42 + 9, then tell me the weather in London.\nAssistant: ");
var response3 = await StreamingChat(client, "Calculate 42 + 9, then tell me the weather in London.");
Console.WriteLine($"\nAssistant message: {response3?.Content ?? "(no text)"}");
Console.WriteLine();

// // ── Demo 4: Image recognition ─────────────────────────────────────────
// Console.WriteLine("═══ Demo 4: Image recognition ═══");
// var imagePath = Path.Combine(AppContext.BaseDirectory, "Strawberry-Juice-min.png");
// if (File.Exists(imagePath))
// {
//     var imageBytes = await File.ReadAllBytesAsync(imagePath);
//     Console.Write("User: What is in this image?\nAssistant: ");
//     var response4 = await StreamingChat(client, "Describe image in few words.", imageBytes, "image/png");
//     Console.WriteLine($"\nAssistant message: {response4?.Content ?? "(no text)"}");
// }
// else
// {
//     Console.WriteLine($"Image not found at {imagePath} — skipping demo 4.");
// }
// Console.WriteLine();

// ── Show full history ─────────────────────────────────────────────────
Console.WriteLine("═══ Chat History ═══");
Console.WriteLine($"Total messages: {client.History.Count}");
foreach (var msg in client.History)
{
    var role = msg.Role.PadRight(10);
    var contentPreview = (msg.Content?.Length > 80 ? msg.Content[..80] + "..." : msg.Content) ?? "(empty)";
    var extra = msg.ToolCallsRaw != null ? " [has tool_calls]" :
                msg.ToolCallId != null ? $" [tool_result for {msg.ToolName}]" : "";
    var reasoningExtra = msg.ReasoningContent != null ? " [has reasoning]" : "";
    Console.WriteLine($"  [{role}] {contentPreview}{extra}{reasoningExtra}");
}

Console.WriteLine();
Console.WriteLine("Done.");


// ── Helper ─────────────────────────────────────────────────────────────
static async Task<ChatMessage?> StreamingChat(OpenAIClient client, string userMessage, byte[]? imageData = null, string? imageMimeType = null)
{
    ChatMessage? lastAssistant = null;
    try
    {
        await foreach (var chunk in client.ChatStreamingAsync(userMessage, imageData, imageMimeType))
        {
            if (client.IsReasoning)
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.Write(chunk);
                Console.ResetColor();
            }
            else
            {
                Console.Write(chunk);
            }
        }
        Console.WriteLine();

        lastAssistant = client.History.LastOrDefault(m => m.Role == "assistant");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[ERROR] {ex.Message}");
    }
    return lastAssistant;
}
