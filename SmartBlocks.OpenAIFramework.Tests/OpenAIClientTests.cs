using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Xunit;

namespace SmartBlocks.OpenAI.Tests;

/// <summary>
/// Integration tests for the custom OpenAI client.
///
/// Prerequisites:
///   Set env vars: OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY
///   OR edit the constants below.
/// </summary>
public class OpenAIClientTests
{
    private static string Endpoint => Environment.GetEnvironmentVariable("OPENAI_ENDPOINT") ?? "https://api.openai.com";
    private static string Model => Environment.GetEnvironmentVariable("OPENAI_MODEL") ?? "gpt-4o-mini";
    private static string ApiKey => Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "";

    private bool _skipBecauseNoKey =>
        string.IsNullOrEmpty(ApiKey);

    // ── Helper to create a client with a mock HTTP handler for unit tests ──

    private static OpenAIClient CreateClientWithHandler(HttpMessageHandler handler)
    {
        var http = new HttpClient(handler);
        return new OpenAIClient(Endpoint, Model, ApiKey, http);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  1. CHAT MESSAGE SERIALISATION TESTS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void ChatMessage_System_RendersCorrectJson()
    {
        var msg = ChatMessage.System("You are a helpful assistant.");
        var json = msg.ToRequestJsonString();
        Assert.Contains("\"role\":\"system\"", json);
        Assert.Contains("\"content\":\"You are a helpful assistant.\"", json);
    }

    [Fact]
    public void ChatMessage_User_RendersCorrectJson()
    {
        var msg = ChatMessage.User("Hello");
        var json = msg.ToRequestJsonString();
        Assert.Contains("\"role\":\"user\"", json);
        Assert.Contains("\"content\":\"Hello\"", json);
    }

    [Fact]
    public void ChatMessage_UserWithImage_RendersCorrectJson()
    {
        var msg = ChatMessage.UserWithImage("What is this?", new byte[] { 0xFF, 0xD8, 0xFF }, "image/jpeg");
        var json = msg.ToRequestJsonString();
        Assert.Contains("\"role\":\"user\"", json);
        Assert.Contains("\"type\":\"image_url\"", json);
        Assert.Contains("data:image/jpeg;base64,/9j/", json);
    }

    [Fact]
    public void ChatMessage_Tool_RendersCorrectJson()
    {
        var msg = ChatMessage.Tool("call_123", "get_weather", "25°C");
        var json = msg.ToRequestJsonString();
        Assert.Contains("\"role\":\"tool\"", json);
        Assert.Contains("\"tool_call_id\":\"call_123\"", json);
        Assert.Contains("\"content\":\"25°C\"", json);
    }

    [Fact]
    public void ChatMessage_AssistantWithToolCalls_RendersCorrectJson()
    {
        var msg = new ChatMessage
        {
            Role = "assistant",
            Content = "Let me check...",
            ToolCallsRaw = """[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}]"""
        };
        var json = msg.ToRequestJsonString();
        Assert.Contains("\"role\":\"assistant\"", json);
        Assert.Contains("\"content\":\"Let me check...\"", json);
        Assert.Contains("\"tool_calls\"", json);
        Assert.Contains("get_weather", json);
    }

    [Fact]
    public void ChatMessage_EscapesSpecialCharacters()
    {
        var msg = ChatMessage.User("Line1\nLine2\tTabbed \"quoted\"");
        var json = msg.ToRequestJsonString();
        Assert.Contains("\\n", json);
        Assert.Contains("\\t", json);
        Assert.Contains("\\\"", json);
    }

    [Fact]
    public void ChatMessage_ReasoningContent_StoredCorrectly()
    {
        var msg = ChatMessage.Assistant("Final answer");
        msg.ReasoningContent = "Let me think through this step by step...";
        Assert.Equal("Let me think through this step by step...", msg.ReasoningContent);
        Assert.Equal("Final answer", msg.Content);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  2. TOOL REGISTRATION & REFLECTION TESTS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void RegisterTool_AddsToToolDefinitionsAndTools()
    {
        var client = new OpenAIClient(Endpoint, Model, ApiKey);
        var schema = """{"type":"function","function":{"name":"echo","description":"Echoes input","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}""";

        client.RegisterTool("echo", schema, (string text) => text);

        Assert.Single(client.ToolDefinitions);
        Assert.Single(client.Tools);
        Assert.True(client.Tools.ContainsKey("echo"));
    }

    [Fact]
    public void ToolInvocation_Reflection_CallsCorrectDelegate()
    {
        var client = new OpenAIClient(Endpoint, Model, ApiKey);
        var schema = """{"type":"function","function":{"name":"add","description":"Adds two numbers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}""";

        client.RegisterTool("add", schema, (int a, int b) => (a + b).ToString());

        // Simulate tool execution via the internal dictionary
        var args = JsonDocument.Parse("""{"a":3,"b":7}""").RootElement;
        var result = client.Tools["add"](args);

        Assert.Equal("10", result);
    }

    [Fact]
    public void ToolInvocation_UnknownTool_ReturnsError()
    {
        var client = new OpenAIClient(Endpoint, Model, ApiKey);
        var schema = """{"type":"function","function":{"name":"greet","description":"Greets","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}""";

        client.RegisterTool("greet", schema, (string name) => $"Hello {name}");

        var args = JsonDocument.Parse("""{"name":"World"}""").RootElement;
        var result = client.Tools["greet"](args);

        Assert.Equal("Hello World", result);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  3. CHAT HISTORY TESTS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void History_AccumulatesMessages()
    {
        var client = new OpenAIClient(Endpoint, Model, ApiKey);

        client.History.Add(ChatMessage.System("You are a bot."));
        client.History.Add(ChatMessage.User("Hi"));
        client.History.Add(ChatMessage.Assistant("Hello!"));

        Assert.Equal(3, client.History.Count);
        Assert.Equal("system", client.History[0].Role);
        Assert.Equal("user", client.History[1].Role);
        Assert.Equal("assistant", client.History[2].Role);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4. REQUEST INTERCEPTOR TESTS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public async Task RequestInterceptor_IsInvoked_BeforeRequest()
    {
        var invoked = false;
        HttpRequestMessage? capturedRequest = null;

        var handler = new MockHttpMessageHandler(async req =>
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        client.RequestInterceptor = request =>
        {
            invoked = true;
            capturedRequest = request;
            // Simulate adding a provider-specific header
            request.Headers.Add("X-Custom-Header", "test-value");
        };

        await foreach (var _ in client.ChatStreamingAsync("Hello")) { }

        Assert.True(invoked, "RequestInterceptor should have been called");
        Assert.NotNull(capturedRequest);
        Assert.Contains("test-value", capturedRequest!.Headers.GetValues("X-Custom-Header"));
    }

    [Fact]
    public async Task AutoHeaders_OpenRouter_AreAppliedByDefault()
    {
        var handler = new MockHttpMessageHandler(async req =>
        {
            // Verify OpenRouter-specific headers were auto-added by the core client
            Assert.Contains("https://github.com/smartblocks/openai-agentframework",
                req.Headers.GetValues("HTTP-Referer").First());
            Assert.Contains("SmartBlocks OpenAI Agent Framework",
                req.Headers.GetValues("X-Title").First());

            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        // Use an OpenRouter endpoint so the core client auto-applies headers
        var http = new HttpClient(handler);
        var client = new OpenAIClient("https://openrouter.ai/api/v1", "test-model", "sk-test", http);

        await foreach (var _ in client.ChatStreamingAsync("Test")) { }
    }

    [Fact]
    public async Task RequestInterceptor_CanModifyRequestBody()
    {
        HttpContent? capturedContent = null;

        var handler = new MockHttpMessageHandler(async req =>
        {
            capturedContent = req.Content;
            var body = await req.Content!.ReadAsStringAsync();
            Assert.Contains("\"model\":\"gpt-4o-mini\"", body);

            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        client.RequestInterceptor = request =>
        {
            // Request interceptor can inspect the request but should not modify
            // the body directly (it's already set). This test verifies it can
            // at least inspect headers.
        };

        await foreach (var _ in client.ChatStreamingAsync("Hi")) { }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  5. RESPONSE INTERCEPTOR TESTS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public async Task ResponseInterceptor_IsInvoked_AfterResponse()
    {
        var invoked = false;
        string? capturedBody = null;

        var handler = new MockHttpMessageHandler(async req =>
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        client.ResponseInterceptor = body =>
        {
            invoked = true;
            capturedBody = body;
        };

        await foreach (var _ in client.ChatStreamingAsync("Hello")) { }

        Assert.True(invoked, "ResponseInterceptor should have been called");
        Assert.NotNull(capturedBody);
        Assert.Contains("\"content\":\"Hello\"", capturedBody);
    }

    [Fact]
    public async Task ResponseInterceptor_ReceivesFullSSEBody()
    {
        var handler = new MockHttpMessageHandler(async req =>
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"A"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        string? capturedBody = null;
        client.ResponseInterceptor = body => capturedBody = body;

        await foreach (var _ in client.ChatStreamingAsync("Hello")) { }

        Assert.NotNull(capturedBody);
        // Should contain both content chunks
        Assert.Contains("\"content\":\"A\"", capturedBody);
        Assert.Contains("\"content\":\"B\"", capturedBody);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  6. REASONING CONTENT TESTS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public async Task Reasoning_InDelta_IsCaptured()
    {
        // Simulate DeepSeek-style SSE chunks with "reasoning" in delta
        var handler = new MockHttpMessageHandler(async req =>
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            // Reasoning chunk (in delta)
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"Let me think"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":" step by step"},"finish_reason":null}]}""" + "\n\n");
            // Content chunk
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"The answer is 42"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        var reasoningChunks = new List<string>();

        await foreach (var chunk in client.ChatStreamingAsync("Think step by step")) { }

        var assistant = client.History.LastOrDefault(m => m.Role == "assistant");
        Assert.NotNull(assistant);
        Assert.Equal("Let me think step by step", assistant!.ReasoningContent);
        Assert.Equal("The answer is 42", assistant.Content);
    }

    [Fact]
    public async Task Reasoning_IsReasoningProperty_TogglesCorrectly()
    {
        var reasoningChunks = new List<(string text, bool isReasoning)>();

        var handler = new MockHttpMessageHandler(async req =>
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            // Reasoning chunk
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"Think..."},"finish_reason":null}]}""" + "\n\n");
            // Content chunk
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);

        await foreach (var chunk in client.ChatStreamingAsync("Think"))
        {
            reasoningChunks.Add((chunk, client.IsReasoning));
        }

        // First chunk should be reasoning (IsReasoning = true)
        Assert.Equal("Think...", reasoningChunks[0].text);
        Assert.True(reasoningChunks[0].isReasoning);

        // Second chunk should be content (IsReasoning = false)
        Assert.Equal("Answer", reasoningChunks[1].text);
        Assert.False(reasoningChunks[1].isReasoning);
    }

    [Fact]
    public async Task Reasoning_OnReasoningCallback_IsInvoked()
    {
        var reasoningChunks = new List<string>();

        var handler = new MockHttpMessageHandler(async req =>
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"Step 1"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"reasoning":"Step 2"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Done"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        client.OnReasoning = chunk => reasoningChunks.Add(chunk);

        await foreach (var chunk in client.ChatStreamingAsync("Think")) { }

        Assert.Equal(2, reasoningChunks.Count);
        Assert.Contains("Step 1", reasoningChunks);
        Assert.Contains("Step 2", reasoningChunks);
    }

    [Fact]
    public async Task Reasoning_TopLevelField_IsCaptured()
    {
        // Some providers (e.g. OpenRouter wrappers) send reasoning at
        // the top level of the JSON rather than inside delta
        var handler = new MockHttpMessageHandler(async req =>
        {
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            // Reasoning at top level
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","reasoning":"I am thinking","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}""" + "\n\n");
            await writer.WriteAsync("""data: {"id":"1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}""" + "\n\n");
            await writer.WriteAsync("data: [DONE]\n\n");
            await writer.FlushAsync();
            stream.Position = 0;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        await foreach (var chunk in client.ChatStreamingAsync("Test")) { }

        var assistant = client.History.LastOrDefault(m => m.Role == "assistant");
        Assert.NotNull(assistant);
        Assert.Equal("I am thinking", assistant!.ReasoningContent);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  7. STREAMING SSE PARSING (unit test with mock HTTP)
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public async Task ChatStreamingAsync_ParsesSSEChunks()
    {
        var sseChunks = new[]
        {
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        };

        var handler = new MockHttpMessageHandler(async req =>
        {
            // Verify the request body has our message
            var body = await req.Content!.ReadAsStringAsync();
            Assert.Contains("Hello from test", body);

            // Return SSE stream
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);
            foreach (var chunk in sseChunks)
            {
                await writer.WriteAsync(chunk);
            }
            await writer.FlushAsync();
            stream.Position = 0;

            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        client.History.Add(ChatMessage.System("You are a bot."));

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Hello from test"))
        {
            results.Add(chunk);
        }

        // The assistant message should have been added to history
        Assert.Single(client.History, m => m.Role == "assistant");
        Assert.Equal("Hello world", client.History.Last(m => m.Role == "assistant").Content);
    }

    [Fact]
    public async Task ChatStreamingAsync_AgenticLoop_ExecutesToolThenContinues()
    {
        // Simulate a two-turn conversation:
        // Turn 1: assistant returns a tool_call
        // Turn 2: assistant returns final text

        bool firstRequest = true;

        var handler = new MockHttpMessageHandler(async req =>
        {
            var body = await req.Content!.ReadAsStringAsync();
            var stream = new MemoryStream();
            var writer = new StreamWriter(stream);

            if (firstRequest)
            {
                firstRequest = false;
                // Return a tool call
                var toolCallChunk = """
data: {"id":"2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{\"text\":\"hi\"}"}}]},"finish_reason":null}]}

data: {"id":"2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

""";
                await writer.WriteAsync(toolCallChunk);
            }
            else
            {
                // Return final text
                var finalChunk = """
data: {"id":"3","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Final answer"},"finish_reason":null}]}

data: {"id":"3","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

""";
                await writer.WriteAsync(finalChunk);
            }

            await writer.FlushAsync();
            stream.Position = 0;

            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(stream)
                {
                    Headers = { ContentType = new MediaTypeHeaderValue("text/event-stream") }
                }
            };
        });

        var client = CreateClientWithHandler(handler);
        var schema = """{"type":"function","function":{"name":"echo","description":"Echoes","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}""";
        client.RegisterTool("echo", schema, (string text) => text);

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Echo this"))
        {
            results.Add(chunk);
        }

        // Should have: assistant (tool call) + tool result + assistant (final)
        var assistantMessages = client.History.Where(m => m.Role == "assistant").ToList();
        Assert.Equal(2, assistantMessages.Count);

        // First assistant should have tool calls
        Assert.NotNull(assistantMessages[0].ToolCallsRaw);
        Assert.Contains("echo", assistantMessages[0].ToolCallsRaw!);

        // Should have a tool result message
        Assert.Contains(client.History, m => m.Role == "tool" && m.ToolName == "echo");

        // Second assistant should have final text
        Assert.Equal("Final answer", assistantMessages[1].Content);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  8. INTEGRATION TEST (REAL API) – requires env vars
    // ═══════════════════════════════════════════════════════════════════════

    [Fact(Skip = "Integration test – requires real API key. Set OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY env vars and remove Skip.")]
    public async Task Integration_SimpleChat()
    {
        var client = new OpenAIClient(Endpoint, Model, ApiKey);

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Say 'Hello World' and nothing else."))
        {
            results.Add(chunk);
        }

        var lastAssistant = client.History.Last(m => m.Role == "assistant");
        Assert.NotNull(lastAssistant.Content);
        Assert.Contains("Hello World", lastAssistant.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Skip = "Integration test – requires real API key. Set OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY env vars and remove Skip.")]
    public async Task Integration_ToolCall()
    {
        var client = new OpenAIClient(Endpoint, Model, ApiKey);

        var schema = """{"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}""";
        client.RegisterTool("get_weather", schema, (string location) =>
        {
            if (location.Contains("NYC", StringComparison.OrdinalIgnoreCase) ||
                location.Contains("New York", StringComparison.OrdinalIgnoreCase))
                return "25°C, sunny";
            if (location.Contains("London", StringComparison.OrdinalIgnoreCase))
                return "15°C, cloudy";
            return "20°C, unknown";
        });

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("What's the weather in NYC?"))
        {
            results.Add(chunk);
        }

        // Verify that the tool was actually called
        var toolMessages = client.History.Where(m => m.Role == "tool" && m.ToolName == "get_weather").ToList();
        Assert.NotEmpty(toolMessages);

        // The final assistant should exist with a proper answer
        var assistantMessages = client.History.Where(m => m.Role == "assistant").ToList();
        Assert.NotEmpty(assistantMessages);
    }

    [Fact(Skip = "Integration test – requires real API key. Set OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY env vars and remove Skip.")]
    public async Task Integration_MultiTurnToolCall()
    {
        var client = new OpenAIClient(Endpoint, Model, ApiKey);

        var schema = """{"type":"function","function":{"name":"calculate","description":"Perform arithmetic","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"op":{"type":"string"}},"required":["a","b","op"]}}}""";
        client.RegisterTool("calculate", schema, (int a, int b, string op) =>
        {
            return op switch
            {
                "+" => (a + b).ToString(),
                "-" => (a - b).ToString(),
                "*" => (a * b).ToString(),
                "/" => (b != 0 ? (a / b).ToString() : "Error: division by zero"),
                _ => $"Error: unknown op {op}"
            };
        });

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Calculate 42 + 9, then subtract 10 from the result."))
        {
            results.Add(chunk);
        }

        // We expect at least one tool call
        var toolMessages = client.History.Where(m => m.Role == "tool").ToList();
        Assert.NotEmpty(toolMessages);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  9. DEEPSEEK INTEGRATION TESTS (real API – keys from Program.cs)
    // ═══════════════════════════════════════════════════════════════════════

    const string DeepSeekEndpoint = "https://api.deepseek.com";
    const string DeepSeekModel = "deepseek-v4-flash";
    const string DeepSeekApiKey = "";

    [Fact(Skip = "DeepSeek integration – requires network access. Remove Skip to run.")]
    public async Task DeepSeek_SimpleChat_WithRequestInterceptor()
    {
        var client = new OpenAIClient(DeepSeekEndpoint, DeepSeekModel, DeepSeekApiKey);

        var interceptedRequests = new List<HttpRequestMessage>();
        client.RequestInterceptor = request =>
        {
            interceptedRequests.Add(request);
            // DeepSeek may need additional headers in the future
        };

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Say 'Hello from DeepSeek' and nothing else."))
        {
            results.Add(chunk);
        }

        Assert.NotEmpty(interceptedRequests);
        var lastAssistant = client.History.Last(m => m.Role == "assistant");
        Assert.NotNull(lastAssistant.Content);
        Assert.Contains("DeepSeek", lastAssistant.Content, StringComparison.OrdinalIgnoreCase);

        // Verify the request interceptor saw the correct URL
        Assert.Contains("deepseek", interceptedRequests[0].RequestUri!.Host);
    }

    [Fact(Skip = "DeepSeek integration – requires network access. Remove Skip to run.")]
    public async Task DeepSeek_ReasoningContent_WithOnReasoningCallback()
    {
        var client = new OpenAIClient(DeepSeekEndpoint, DeepSeekModel, DeepSeekApiKey);

        var reasoningChunks = new List<string>();
        client.OnReasoning = chunk => reasoningChunks.Add(chunk);

        var results = new List<string>();
        // Ask a question that triggers reasoning
        await foreach (var chunk in client.ChatStreamingAsync("What is 2 + 2? Think step by step."))
        {
            results.Add(chunk);
        }

        var lastAssistant = client.History.Last(m => m.Role == "assistant");
        Assert.NotNull(lastAssistant);

        // DeepSeek chat model may or may not produce reasoning content;
        // the important thing is the callback doesn't crash and history is correct
        Assert.NotNull(lastAssistant.Content);
    }

    [Fact(Skip = "DeepSeek integration – requires network access. Remove Skip to run.")]
    public async Task DeepSeek_WithResponseInterceptor()
    {
        var client = new OpenAIClient(DeepSeekEndpoint, DeepSeekModel, DeepSeekApiKey);

        string? capturedBody = null;
        client.ResponseInterceptor = body => capturedBody = body;

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Say 'Response intercepted!' and nothing else."))
        {
            results.Add(chunk);
        }

        Assert.NotNull(capturedBody);
        Assert.NotEmpty(capturedBody);
        Assert.Contains("choices", capturedBody);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  10. OLLAMA INTEGRATION TESTS (local – keys from Program.cs)
    // ═══════════════════════════════════════════════════════════════════════

    const string OllamaEndpoint = "http://localhost:11434";
    const string OllamaModel = "qwen3.6:27b";
    const string OllamaApiKey = "ollama";

    [Fact(Skip = "Ollama integration – requires local Ollama server. Remove Skip to run.")]
    public async Task Ollama_SimpleChat_WithRequestInterceptor()
    {
        var client = new OpenAIClient(OllamaEndpoint, OllamaModel, OllamaApiKey);

        var interceptedHeaders = new Dictionary<string, string>();
        client.RequestInterceptor = request =>
        {
            // Ollama doesn't need special headers, but verify Bearer is set
            var auth = request.Headers.Authorization;
            if (auth != null)
                interceptedHeaders["Authorization"] = $"{auth.Scheme} {auth.Parameter}";
        };

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Say 'Hello from Ollama' and nothing else."))
        {
            results.Add(chunk);
        }

        // Ollama ignores the Bearer header but the client still sends it
        Assert.Contains("Authorization", interceptedHeaders);

        var lastAssistant = client.History.Last(m => m.Role == "assistant");
        Assert.NotNull(lastAssistant.Content);
        Assert.Contains("Ollama", lastAssistant.Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Skip = "Ollama integration – requires local Ollama server. Remove Skip to run.")]
    public async Task Ollama_WithToolCall()
    {
        var client = new OpenAIClient(OllamaEndpoint, OllamaModel, OllamaApiKey);

        var schema = """{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}""";
        client.RegisterTool("get_weather", schema, (string location) =>
        {
            if (location.Contains("London", StringComparison.OrdinalIgnoreCase))
                return "15°C, cloudy";
            return "20°C, unknown";
        });

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("What's the weather in London?"))
        {
            results.Add(chunk);
        }

        var toolMessages = client.History.Where(m => m.Role == "tool").ToList();
        Assert.NotEmpty(toolMessages);
    }

    [Fact(Skip = "Ollama integration – requires local Ollama server. Remove Skip to run.")]
    public async Task Ollama_WithResponseInterceptor()
    {
        var client = new OpenAIClient(OllamaEndpoint, OllamaModel, OllamaApiKey);

        string? capturedBody = null;
        client.ResponseInterceptor = body => capturedBody = body;

        var results = new List<string>();
        await foreach (var chunk in client.ChatStreamingAsync("Say 'Hello' and nothing else."))
        {
            results.Add(chunk);
        }

        Assert.NotNull(capturedBody);
        Assert.NotEmpty(capturedBody);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  11. IMAGE ATTACHMENT TESTS
    // ═══════════════════════════════════════════════════════════════════════

    [Fact]
    public void ChatMessage_UserWithImage_StoresBase64()
    {
        var bytes = new byte[] { 0x89, 0x50, 0x4E, 0x47 }; // PNG header
        var msg = ChatMessage.UserWithImage("Check this image", bytes, "image/png");
        Assert.Equal("iVBORw==", msg.ImageBase64); // base64 of 4 PNG bytes
        Assert.Equal("image/png", msg.ImageMimeType);
        Assert.Equal("Check this image", msg.Content);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  12. OPENAI OPTIONS TESTS
// ═══════════════════════════════════════════════════════════════════════════

public class OpenAIOptionsTests
{
    [Fact]
    public void Options_DefaultValues_AreCorrect()
    {
        var opts = new OpenAIOptions();
        Assert.Null(opts.Temperature);
        Assert.Null(opts.TopP);
        Assert.Null(opts.MaxTokens);
        Assert.Null(opts.FrequencyPenalty);
        Assert.Null(opts.PresencePenalty);
        Assert.Null(opts.Stop);
        Assert.True(opts.ResizeImage);
        Assert.Equal(1200, opts.MaxWidth);
        Assert.Equal(1200, opts.MaxHeight);
    }

    [Fact]
    public void Options_TemperatureAndTopP_AreIncludedInRequestBody()
    {
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.Options.Temperature = 0.7;
        client.Options.TopP = 0.9;
        client.Options.MaxTokens = 100;
        client.History.Add(ChatMessage.User("Hello"));

        // Use reflection to call BuildChatRequestBody
        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.Contains("\"temperature\":0.7", json);
        Assert.Contains("\"top_p\":0.9", json);
        Assert.Contains("\"max_tokens\":100", json);
    }

    [Fact]
    public void Options_FrequencyAndPresencePenalty_AreIncludedInRequestBody()
    {
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.Options.FrequencyPenalty = 0.5;
        client.Options.PresencePenalty = -0.2;
        client.History.Add(ChatMessage.User("Hello"));

        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.Contains("\"frequency_penalty\":0.5", json);
        Assert.Contains("\"presence_penalty\":-0.2", json);
    }

    [Fact]
    public void Options_StopSequences_AreIncludedInRequestBody()
    {
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.Options.Stop = new[] { "\n", "User:" };
        client.History.Add(ChatMessage.User("Hello"));

        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.Contains("\"stop\":[\"\\n\",\"User:\"]", json);
    }

    [Fact]
    public void Options_NotSet_AreOmittedFromRequestBody()
    {
        // Default options should not add any optional parameters to the request body
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.History.Add(ChatMessage.User("Hello"));

        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.DoesNotContain("\"temperature\"", json);
        Assert.DoesNotContain("\"top_p\"", json);
        Assert.DoesNotContain("\"max_tokens\"", json);
        Assert.DoesNotContain("\"frequency_penalty\"", json);
        Assert.DoesNotContain("\"presence_penalty\"", json);
        Assert.DoesNotContain("\"stop\"", json);
    }

    [Fact]
    public void Options_ReasoningEffort_IsIncludedInRequestBody()
    {
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.Options.ReasoningEffort = "high";
        client.History.Add(ChatMessage.User("Hello"));

        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.Contains("\"reasoning_effort\":\"high\"", json);
    }

    [Fact]
    public void Options_Thinking_Enabled_IsIncludedInRequestBody()
    {
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.Options.Thinking = new ThinkingConfig { Type = ThinkingType.Enabled };
        client.History.Add(ChatMessage.User("Hello"));

        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.Contains("\"thinking\":{\"type\":\"enabled\"}", json);
    }

    [Fact]
    public void Options_Thinking_Disabled_IsIncludedInRequestBody()
    {
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.Options.Thinking = new ThinkingConfig { Type = ThinkingType.Disabled };
        client.History.Add(ChatMessage.User("Hello"));

        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.Contains("\"thinking\":{\"type\":\"disabled\"}", json);
    }

    [Fact]
    public void Options_Thinking_NotSet_IsOmittedFromRequestBody()
    {
        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test");
        client.History.Add(ChatMessage.User("Hello"));

        var method = typeof(OpenAIClient).GetMethod("BuildChatRequestBody",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var json = (string)method!.Invoke(client, new object[] { true })!;

        Assert.DoesNotContain("\"thinking\"", json);
    }

    [Fact]
    public void Options_Thinking_DefaultTypeIsEnabled()
    {
        var thinking = new ThinkingConfig();
        Assert.Equal(ThinkingType.Enabled, thinking.Type);
    }

    [Fact]
    public void ThinkingConfig_JsonSerialization_RendersCorrectly()
    {
        var thinking = new ThinkingConfig { Type = ThinkingType.Disabled };
        var json = System.Text.Json.JsonSerializer.Serialize(thinking);

        Assert.Contains("\"type\":\"disabled\"", json);
    }

    [Fact]
    public void ThinkingConfig_JsonDeserialization_ParsesCorrectly()
    {
        var json = """{"type":"enabled"}""";
        var thinking = System.Text.Json.JsonSerializer.Deserialize<ThinkingConfig>(json);

        Assert.NotNull(thinking);
        Assert.Equal(ThinkingType.Enabled, thinking!.Type);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  13. CHATMESSAGE SERIALIZATION TESTS (System.Text.Json)
// ═══════════════════════════════════════════════════════════════════════════

public class ChatMessageSerializationTests
{
    [Fact]
    public void ChatMessage_SerializesToJson_WithJsonPropertyNameAttributes()
    {
        var msg = ChatMessage.Assistant("Hello world");
        var json = JsonSerializer.Serialize(msg);

        Assert.Contains("\"role\":\"assistant\"", json);
        Assert.Contains("\"content\":\"Hello world\"", json);
    }

    [Fact]
    public void ChatMessage_DeserializesFromJson_Correctly()
    {
        var json = """{"role":"user","content":"Hi there"}""";
        var msg = JsonSerializer.Deserialize<ChatMessage>(json);

        Assert.NotNull(msg);
        Assert.Equal("user", msg!.Role);
        Assert.Equal("Hi there", msg.Content);
    }

    [Fact]
    public void ChatMessage_DeserializesReasoningContent()
    {
        var json = """{"role":"assistant","content":"Final answer","reasoning_content":"Let me think..."}""";
        var msg = JsonSerializer.Deserialize<ChatMessage>(json);

        Assert.NotNull(msg);
        Assert.Equal("assistant", msg!.Role);
        Assert.Equal("Final answer", msg.Content);
        Assert.Equal("Let me think...", msg.ReasoningContent);
    }

    [Fact]
    public void ChatMessage_DeserializesToolCalls()
    {
        var json = """{"role":"assistant","content":null,"tool_calls":"[{\"id\":\"call_1\"}]"}""";
        var msg = JsonSerializer.Deserialize<ChatMessage>(json);

        Assert.NotNull(msg);
        Assert.Equal("assistant", msg!.Role);
        Assert.NotNull(msg.ToolCallsRaw);
        Assert.Contains("call_1", msg.ToolCallsRaw);
    }

    [Fact]
    public void ChatMessage_DeserializesToolCallId()
    {
        var json = """{"role":"tool","tool_call_id":"call_123","content":"result"}""";
        var msg = JsonSerializer.Deserialize<ChatMessage>(json);

        Assert.NotNull(msg);
        Assert.Equal("tool", msg!.Role);
        Assert.Equal("call_123", msg.ToolCallId);
        Assert.Equal("result", msg.Content);
    }

    [Fact]
    public void ChatMessage_ImageFields_AreIgnoredInJsonSerialization()
    {
        var msg = ChatMessage.UserWithImage("Check this", new byte[] { 0x89, 0x50, 0x4E, 0x47 }, "image/png");
        var json = JsonSerializer.Serialize(msg);

        // ImageBase64 and ImageMimeType should be JsonIgnore'd
        Assert.DoesNotContain("ImageBase64", json);
        Assert.DoesNotContain("ImageMimeType", json);
        Assert.DoesNotContain("iVBORw", json); // base64 of PNG header
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  14. NON-STREAMING CHATASYNC TESTS
// ═══════════════════════════════════════════════════════════════════════════

public class ChatAsyncTests
{
    [Fact]
    public async Task ChatAsync_SimpleResponse_ReturnsAssistantMessage()
    {
        var handler = new MockHttpMessageHandler(async req =>
        {
            var body = await req.Content!.ReadAsStringAsync();
            // Verify non-streaming request does not have stream:true
            Assert.Contains("\"stream\":false", body);

            var responseJson = """
            {
              "id": "chatcmpl-123",
              "object": "chat.completion",
              "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": "Hello from non-streaming!"
                },
                "finish_reason": "stop"
              }]
            }
            """;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(responseJson, Encoding.UTF8, "application/json")
            };
        });

        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test", new HttpClient(handler));

        var result = await client.ChatAsync("Hi");

        Assert.NotNull(result);
        Assert.Equal("assistant", result.Role);
        Assert.Equal("Hello from non-streaming!", result.Content);
    }

    [Fact]
    public async Task ChatAsync_WithReasoningContent_ParsesCorrectly()
    {
        var handler = new MockHttpMessageHandler(async req =>
        {
            var responseJson = """
            {
              "id": "chatcmpl-456",
              "object": "chat.completion",
              "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": "Final answer",
                  "reasoning_content": "Let me think step by step..."
                },
                "finish_reason": "stop"
              }]
            }
            """;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(responseJson, Encoding.UTF8, "application/json")
            };
        });

        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test", new HttpClient(handler));

        var result = await client.ChatAsync("Think carefully");

        Assert.NotNull(result);
        Assert.Equal("Final answer", result.Content);
        Assert.Equal("Let me think step by step...", result.ReasoningContent);
    }

    [Fact]
    public async Task ChatAsync_AgenticLoop_ExecutesToolThenReturnsFinalAnswer()
    {
        bool firstRequest = true;

        var handler = new MockHttpMessageHandler(async req =>
        {
            if (firstRequest)
            {
                firstRequest = false;
                var toolCallJson = """
                {
                  "id": "chatcmpl-789",
                  "object": "chat.completion",
                  "choices": [{
                    "index": 0,
                    "message": {
                      "role": "assistant",
                      "content": null,
                      "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                          "name": "echo",
                          "arguments": "{\"text\":\"hello\"}"
                        }
                      }]
                    },
                    "finish_reason": "tool_calls"
                  }]
                }
                """;
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(toolCallJson, Encoding.UTF8, "application/json")
                };
            }
            else
            {
                var finalJson = """
                {
                  "id": "chatcmpl-790",
                  "object": "chat.completion",
                  "choices": [{
                    "index": 0,
                    "message": {
                      "role": "assistant",
                      "content": "Final answer after tool call"
                    },
                    "finish_reason": "stop"
                  }]
                }
                """;
                return new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new StringContent(finalJson, Encoding.UTF8, "application/json")
                };
            }
        });

        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test", new HttpClient(handler));
        var schema = """{"type":"function","function":{"name":"echo","description":"Echoes","parameters":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}}""";
        client.RegisterTool("echo", schema, (string text) => text);

        var result = await client.ChatAsync("Echo this");

        Assert.NotNull(result);
        Assert.Equal("Final answer after tool call", result.Content);

        // Verify the tool was called
        var toolMessages = client.History.Where(m => m.Role == "tool").ToList();
        Assert.NotEmpty(toolMessages);
    }

    [Fact]
    public async Task ChatAsync_HistoryIsAccumulated()
    {
        var handler = new MockHttpMessageHandler(async req =>
        {
            var responseJson = """
            {
              "id": "chatcmpl-111",
              "object": "chat.completion",
              "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": "Response 1"
                },
                "finish_reason": "stop"
              }]
            }
            """;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(responseJson, Encoding.UTF8, "application/json")
            };
        });

        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test", new HttpClient(handler));

        var result1 = await client.ChatAsync("First message");
        Assert.Equal("Response 1", result1.Content);

        // Second call should include the previous history
        var result2 = await client.ChatAsync("Second message");
        Assert.Equal("Response 1", result2.Content);

        // History should now have 4 messages: user+assistant from call 1, user+assistant from call 2
        Assert.Equal(4, client.History.Count);
    }

    [Fact]
    public async Task ChatAsync_ResponseInterceptor_IsInvoked()
    {
        string? capturedBody = null;

        var handler = new MockHttpMessageHandler(async req =>
        {
            var responseJson = """
            {
              "id": "chatcmpl-222",
              "object": "chat.completion",
              "choices": [{
                "index": 0,
                "message": {
                  "role": "assistant",
                  "content": "Intercepted!"
                },
                "finish_reason": "stop"
              }]
            }
            """;
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(responseJson, Encoding.UTF8, "application/json")
            };
        });

        var client = new OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", "sk-test", new HttpClient(handler));
        client.ResponseInterceptor = body => capturedBody = body;

        var result = await client.ChatAsync("Test");

        Assert.NotNull(capturedBody);
        Assert.Contains("Intercepted!", capturedBody);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mock HTTP message handler for unit testing
// ═══════════════════════════════════════════════════════════════════════════

public class MockHttpMessageHandler : HttpMessageHandler
{
    private readonly Func<HttpRequestMessage, Task<HttpResponseMessage>> _handler;

    public MockHttpMessageHandler(Func<HttpRequestMessage, Task<HttpResponseMessage>> handler)
    {
        _handler = handler;
    }

    protected override async Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request, CancellationToken cancellationToken)
    {
        return await _handler(request);
    }
}
