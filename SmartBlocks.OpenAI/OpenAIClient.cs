using System.Net.Http.Headers;
using System.Reflection;
using System.Text;
using System.Text.Json;
using SkiaSharp;

namespace SmartBlocks.OpenAI;

/// <summary>
/// Minimal OpenAI chat-completions client.
///
/// Features:
///   - Streaming responses (SSE)
///   - Tool/function calling (JSON schema definitions + reflection-invoked delegates)
///   - Automatic agentic loop (re-calls the API when the model requests tool calls)
///   - Image attachments (via base64 data URIs) with optional SkiaSharp downscaling
///   - Built-in chat history
///   - Request/Response interceptors (provider-specific customisation)
///   - Reasoning content support (DeepSeek R1, QwQ, etc.)
///   - Configurable options (temperature, max_tokens, top_p, etc.)
/// </summary>
public sealed class OpenAIClient : IDisposable
{
    // ── Configuration ───────────────────────────────────────────────────────

    private readonly HttpClient _http;
    private readonly string _chatCompletionsUrl;
    private readonly string _model;
    private readonly string _apiKey;
    private readonly string _endpoint;

    // ── Options ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Configurable options for the chat completion request.
    /// Includes temperature, max_tokens, top_p, frequency_penalty,
    /// presence_penalty, stop sequences, and image resizing settings.
    /// </summary>
    public OpenAIOptions Options { get; set; } = new();

    // ── Interceptors & callbacks ───────────────────────────────────────────

    /// <summary>
    /// Optional interceptor invoked just before the HTTP request is sent.
    /// Receives the <see cref="HttpRequestMessage"/> for customisation
    /// (e.g. adding provider-specific headers, modifying the body).
    /// </summary>
    public Action<HttpRequestMessage>? RequestInterceptor { get; set; }

    /// <summary>
    /// Optional interceptor invoked after a successful HTTP response is received
    /// but before SSE parsing begins. Receives the raw JSON response body string
    /// for inspection / logging.
    /// </summary>
    public Action<string>? ResponseInterceptor { get; set; }

    /// <summary>
    /// Optional callback invoked for each chunk of reasoning content as it
    /// is received from the stream. Reasoning is output in separate "chunks"
    /// from the main content and typically represents the model's internal
    /// chain-of-thought.
    /// </summary>
    public Action<string>? OnReasoning { get; set; }

    /// <summary>
    /// When <see langword="true"/>, the current chunk being yielded from
    /// <see cref="ChatStreamingAsync"/> is reasoning content (chain-of-thought).
    /// When <see langword="false"/>, it is regular content text.
    /// Check this property inside the <c>await foreach</c> loop to decide
    /// how to render each token (e.g. reasoning in DarkGray, content in default).
    /// </summary>
    public bool IsReasoning { get; private set; }

    /// <summary>
    /// Registered tools accessible to the agentic loop.
    /// Key   = tool name (must match the function name the model requests).
    /// Value = (JsonElement args) => Task<string> result (async handler).
    /// </summary>
    public Dictionary<string, Func<JsonElement, Task<string>>> Tools { get; } = new();

    /// <summary>
    /// JSON schema definitions for each tool, serialised as a JsonElement array
    /// matching the OpenAI "tools" parameter.
    /// </summary>
    public List<JsonElement> ToolDefinitions { get; } = new();

    /// <summary>Messages accumulated during the current session.</summary>
    public List<ChatMessage> History { get; } = new();

    // ── Construction ────────────────────────────────────────────────────────

    public OpenAIClient(string endpoint, string model, string apiKey)
        : this(endpoint, model, apiKey, new HttpClient()) { }

    public OpenAIClient(string endpoint, string model, string apiKey, HttpClient http)
    {
        // Normalise the endpoint URL so we safely append /chat/completions.
        // Supported forms:
        //   "https://api.openai.com/v1"          -> "https://api.openai.com/v1/chat/completions"
        //   "https://openrouter.ai/api/v1/"       -> "https://openrouter.ai/api/v1/chat/completions"
        //   "https://api.deepseek.com"            -> "https://api.deepseek.com/v1/chat/completions"
        //   "http://localhost:11434" (Ollama)     -> "http://localhost:11434/v1/chat/completions"
        var urlBase = endpoint.TrimEnd('/');
        if (!urlBase.EndsWith("/v1", StringComparison.OrdinalIgnoreCase) &&
            !urlBase.Contains("/v1", StringComparison.OrdinalIgnoreCase))
            urlBase += "/v1";
        _chatCompletionsUrl = urlBase + "/chat/completions";
        _model = model;
        _apiKey = apiKey;
        _http = http;
        _endpoint = endpoint;

        // Increase default timeout for reasoning models that may take
        // a long time before outputting any tokens.
        if (_http.Timeout == TimeSpan.FromSeconds(100))
            _http.Timeout = TimeSpan.FromSeconds(300);
    }

    // ── Tool registration (convenience) ─────────────────────────────────────

    /// <summary>
    /// Register a tool using a Delegate plus a JSON schema string.
    /// The schema is the full tool definition object as described in
    /// the OpenAI API docs.
    /// </summary>
    public void RegisterTool(string name, string jsonSchema, Delegate handler)
    {
        using var doc = JsonDocument.Parse(jsonSchema);
        ToolDefinitions.Add(doc.RootElement.Clone());

        Tools[name] = async args =>
        {
            var method = handler.GetMethodInfo();
            var parameters = method.GetParameters();
            var invokeArgs = new object?[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                var p = parameters[i];
                if (args.TryGetProperty(p.Name!, out var prop))
                {
                    invokeArgs[i] = DeserializeArgument(prop, p.ParameterType);
                }
                else
                {
                    invokeArgs[i] = p.HasDefaultValue ? p.DefaultValue : null;
                }
            }

            var result = handler.DynamicInvoke(invokeArgs);
            if (result is Task taskResult)
            {
                await taskResult.ConfigureAwait(false);
                var resultProperty = taskResult.GetType().GetProperty("Result");
                return resultProperty?.GetValue(taskResult)?.ToString() ?? "";
            }
            return result?.ToString() ?? "";
        };
    }

    private static object? DeserializeArgument(JsonElement element, Type targetType)
    {
        if (targetType == typeof(string)) return element.GetString();
        if (targetType == typeof(int)) return element.GetInt32();
        if (targetType == typeof(long)) return element.GetInt64();
        if (targetType == typeof(double)) return element.GetDouble();
        if (targetType == typeof(float)) return element.GetSingle();
        if (targetType == typeof(bool)) return element.GetBoolean();
        if (targetType == typeof(decimal)) return element.GetDecimal();
        // fallback – use System.Text.Json
        return JsonSerializer.Deserialize(element.GetRawText(), targetType);
    }

    // ── Core chat methods ─────────────────────────────────────────────────

    /// <summary>
    /// Send a streaming chat request. Yields SSE data lines as strings so the
    /// caller can process them however they wish (e.g. write to console).
    /// </summary>
    public async IAsyncEnumerable<string> ChatStreamingAsync(
        string userMessage,
        byte[]? imageData = null,
        string? imageMimeType = null)
    {
        // 1. Resize image if needed, then append the new user message to history
        if (imageData != null && imageMimeType != null)
        {
            var resizedData = Options.ResizeImage
                ? ResizeImageIfNeeded(imageData)
                : imageData;
            History.Add(ChatMessage.UserWithImage(userMessage, resizedData, imageMimeType));
        }
        else
        {
            History.Add(ChatMessage.User(userMessage));
        }

        // 2. Agentic loop – keep going until the model stops requesting tools
        bool requiresAction;
        do
        {
            requiresAction = false;

            // Build request body (streaming = true)
            var bodyJson = BuildChatRequestBody(stream: true);
            using var httpContent = new StringContent(bodyJson, Encoding.UTF8, "application/json");

            using var request = new HttpRequestMessage(HttpMethod.Post, _chatCompletionsUrl);
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            request.Content = httpContent;

            // ── Provider-specific default headers ────────────────────────
            ApplyProviderDefaultHeaders(request);

            // ── Request interceptor ──────────────────────────────────────
            RequestInterceptor?.Invoke(request);

            using var response = await _http.SendAsync(
                request,
                HttpCompletionOption.ResponseHeadersRead);

            response.EnsureSuccessStatusCode();

            // 3. Read SSE stream
            var assistantMsg = new ChatMessage { Role = "assistant" };
            var contentBuilder = new StringBuilder();
            var reasoningBuilder = new StringBuilder();
            var toolCallsBuilder = new List<PendingToolCall>();

            using var stream = await response.Content.ReadAsStreamAsync();
            using var reader = new StreamReader(stream);

            // Collect the raw SSE body for the response interceptor
            var rawSseBuilder = ResponseInterceptor != null ? new StringBuilder() : null;

            while (true)
            {
                var line = await reader.ReadLineAsync();
                if (line == null) break;

                if (line.StartsWith("data: "))
                {
                    var data = line.Substring(6);

                    // Capture raw SSE for response interceptor
                    rawSseBuilder?.AppendLine(data);

                    if (data == "[DONE]") break;

                    // Yield content delta chunks for the caller
                    Console.ResetColor();
                    IsReasoning = false;
                    foreach (var delta in YieldContentChunks(data))
                    {
                        Console.Write(delta);
                        yield return delta;
                    }

                    // Yield reasoning chunks separately
                    IsReasoning = true;
                    foreach (var reasoningDelta in YieldReasoningChunks(data))
                    {
                        OnReasoning?.Invoke(reasoningDelta);
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.Write(reasoningDelta);
                        yield return reasoningDelta;
                    }

                    ProcessStreamChunk(data, contentBuilder, reasoningBuilder, toolCallsBuilder);
                }
            }

            // ── Response interceptor ─────────────────────────────────────
            if (rawSseBuilder != null)
                ResponseInterceptor?.Invoke(rawSseBuilder.ToString());

            // 4. Store the assistant message
            var content = contentBuilder.Length > 0 ? contentBuilder.ToString() : null;
            assistantMsg.Content = content;

            var reasoning = reasoningBuilder.Length > 0 ? reasoningBuilder.ToString() : null;
            assistantMsg.ReasoningContent = reasoning;

            // 5. Process tool calls if any
            if (toolCallsBuilder.Count > 0)
            {
                requiresAction = true;

                var tcArray = BuildToolCallsJson(toolCallsBuilder);
                assistantMsg.ToolCallsRaw = tcArray;

                History.Add(assistantMsg);

                foreach (var tc in toolCallsBuilder)
                {
                    var result = await ExecuteToolAsync(tc);
                    var resultMsg = ChatMessage.Tool(tc.Id, tc.Name, result);
                    History.Add(resultMsg);
                    yield return $"[TOOL {tc.Name} => {result}]";
                }
            }
            else
            {
                History.Add(assistantMsg);
            }

        } while (requiresAction);
    }

    /// <summary>
    /// Send a non-streaming chat request and return the assistant's response
    /// as a <see cref="ChatMessage"/>. The agentic loop is fully handled:
    /// if the model requests tool calls, they are executed automatically and
    /// the API is re-invoked until a final text response is received.
    /// </summary>
    public async Task<ChatMessage> ChatAsync(
        string userMessage,
        byte[]? imageData = null,
        string? imageMimeType = null)
    {
        // 1. Resize image if needed, then append the new user message to history
        if (imageData != null && imageMimeType != null)
        {
            var resizedData = Options.ResizeImage
                ? ResizeImageIfNeeded(imageData)
                : imageData;
            History.Add(ChatMessage.UserWithImage(userMessage, resizedData, imageMimeType));
        }
        else
        {
            History.Add(ChatMessage.User(userMessage));
        }

        // 2. Agentic loop – keep going until the model stops requesting tools
        bool requiresAction;
        ChatMessage? lastAssistant = null;

        do
        {
            requiresAction = false;

            // Build request body (streaming = false)
            var bodyJson = BuildChatRequestBody(stream: false);
            using var httpContent = new StringContent(bodyJson, Encoding.UTF8, "application/json");

            using var request = new HttpRequestMessage(HttpMethod.Post, _chatCompletionsUrl);
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
            request.Content = httpContent;

            // ── Provider-specific default headers ────────────────────────
            ApplyProviderDefaultHeaders(request);

            // ── Request interceptor ──────────────────────────────────────
            RequestInterceptor?.Invoke(request);

            using var response = await _http.SendAsync(request);
            response.EnsureSuccessStatusCode();

            var rawJson = await response.Content.ReadAsStringAsync();

            // ── Response interceptor ─────────────────────────────────────
            ResponseInterceptor?.Invoke(rawJson);

            // 3. Parse the non-streaming response
            using var doc = JsonDocument.Parse(rawJson);
            var root = doc.RootElement;

            var assistantMsg = new ChatMessage { Role = "assistant" };

            if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0)
            {
                var choice = choices[0];
                if (choice.TryGetProperty("message", out var message))
                {
                    // Content
                    if (message.TryGetProperty("content", out var contentEl) && contentEl.ValueKind == JsonValueKind.String)
                        assistantMsg.Content = contentEl.GetString();

                    // Reasoning content (e.g. DeepSeek R1)
                    if (message.TryGetProperty("reasoning_content", out var reasoningEl) && reasoningEl.ValueKind == JsonValueKind.String)
                        assistantMsg.ReasoningContent = reasoningEl.GetString();

                    // Tool calls
                    if (message.TryGetProperty("tool_calls", out var tcArray) && tcArray.GetArrayLength() > 0)
                    {
                        var toolCallsBuilder = new List<PendingToolCall>();

                        foreach (var tcElem in tcArray.EnumerateArray())
                        {
                            var pending = new PendingToolCall();

                            if (tcElem.TryGetProperty("id", out var idEl))
                                pending.Id = idEl.GetString() ?? "";

                            if (tcElem.TryGetProperty("type", out var typeEl))
                                pending.Type = typeEl.GetString() ?? "function";

                            if (tcElem.TryGetProperty("function", out var funcEl))
                            {
                                if (funcEl.TryGetProperty("name", out var nameEl))
                                    pending.Name = nameEl.GetString() ?? "";

                                if (funcEl.TryGetProperty("arguments", out var argsEl))
                                    pending.Arguments = argsEl.GetString() ?? "";
                            }

                            toolCallsBuilder.Add(pending);
                        }

                        if (toolCallsBuilder.Count > 0)
                        {
                            requiresAction = true;
                            assistantMsg.ToolCallsRaw = BuildToolCallsJson(toolCallsBuilder);

                            History.Add(assistantMsg);

                            foreach (var tc in toolCallsBuilder)
                            {
                                var result = await ExecuteToolAsync(tc);
                                var resultMsg = ChatMessage.Tool(tc.Id, tc.Name, result);
                                History.Add(resultMsg);
                            }
                        }
                    }
                }
            }

            if (!requiresAction)
            {
                History.Add(assistantMsg);
                lastAssistant = assistantMsg;
            }

        } while (requiresAction);

        return lastAssistant ?? new ChatMessage { Role = "assistant" };
    }

    // ── Image resizing ──────────────────────────────────────────────────────

    /// <summary>
    /// Downsizes an image to fit within the configured <see cref="OpenAIOptions.MaxWidth"/>
    /// and <see cref="OpenAIOptions.MaxHeight"/> bounds while preserving aspect ratio.
    /// Uses SkiaSharp for high-quality downscaling.
    /// </summary>
    private byte[] ResizeImageIfNeeded(byte[] imageData)
    {
        var maxWidth = Options.MaxWidth;
        var maxHeight = Options.MaxHeight;

        using var inputStream = new MemoryStream(imageData);
        using var original = SKBitmap.Decode(inputStream);
        if (original == null)
            return imageData; // Could not decode, return as-is

        // Check if resizing is actually needed
        if (original.Width <= maxWidth && original.Height <= maxHeight)
            return imageData;

        // Calculate new dimensions preserving aspect ratio
        var ratioX = (double)maxWidth / original.Width;
        var ratioY = (double)maxHeight / original.Height;
        var ratio = Math.Min(ratioX, ratioY);

        var newWidth = (int)(original.Width * ratio);
        var newHeight = (int)(original.Height * ratio);

        // Ensure at least 1 pixel
        if (newWidth < 1) newWidth = 1;
        if (newHeight < 1) newHeight = 1;

        using var resized = original.Resize(new SKImageInfo(newWidth, newHeight), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
        if (resized == null)
            return imageData; // Resize failed, return original

        using var image = SKImage.FromBitmap(resized);
        using var outputStream = new MemoryStream();
        // Encode as PNG to preserve transparency and quality
        image.Encode(SKEncodedImageFormat.Png, 90)
             .SaveTo(outputStream);

        return outputStream.ToArray();
    }

    // ── Request body builder ────────────────────────────────────────────────

    private string BuildChatRequestBody(bool stream = true)
    {
        var sb = new StringBuilder();
        sb.Append("{\"model\":\"").Append(EscapeJson(_model)).Append('"');

        // ── Optional parameters ──────────────────────────────────────────
        if (Options.Temperature.HasValue)
            sb.Append(",\"temperature\":").Append(Options.Temperature.Value.ToString("0.0##", System.Globalization.CultureInfo.InvariantCulture));

        if (Options.TopP.HasValue)
            sb.Append(",\"top_p\":").Append(Options.TopP.Value.ToString("0.0##", System.Globalization.CultureInfo.InvariantCulture));

        if (Options.MaxTokens.HasValue)
            sb.Append(",\"max_tokens\":").Append(Options.MaxTokens.Value);

        if (Options.FrequencyPenalty.HasValue)
            sb.Append(",\"frequency_penalty\":").Append(Options.FrequencyPenalty.Value.ToString("0.0##", System.Globalization.CultureInfo.InvariantCulture));

        if (Options.PresencePenalty.HasValue)
            sb.Append(",\"presence_penalty\":").Append(Options.PresencePenalty.Value.ToString("0.0##", System.Globalization.CultureInfo.InvariantCulture));

        if (Options.Stop is { Length: > 0 })
        {
            sb.Append(",\"stop\":[");
            for (int i = 0; i < Options.Stop.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(EscapeJson(Options.Stop[i])).Append('"');
            }
            sb.Append(']');
        }

        if (Options.ReasoningEffort != null)
            sb.Append(",\"reasoning_effort\":\"").Append(EscapeJson(Options.ReasoningEffort)).Append('"');

        // Thinking config (o-series models: o1, o3, etc.)
        if (Options.Thinking != null)
        {
            sb.Append(",\"thinking\":{\"type\":\"")
              .Append(EscapeJson(Options.Thinking.Type))
              .Append("\"}");
        }

        // Messages
        sb.Append(",\"messages\":[");
        for (int i = 0; i < History.Count; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append(History[i].ToRequestJsonString());
        }
        sb.Append(']');

        // Tools (if any registered)
        if (ToolDefinitions.Count > 0)
        {
            sb.Append(",\"tools\":[");
            for (int i = 0; i < ToolDefinitions.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(ToolDefinitions[i].GetRawText());
            }
            sb.Append(']');
        }

        // Streaming
        sb.Append(",\"stream\":").Append(stream ? "true" : "false");

        sb.Append('}');
        return sb.ToString();
    }

    // ── Stream chunk processing ─────────────────────────────────────────────

    private static void ProcessStreamChunk(
        string data,
        StringBuilder contentBuilder,
        StringBuilder reasoningBuilder,
        List<PendingToolCall> toolCallsBuilder)
    {
        using var doc = JsonDocument.Parse(data);
        var root = doc.RootElement;

        // Content delta
        if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0)
        {
            var choice = choices[0];

            if (choice.TryGetProperty("delta", out var delta))
            {
                // ── Reasoning content ────────────────────────────────────
                // Some providers (DeepSeek R1, QwQ) send reasoning via a
                // "reasoning" field in the delta, or as a top-level field.
                // DeepSeek also uses "reasoning_content" in the delta
                // (matching the non-streaming response field name).
                string? reasoningText = null;
                if (delta.TryGetProperty("reasoning_content", out var reasoningContent))
                    reasoningText = reasoningContent.GetString();
                else if (delta.TryGetProperty("reasoning", out var reasoning))
                    reasoningText = reasoning.GetString();

                if (reasoningText != null)
                    reasoningBuilder.Append(reasoningText);

                // Standard content
                if (delta.TryGetProperty("content", out var content))
                {
                    var text = content.GetString();
                    if (text != null)
                        contentBuilder.Append(text);
                }

                // Tool calls delta
                if (delta.TryGetProperty("tool_calls", out var tcArray))
                {
                    foreach (var tcElem in tcArray.EnumerateArray())
                    {
                        var index = tcElem.TryGetProperty("index", out var idxElem)
                            ? idxElem.GetInt32()
                            : 0;

                        // Ensure we have a slot
                        while (toolCallsBuilder.Count <= index)
                            toolCallsBuilder.Add(new PendingToolCall());

                        var pending = toolCallsBuilder[index];

                        if (tcElem.TryGetProperty("id", out var idElem))
                            pending.Id += idElem.GetString();

                        if (tcElem.TryGetProperty("type", out var typeElem))
                            pending.Type += typeElem.GetString();

                        if (tcElem.TryGetProperty("function", out var funcElem))
                        {
                            if (funcElem.TryGetProperty("name", out var nameElem))
                                pending.Name += nameElem.GetString();

                            if (funcElem.TryGetProperty("arguments", out var argsElem))
                                pending.Arguments += argsElem.GetString();
                        }
                    }
                }
            }

            // Also check top-level "reasoning" field (some providers use
            // non-standard placement outside of delta)
            if (choice.TryGetProperty("reasoning", out var topLevelReasoning) &&
                topLevelReasoning.ValueKind == JsonValueKind.String)
            {
                reasoningBuilder.Append(topLevelReasoning.GetString());
            }
        }

        // Some providers (e.g. OpenRouter with certain models) send reasoning
        // as a top-level field alongside choices
        if (root.TryGetProperty("reasoning", out var rootReasoning) &&
            rootReasoning.ValueKind == JsonValueKind.String)
        {
            reasoningBuilder.Append(rootReasoning.GetString());
        }
    }

    // ── Tool execution ──────────────────────────────────────────────────────

    private async Task<string> ExecuteToolAsync(PendingToolCall tc)
    {
        if (!Tools.TryGetValue(tc.Name, out var handler))
            return $"Error: unknown tool '{tc.Name}'";

        try
        {
            using var argsDoc = JsonDocument.Parse(tc.Arguments);
            return await handler(argsDoc.RootElement);
        }
        catch (Exception ex)
        {
            return $"Error executing tool '{tc.Name}': {ex.Message}";
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static string BuildToolCallsJson(List<PendingToolCall> calls)
    {
        var sb = new StringBuilder();
        sb.Append('[');
        for (int i = 0; i < calls.Count; i++)
        {
            if (i > 0) sb.Append(',');
            var c = calls[i];
            sb.Append("{\"id\":\"")
              .Append(EscapeJson(c.Id))
              .Append("\",\"type\":\"function\",\"function\":{\"name\":\"")
              .Append(EscapeJson(c.Name))
              .Append("\",\"arguments\":\"")
              .Append(EscapeJson(c.Arguments))
              .Append("\"}}");
        }
        sb.Append(']');
        return sb.ToString();
    }

    private static string EscapeJson(string s) =>
        s.Replace("\\", "\\\\")
         .Replace("\"", "\\\"")
         .Replace("\n", "\\n")
         .Replace("\r", "\\r")
         .Replace("\t", "\\t");

    /// <summary>
    /// Extracts the content text delta from an SSE data chunk and yields it,
    /// so the caller receives per-token streaming output in real time.
    /// Returns nothing for chunks that lack a content delta (e.g. tool-call
    /// deltas, finish signals).
    /// </summary>
    private static IEnumerable<string> YieldContentChunks(string data)
    {
        using var doc = JsonDocument.Parse(data);
        var root = doc.RootElement;

        if (!root.TryGetProperty("choices", out var choices) || choices.GetArrayLength() == 0)
            yield break;

        var choice = choices[0];
        if (!choice.TryGetProperty("delta", out var delta))
            yield break;

        if (delta.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String)
        {
            var text = content.GetString();
            if (!string.IsNullOrEmpty(text))
                yield return text;
        }
    }

    /// <summary>
    /// Extracts reasoning content chunks from an SSE data line.
    /// Reasoning fields can appear in:
    ///   - delta.reasoning  (DeepSeek R1 style)
    ///   - choice.reasoning (non-standard placement)
    ///   - root.reasoning   (OpenRouter wrapper style)
    /// </summary>
    private static IEnumerable<string> YieldReasoningChunks(string data)
    {
        using var doc = JsonDocument.Parse(data);
        var root = doc.RootElement;

        // Check delta.reasoning or delta.reasoning_content
        if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0)
        {
            var choice = choices[0];
            if (choice.TryGetProperty("delta", out var delta))
            {
                // DeepSeek uses "reasoning_content"; some providers use "reasoning"
                if (delta.TryGetProperty("reasoning_content", out var deltaReasoning) ||
                    delta.TryGetProperty("reasoning", out deltaReasoning))
                {
                    if (deltaReasoning.ValueKind == JsonValueKind.String)
                    {
                        var text = deltaReasoning.GetString();
                        if (!string.IsNullOrEmpty(text))
                            yield return text;
                    }
                }
            }

            // Check choice-level reasoning
            if (choice.TryGetProperty("reasoning", out var choiceReasoning) &&
                choiceReasoning.ValueKind == JsonValueKind.String)
            {
                var text = choiceReasoning.GetString();
                if (!string.IsNullOrEmpty(text))
                    yield return text;
            }
        }

        // Check root-level reasoning
        if (root.TryGetProperty("reasoning", out var rootReasoning) &&
            rootReasoning.ValueKind == JsonValueKind.String)
        {
            var text = rootReasoning.GetString();
            if (!string.IsNullOrEmpty(text))
                yield return text;
        }
    }

    public void Dispose() => _http.Dispose();

    // ── Provider-specific defaults ─────────────────────────────────────────

    /// <summary>
    /// Applies known provider-specific HTTP headers automatically based on the
    /// endpoint URL. These defaults are applied <em>before</em> the request
    /// interceptor runs, so the interceptor can still override them if needed.
    /// </summary>
    private void ApplyProviderDefaultHeaders(HttpRequestMessage request)
    {
        // OpenRouter requires an HTTP-Referer and X-Title header for
        // API access identification / rate limiting purposes.
        if (_endpoint.Contains("openrouter", StringComparison.OrdinalIgnoreCase))
        {
            request.Headers.TryAddWithoutValidation("HTTP-Referer",
                "https://github.com/smartblocks/openai-agentframework");
            request.Headers.TryAddWithoutValidation("X-Title",
                "SmartBlocks OpenAI Agent Framework");
        }
    }

    // ── Internal state ──────────────────────────────────────────────────────

    private sealed class PendingToolCall
    {
        public string Id { get; set; } = "";
        public string Type { get; set; } = "function";
        public string Name { get; set; } = "";
        public string Arguments { get; set; } = "";
    }
}
