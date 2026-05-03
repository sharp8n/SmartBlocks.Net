using System.Net.Http.Headers;
using System.Net.ServerSentEvents;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace SmartBlocks.OpenAiConnect;

public partial class SmartOpenAiClient(HttpClient? httpClient = null)
{
    readonly HttpClient httpClient = httpClient ?? new();
    public SmartOpenAISettings Settings { get; set; } = new();
    public List<JsonElement> Tools { get; set; } = [];

    public async IAsyncEnumerable<OpenAiResponseChunk> ChatCompletionStreamAsync(
        OpenAiRequest request,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        httpClient.BaseAddress ??= new Uri(Settings.Url ?? "https://api.openai.com");
        using var requestMessage = PrepareRequest(request, true);
        using var response = await httpClient.SendAsync(requestMessage, ct);
        await HandleError(response, ct);

        using var stream = await response.Content.ReadAsStreamAsync(ct);

        var parser = SseParser.Create(stream);

        var outputBuilder = new StringBuilder();
        var outputReasoningBuilder = new StringBuilder();
        var toolCallResults = new List<OpenAiMessage>();
        var toolCallsMap = new Dictionary<int, ToolCall>();

        await foreach (var item in parser.EnumerateAsync(ct))
        {
            if (item.Data == "[DONE]")
                break;
            var chunk = JsonSerializer.Deserialize(item.Data, OpenAiJsonContext.Default.OpenAiResponseChunk)!;
            yield return chunk;

            foreach (var choice in chunk.choices)
            {
                var delta = choice.delta!;
                if (!string.IsNullOrEmpty(delta.reasoning_content))
                    request.onProgress?.Invoke(delta.reasoning_content, true);
                if (!string.IsNullOrEmpty(delta.content))
                    request.onProgress?.Invoke(delta.content, false);

                if (delta.content != null) outputBuilder.Append(delta.content);
                if (delta.reasoning_content != null) outputReasoningBuilder.Append(delta.reasoning_content);

                var toolCalls = delta.tool_calls;

                if (toolCalls != null)
                {
                    foreach (var toolCall in toolCalls)
                    {
                        if (!toolCallsMap.TryGetValue(toolCall.index, out var toolCallInternal))
                        {
                            toolCallInternal = new ToolCall(
                                toolCall.index,
                                toolCall.id,
                                toolCall.type,
                                new()
                                {
                                    name = toolCall.function.name,
                                    arguments = toolCall.function.arguments,
                                    deltaArguments = new StringBuilder()
                                }
                            );
                            toolCallsMap[toolCall.index] = toolCallInternal;
                        }
                        else
                        {
                            toolCallInternal.function.deltaArguments?.Append(toolCall.function.arguments?.ToString());
                        }
                    }
                }

                if (choice.finish_reason == "tool_calls")
                {
                    if (request.toolCallHandlers == null) break;
                    foreach (var tool in toolCallsMap.Values)
                    {
                        var fn = tool.function;
                        fn.arguments = tool.function.deltaArguments?.ToString();

                        try
                        {
                            var result = await request.toolCallHandlers.Invoke(tool.function.name ?? "", JsonSerializer.Deserialize<JsonElement>(fn.arguments ?? "{}"));
                            if (result != null)
                            {
                                toolCallResults.Add(new OpenAiMessage("tool", result, tool_call_id: tool.id));
                            }
                        }
                        catch (Exception ex)
                        {
                            toolCallResults.Add(new OpenAiMessage("tool", $"Error: {ex.Message}", tool_call_id: tool.id));
                        }
                    }
                }
            }
        }

        request.messages.Add(new OpenAiMessage("assistant", outputBuilder.ToString())
        {
            reasoning_content = outputReasoningBuilder.ToString(),
            tool_calls = [.. toolCallsMap.Values]
        });
        request.messages.AddRange(toolCallResults);
    }

    public async Task<OpenAiResponse> ChatCompletionAsync(
        OpenAiRequest request,
        CancellationToken ct = default)
    {
        httpClient.BaseAddress ??= new Uri(Settings.Url ?? "https://api.openai.com");
        using var requestMessage = PrepareRequest(request);
        using var response = await httpClient.SendAsync(requestMessage, ct);
        await HandleError(response, ct);

        var responseJson = await response.Content.ReadAsStringAsync(ct);
        var r = JsonSerializer.Deserialize(responseJson, OpenAiJsonContext.Default.OpenAiResponse)!;
        var toolCallResults = new List<OpenAiMessage>();
        foreach (var c in r.choices)
        {
            foreach (var tool in c.message?.tool_calls ?? [])
            {
                if (request.toolCallHandlers == null) continue;
                try
                {
                    var result = await request.toolCallHandlers.Invoke(tool.function.name ?? "", JsonSerializer.Deserialize<JsonElement>(tool.function.arguments ?? "{}"));
                    if (result != null)
                    {
                        toolCallResults.Add(new OpenAiMessage("tool", result, tool_call_id: tool.id));
                    }
                }
                catch (Exception ex)
                {
                    toolCallResults.Add(new OpenAiMessage("tool", $"Error: {ex.Message}", tool_call_id: tool.id));
                }
            }
        }

        if (!string.IsNullOrEmpty(r.choices[0].message?.reasoning_content))
            request.onProgress?.Invoke(r.choices[0].message!.reasoning_content!, true);
        if (!string.IsNullOrEmpty(r.choices[0].message?.content))
            request.onProgress?.Invoke(r.choices[0].message!.content!, false);

        request.messages.Add(r.choices[0].message!);
        request.messages.AddRange(toolCallResults);
        return r;
    }

    private static async Task HandleError(HttpResponseMessage response, CancellationToken ct)
    {
        if (!response.IsSuccessStatusCode)
        {
            var errorJson = await response.Content.ReadAsStringAsync(ct);
            var errorResponse = JsonSerializer.Deserialize(errorJson, OpenAiJsonContext.Default.OpenAiErrorResponse);
            throw new HttpRequestException(
                $"OpenAI API error: {(int)response.StatusCode} {response.StatusCode} - {errorResponse?.error?.message ?? "Unknown error"}",
                null,
                response.StatusCode);
        }
    }

    private HttpRequestMessage PrepareRequest(OpenAiRequest request, bool stream = false)
    {
        // Enrich request with default values
        request.temperature ??= Settings.Temperature;
        request.max_tokens ??= Settings.MaxTokens;
        request.reasoning ??= Settings.Reasoning;
        request.instructions ??= Settings.Instructions;
        request.model ??= Settings.Model ?? "gpt-3.5-turbo";
        request.tools ??= Tools;
        request.stream = stream;

        var json = JsonSerializer.SerializeToElement(request, OpenAiJsonContext.Default.OpenAiRequest);
        PrepareRequestData(json);

        var requestContent = new StringContent(json.ToString(), Encoding.UTF8, MediaTypeHeaderValue.Parse("application/json"));

        var requestMessage = new HttpRequestMessage(HttpMethod.Post, "chat/completions");
        requestMessage.Headers.Authorization = new AuthenticationHeaderValue("Bearer", Settings.ApiKey);
        requestMessage.Content = requestContent;
        return requestMessage;
    }

    partial void PrepareRequestData(JsonElement body);
}
