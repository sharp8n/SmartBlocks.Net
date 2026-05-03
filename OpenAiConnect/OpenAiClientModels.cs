#pragma warning disable IDE1006 // Naming Styles

using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SmartBlocks.OpenAiConnect;

public record OpenAiChoice(
    int index,
    // For non-streamed responses
    OpenAiMessage? message = null,
    // For streamed responses
    OpenAiMessage? delta = null,
    string? finish_reason = null);

public record OpenAiUsage(
    int prompt_tokens,
    int completion_tokens,
    int total_tokens);

public record OpenAiErrorDetail(
    string message,
    string type,
    string? param,
    string? code);

public record OpenAiErrorResponse(OpenAiErrorDetail error);

public record OpenAiMessageContent(
    string type,
    string? text = null,
    OpenAiImageUrl? image_url = null);

public record OpenAiImageUrl(
    string url,
    // "auto", "low", or "high"
    string detail = "auto");

public record struct OpenAiRequest(List<OpenAiMessage> messages,
        bool stream = false,
        string? model = null,
        double? temperature = null,
        int? max_tokens = null,
        SmartAiReasoning? reasoning = null,
        string? instructions = null,
        List<JsonElement>? tools = null,
        [property: JsonIgnore] ToolCallDelegate? toolCallHandlers = null,
        [property: JsonIgnore] Action<string, bool>? onProgress = null
        )
{
    public OpenAiRequest NoReasoning()
    {
        reasoning = new SmartAiReasoning { Enabled = false };
        return this;
    }

}

public delegate Task<string?> ToolCallDelegate(string functionName, JsonElement? args);

public record OpenAiResponse(
    string id,
    List<OpenAiChoice> choices,
    OpenAiUsage? usage = null,
    long? created = null,
    string? model = null,
    [property: JsonPropertyName("object")] string? objectType = null
    ) : OpenAiResponseChunk(id, choices, usage);

public record OpenAiResponseChunk(
    string id,
    List<OpenAiChoice> choices,
    OpenAiUsage? usage);

public record ToolCall(
    int index,
    string? id,
    string? type,
    ToolCallFunction function
);

public record struct ToolCallFunction(string? name, string? arguments)
{
    [JsonIgnore]
    public StringBuilder? deltaArguments;
}

public record OpenAiMessage(
        // user, assistant, system, tool
        string role,
        string? content,
        string? reasoning_content = null,
        string? tool_call_id = null,
        List<ToolCall>? tool_calls = null,
        OpenAiMessage? message = null,
        OpenAiMessage? delta = null,
        string? finish_reason = null)
{
    public override string ToString()
    {
        var toolCalls = tool_calls != null ? string.Join(";", tool_calls.Select(t => $" {t.type}: {t.function.name}()")) : null;
        if (reasoning_content != null) return $"{role}: <thinking>{reasoning_content}</thinking>\n{content}{toolCalls}";
        return $"{role}: {content} {toolCalls}";
    }
}

public record UsageInfo(int prompt_tokens, int completion_tokens, int total_tokens);



public class SmartOpenAISettings
{
    public string? ApiKey { get; set; }
    public string? Url { get; set; }
    public string? Model { get; set; }

    public double? Temperature { get; set; }
    public int? MaxTokens { get; set; }
    public string? Instructions { get; set; }
    public SmartAiReasoning? Reasoning { get; set; } = null;
}

public class SmartAiReasoning
{
    // normal, low, high, none
    public string? Effort { get; set; }
    public bool Exclude { get; set; } = false;
    public bool Enabled { get; set; } = false;
}


[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    PropertyNameCaseInsensitive = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
)]
[JsonSerializable(typeof(OpenAiResponse))]
[JsonSerializable(typeof(OpenAiResponseChunk))]
[JsonSerializable(typeof(ToolCallFunction))]
[JsonSerializable(typeof(OpenAiRequest))]
[JsonSerializable(typeof(OpenAiMessage))]
[JsonSerializable(typeof(UsageInfo))]
[JsonSerializable(typeof(OpenAiErrorResponse))]
[JsonSerializable(typeof(OpenAiErrorDetail))]
[JsonSerializable(typeof(OpenAiImageUrl))]
[JsonSerializable(typeof(SmartAiReasoning))]
[JsonSerializable(typeof(SmartOpenAISettings))]
[JsonSerializable(typeof(OpenAiChoice))]
[JsonSerializable(typeof(OpenAiUsage))]
[JsonSerializable(typeof(OpenAiMessageContent))]
[JsonSerializable(typeof(ToolCall))]
[JsonSerializable(typeof(JsonElement))]
internal partial class OpenAiJsonContext : JsonSerializerContext
{
}

#pragma warning restore IDE1006 // Naming Styles
