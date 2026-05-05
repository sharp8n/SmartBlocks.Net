using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SmartBlocks.OpenAI;

/// <summary>
/// The only message model. Supports text, images, and tool call/result data.
/// Images are conveyed via base64 data URIs embedded in ContentParts.
/// Tool calls from the assistant are stored as a raw JSON string.
/// </summary>
public class ChatMessage
{
    /// <summary>One of: "system", "user", "assistant", "tool".</summary>
    [JsonPropertyName("role")]
    public string Role { get; set; } = "user";

    /// <summary>
    /// Plain text content. For messages with images this is the text portion.
    /// For tool messages this is the result payload.
    /// </summary>
    [JsonPropertyName("content")]
    public string? Content { get; set; }

    /// <summary>
    /// Reasoning content from the model (e.g. chain-of-thought, internal monologue).
    /// Populated when the API sends a "reasoning" field in the SSE delta.
    /// Commonly used by DeepSeek R1, QwQ, and other reasoning models.
    /// </summary>
    [JsonPropertyName("reasoning_content")]
    public string? ReasoningContent { get; set; }

    /// <summary>Base-64 encoded image bytes (optional).</summary>
    [JsonIgnore]
    public string? ImageBase64 { get; set; }

    /// <summary>MIME type of the image, e.g. "image/jpeg" (required if ImageBase64 is set).</summary>
    [JsonIgnore]
    public string? ImageMimeType { get; set; }

    // ── Tool-call fields (populated on assistant messages) ──────────────────

    /// <summary>
    /// Raw JSON string representing the array of tool_calls returned by the API.
    /// Example: [{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}]
    /// </summary>
    [JsonPropertyName("tool_calls")]
    public string? ToolCallsRaw { get; set; }

    // ── Tool-result fields (populated on tool-role messages) ────────────────

    /// <summary>The tool_call_id this result is for.</summary>
    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; set; }

    /// <summary>The name of the tool that was invoked.</summary>
    [JsonIgnore]
    public string? ToolName { get; set; }

    // ── Factories ───────────────────────────────────────────────────────────

    public static ChatMessage System(string content) => new() { Role = "system", Content = content };
    public static ChatMessage User(string content) => new() { Role = "user", Content = content };
    public static ChatMessage UserWithImage(string text, byte[] imageData, string mimeType) => new()
    {
        Role = "user",
        Content = text,
        ImageBase64 = Convert.ToBase64String(imageData),
        ImageMimeType = mimeType
    };
    public static ChatMessage Assistant(string? content = null) => new() { Role = "assistant", Content = content };
    public static ChatMessage Tool(string toolCallId, string toolName, string result) => new()
    {
        Role = "tool",
        ToolCallId = toolCallId,
        ToolName = toolName,
        Content = result
    };

    // ── Serialization ───────────────────────────────────────────────────────

    /// <summary>
    /// Serialises this message to a JSON string suitable for the OpenAI chat
    /// completions request body.
    /// </summary>
    public string ToRequestJsonString()
    {
        // ── Simple text-only message ────────────────────────────────────
        if (ImageBase64 == null && ToolCallsRaw == null && ToolCallId == null)
        {
            var sb = new StringBuilder();
            sb.Append("{\"role\":\"").Append(Escape(Role)).Append('"');
            sb.Append(",\"content\":\"").Append(Escape(Content ?? "")).Append('"');
            // DeepSeek reasoning/thinking mode requires reasoning_content to be
            // echoed back when the assistant message is included in subsequent requests.
            if (Role == "assistant" && ReasoningContent != null)
                sb.Append(",\"reasoning_content\":\"").Append(Escape(ReasoningContent)).Append('"');
            sb.Append('}');
            return sb.ToString();
        }

        // ── Message with image attachment ───────────────────────────────
        if (ImageBase64 != null)
        {
            var dataUri = "data:" + ImageMimeType + ";base64," + ImageBase64;
            var text = Escape(Content ?? "");
            return "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"" + text + "\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + dataUri + "\"}}]}";
        }

        // ── Tool-role message ───────────────────────────────────────────
        if (Role == "tool" && ToolCallId != null)
        {
            return "{\"role\":\"tool\",\"tool_call_id\":\"" + Escape(ToolCallId) + "\",\"content\":\"" + Escape(Content ?? "") + "\"}";
        }

        // ── Assistant message with tool_calls ───────────────────────────
        if (Role == "assistant" && ToolCallsRaw != null)
        {
            var sb = new StringBuilder();
            sb.Append("{\"role\":\"assistant\"");
            if (Content != null)
                sb.Append(",\"content\":\"").Append(Escape(Content)).Append("\"");
            // DeepSeek reasoning/thinking mode requires reasoning_content to be
            // echoed back when the assistant message is included in subsequent requests.
            if (ReasoningContent != null)
                sb.Append(",\"reasoning_content\":\"").Append(Escape(ReasoningContent)).Append("\"");
            sb.Append(",\"tool_calls\":").Append(ToolCallsRaw);
            sb.Append("}");
            return sb.ToString();
        }

        // Fallback
        return "{\"role\":\"" + Escape(Role) + "\",\"content\":\"" + Escape(Content ?? "") + "\"}";
    }

    private static string Escape(string s) =>
        s.Replace("\\", "\\\\")
         .Replace("\"", "\\\"")
         .Replace("\n", "\\n")
         .Replace("\r", "\\r")
         .Replace("\t", "\\t");
}
