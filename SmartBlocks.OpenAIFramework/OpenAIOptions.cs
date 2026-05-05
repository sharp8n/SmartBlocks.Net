using System.Text.Json.Serialization;

namespace SmartBlocks.OpenAIFramework;

/// <summary>
/// Configurable options for OpenAI chat completion requests.
/// Maps directly to the optional parameters in the OpenAI API request body.
/// </summary>
public class OpenAIOptions
{
    /// <summary>
    /// Configuration for the <c>thinking</c> parameter (o-series models).
    /// When set, controls whether the model uses thinking/reasoning.
    /// See <see cref="ThinkingConfig"/> for details.
    /// Default: <see langword="null"/> (not sent — API uses its default behavior).
    /// </summary>
    [JsonPropertyName("thinking")]
    public ThinkingConfig? Thinking { get; set; }

    /// <summary>
    /// What sampling temperature to use, between 0 and 2.
    /// Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    /// Default: null (not sent — API uses its default, typically 1.0).
    /// </summary>
    [JsonPropertyName("temperature")]
    public double? Temperature { get; set; }

    /// <summary>
    /// An alternative to sampling with temperature, called nucleus sampling,
    /// where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// Default: null (not sent).
    /// </summary>
    [JsonPropertyName("top_p")]
    public double? TopP { get; set; }

    /// <summary>
    /// The maximum number of tokens that can be generated in the chat completion.
    /// Default: null (not sent — API uses its default).
    /// </summary>
    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; set; }

    /// <summary>
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on
    /// their existing frequency in the text so far, decreasing the model's likelihood
    /// to repeat the same line verbatim.
    /// Default: null (not sent).
    /// </summary>
    [JsonPropertyName("frequency_penalty")]
    public double? FrequencyPenalty { get; set; }

    /// <summary>
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on
    /// whether they appear in the text so far, increasing the model's likelihood
    /// to talk about new topics.
    /// Default: null (not sent).
    /// </summary>
    [JsonPropertyName("presence_penalty")]
    public double? PresencePenalty { get; set; }

    /// <summary>
    /// Up to 4 sequences where the API will stop generating further tokens.
    /// Default: null (not sent).
    /// </summary>
    [JsonPropertyName("stop")]
    public string[]? Stop { get; set; }

    /// <summary>
    /// Reasoning effort for o-series models (o1, o3, etc.).
    /// Supported values: "low", "medium", "high".
    /// Controls how much reasoning the model spends on a response.
    /// Default: null (not sent — API uses its default, typically "medium").
    /// </summary>
    [JsonPropertyName("reasoning_effort")]
    public string? ReasoningEffort { get; set; }

    /// <summary>
    /// Whether to enable image resizing for incoming images.
    /// When <see langword="true"/>, images attached via <see cref="ChatMessage.UserWithImage"/>
    /// will be downscaled to fit within <see cref="MaxWidth"/> x <see cref="MaxHeight"/>
    /// using SkiaSharp before being base64-encoded.
    /// Default: <see langword="true"/>.
    /// </summary>
    [JsonIgnore]
    public bool ResizeImage { get; set; } = true;

    /// <summary>
    /// Maximum width in pixels for image resizing.
    /// Only used when <see cref="ResizeImage"/> is <see langword="true"/>.
    /// Default: 1200.
    /// </summary>
    [JsonIgnore]
    public int MaxWidth { get; set; } = 1200;

    /// <summary>
    /// Maximum height in pixels for image resizing.
    /// Only used when <see cref="ResizeImage"/> is <see langword="true"/>.
    /// Default: 1200.
    /// </summary>
    [JsonIgnore]
    public int MaxHeight { get; set; } = 1200;
}

/// <summary>
/// Configuration for the <c>thinking</c> parameter in OpenAI chat completions.
/// Used with o-series models (o1, o3, etc.) to enable or disable visible
/// thinking/reasoning tokens.
/// Maps to: <c>{ "type": "enabled" | "disabled" }</c>
/// </summary>
/// <example>
/// <code>
/// // Enable thinking
/// var thinking = new ThinkingConfig { Type = ThinkingType.Enabled };
/// 
/// // Disable thinking
/// var thinking = new ThinkingConfig { Type = ThinkingType.Disabled };
/// </code>
/// </example>
public class ThinkingConfig
{
    /// <summary>
    /// Whether to enable thinking. Must be one of <c>"enabled"</c> or <c>"disabled"</c>.
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; set; } = ThinkingType.Enabled;
}

/// <summary>
/// Constants for <see cref="ThinkingConfig.Type"/> values.
/// </summary>
public static class ThinkingType
{
    /// <summary>Enables visible thinking/reasoning tokens.</summary>
    public const string Enabled = "enabled";

    /// <summary>Disables visible thinking/reasoning tokens.</summary>
    public const string Disabled = "disabled";
}
