namespace SmartBlocks.OpenAiConnect.Providers;

public class SmartOpenAiClientProviders
{
    public static string GetUrl(OpenAiClientProvider provider)
    {
        return provider switch
        {
            OpenAiClientProvider.Deepseek => "https://api.deepseek.com",
            OpenAiClientProvider.OpenAI => "https://api.openai.com",
            OpenAiClientProvider.OpenRouter => "https://openrouter.ai/api/v1/",
            OpenAiClientProvider.HolySheepMinimax => "https://api.holysheep.ai/minimax/v1/",
            OpenAiClientProvider.Ollama => "http://api.ollama.com/v1/",
            OpenAiClientProvider.OllamaLocal => "http://localhost:11434/v1/",
            _ => throw new NotImplementedException(),
        };
    }

    public static string GetModel(OpenAiClientProvider provider)
    {
        return provider switch
        {
            OpenAiClientProvider.Deepseek => "deepseek-v4-flash",
            OpenAiClientProvider.OpenAI => "gpt-3.5-turbo",
            OpenAiClientProvider.OpenRouter => "qwen/qwen3.6-27b",
            OpenAiClientProvider.HolySheepMinimax => "MiniMax-M2.7",
            OpenAiClientProvider.Ollama => "kimi-k2.6:cloud",
            OpenAiClientProvider.OllamaLocal => "qwen3.6:27b",
            _ => throw new NotImplementedException(),
        };
    }

    public static string? ApiKey(OpenAiClientProvider provider)
    {
        return provider switch
        {
            OpenAiClientProvider.Deepseek => Environment.GetEnvironmentVariable("DEEPSEEK_API"),
            OpenAiClientProvider.OpenAI => Environment.GetEnvironmentVariable("OPENAI_API"),
            OpenAiClientProvider.OpenRouter => Environment.GetEnvironmentVariable("OPENROUTER_API"),
            OpenAiClientProvider.HolySheepMinimax => Environment.GetEnvironmentVariable("HOLYSHEEP_API"),
            OpenAiClientProvider.Ollama or OpenAiClientProvider.OllamaLocal => Environment.GetEnvironmentVariable("OLLAMA_API"),
            _ => throw new NotImplementedException(),
        };
    }
}

public enum OpenAiClientProvider
{
    Deepseek,
    OpenAI,
    OpenRouter,
    HolySheepMinimax,
    Ollama,
    OllamaLocal,
}
