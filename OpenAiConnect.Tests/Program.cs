using System.Text.Json;
using SmartBlocks.OpenAiConnect;
using SmartBlocks.OpenAiConnect.Providers;

var selectedProvider = OpenAiClientProvider.HolySheepMinimax;

var client = new SmartOpenAiClient(new HttpClient())
{
    Settings = new()
    {
        Url = SmartOpenAiClientProviders.GetUrl(selectedProvider),
        ApiKey = SmartOpenAiClientProviders.ApiKey(selectedProvider),
        Model = SmartOpenAiClientProviders.GetModel(selectedProvider)
    }
};

List<JsonElement> tools = [
    JsonSerializer.Deserialize<JsonElement>("""
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
    """)
];

await RunTool();





// Tests *****************************************************************************


// Generates a response from the OpenAI API
async Task RunGenerate()
{
    var response = await client.ChatCompletionAsync(new()
    {
        messages = [new OpenAiMessage("user", "Just tell OK")]
    });
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine(">>> ", response.choices[0].message?.reasoning_content);
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine("Response: {0}", response.choices[0].message?.content);
    Console.ResetColor();
}

// Generates a response from the OpenAI API, streaming the response
async Task RunGenerateStream()
{
    var response = client.ChatCompletionStreamAsync(new()
    {
        messages = [new OpenAiMessage("user", "Tell me a joke")]
    });
    Console.Write(">>> ");
    bool startContent = false;
    await foreach (var chunk in response)
    {
        var reasoning = chunk.choices[0].delta?.reasoning_content;
        var content = chunk.choices[0]?.delta?.content;
        if (reasoning != null)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.Write(reasoning);
        }
        else
        {
            if (!startContent)
            {
                startContent = true;
                Console.WriteLine();
                Console.ForegroundColor = ConsoleColor.Green;
            }
            Console.Write(content);
        }
    }
    Console.ResetColor();
}

async Task RunTool()
{
    var messages = new List<OpenAiMessage> { new("user", "What is the weather in London?") };
    var response = await client.ChatCompletionAsync(new(messages)
    {
        tools = tools,
        toolCallHandlers = async (function, arg) =>
        {
            return function switch
            {
                "weather" => $"{arg?.GetProperty("location").GetString() ?? "?"} is 24°C +/- 2°C",
                _ => null,
            };
        }
    });

    response = await client.ChatCompletionAsync(new(messages));

    Console.WriteLine("Chat: {0}", string.Join("\n", messages.Select(m => m.ToString())));
}


async Task RunToolStream()
{
    var messages = new List<OpenAiMessage> { new("user", "What is the weather in London?") };
    client.Tools = tools;
    var response = client.ChatCompletionStreamAsync(new(messages)
    {
        toolCallHandlers = async (function, arg) =>
        {
            return function switch
            {
                "weather" => $"{arg?.GetProperty("location").GetString() ?? "?"} is 24°C +/- 2°C",
                _ => null,
            };
        }
    });
    await foreach (var chunk in response) { }
    await client.ChatCompletionAsync(new(messages));

    Console.WriteLine("Chat: {0}", string.Join("\n", messages.Select(m => m.ToString())));
}
