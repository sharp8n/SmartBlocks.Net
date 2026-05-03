namespace SmartHelpers;

public static partial class ImageHelper
{
    public static string ToBase64DataUri(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Image file not found.", filePath);

        var bytes = File.ReadAllBytes(filePath);
        var ext = Path.GetExtension(filePath).TrimStart('.').ToLowerInvariant();
        return ToBase64DataUri(bytes, ext);
    }

    public static string ToBase64DataUri(byte[] bytes, string ext)
    {
        var base64 = Convert.ToBase64String(bytes);
        return $"data:image/{ext};base64,{base64}";
    }
}
